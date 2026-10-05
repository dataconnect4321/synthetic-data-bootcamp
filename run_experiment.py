"""Config-driven runner for the FLIP pipeline (configs/flip.yaml).

Resolves the base config plus one named experiment, applies it to the
existing scripts' module-level settings, and runs the selected stages:

  preprocess -> audit -> report -> demo -> full  (+ eval, with --fold)

The scripts themselves are unchanged and still run standalone. Outputs for
each experiment go to paths.output_dir (default runs/<experiment>/),
together with the resolved config for reproducibility.

Examples:
  python run_experiment.py --list
  python run_experiment.py --experiment smoke
  python run_experiment.py --experiment eps_10 --predict-eps
  python run_experiment.py --experiment eps_10 --stages demo
  python run_experiment.py --experiment adult_paper --fold 0 --stages full,eval
  python run_experiment.py --experiment adult_paper --summarize
"""

import argparse
import copy
import math
import os
import sys
import traceback

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tabsyn-main', 'tabsyn-main'))

STAGES = ['preprocess', 'audit', 'report', 'demo', 'full', 'eval']
N_FOLDS = 3  # the paper's 3-fold cross-validation
SUMMARY_METRICS = ['synth_auc', 'alpha_precision', 'beta_recall', 'sp', 'eo',
                   'eps_identifiability', 'ber', 'a_ncb', 'syn_positive_rate',
                   'real_train_auc']
# Paper Figure 3 (Adult), keyed by (lambda, DP on): used when the config's
# data.paper_reference is 'adult'.
PAPER_ADULT = {
    (0.0, False): dict(synth_auc=.75, alpha_precision=.90, beta_recall=.41, sp=.18,
                       eo=.12, eps_identifiability=.21, ber=.41, a_ncb=.32),
    (4.0, False): dict(synth_auc=.70, alpha_precision=.64, beta_recall=.27, sp=.13,
                       eo=.15, eps_identifiability=.10, ber=.50, a_ncb=.92),
    (0.0, True): dict(synth_auc=.75, alpha_precision=.85, beta_recall=.34, sp=.18,
                      eo=.10, eps_identifiability=.15, ber=.59, a_ncb=.26),
    (4.0, True): dict(synth_auc=.66, alpha_precision=.40, beta_recall=.21, sp=.12,
                      eo=.26, eps_identifiability=.09, ber=.49, a_ncb=.91),
}


# ==========================================================================
# Config resolution
# ==========================================================================
def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _experiment_chain(experiments, name, seen=()):
    """Overrides for `name`, applied after everything it `extends`."""
    if name not in experiments:
        raise SystemExit(f'Unknown experiment {name!r}. '
                         f'Available: {", ".join(experiments)}')
    if name in seen:
        raise SystemExit(f'Circular `extends` involving {name!r}')
    exp = dict(experiments[name] or {})
    parent = exp.pop('extends', None)
    chain = _experiment_chain(experiments, parent, seen + (name,)) if parent else []
    return chain + [exp]


def load_config(path, experiment):
    with open(path) as f:
        raw = yaml.safe_load(f)
    experiments = raw.pop('experiments', {}) or {}
    cfg = raw
    for overrides in _experiment_chain(experiments, experiment):
        cfg = _deep_merge(cfg, overrides)
    cfg['experiment'] = experiment
    cfg['paths']['output_dir'] = cfg['paths']['output_dir'].format(
        experiment=experiment)
    return cfg, experiments


def _flatten_stage1(stage1):
    """stage1's readability subsections -> one flat dict."""
    flat = {}
    for key, val in stage1.items():
        if isinstance(val, dict):
            flat.update(val)
        else:
            flat[key] = val
    return flat


def _to_params(section, defaults, name):
    """Upper-case config keys onto a copy of a script's param dict,
    rejecting keys the script does not know (catches typos)."""
    params = copy.deepcopy(defaults)
    unknown = [k for k in section if k.upper() not in defaults]
    if unknown:
        raise SystemExit(f'Unknown {name} key(s): {unknown}. '
                         f'Valid: {sorted(k.lower() for k in defaults)}')
    for k, v in section.items():
        params[k.upper()] = v
    return params


def _resolve_device(device):
    import torch
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


# ==========================================================================
# Apply config to the scripts' module-level settings
# ==========================================================================
def _abs(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


# ==========================================================================
# Cross-validation folds (the paper's 3-fold protocol)
# ==========================================================================
def make_fold(cfg, fold, fold_dir):
    """Split the preprocessed data (protected groups only) into fold
    `fold`'s train/test CSVs, stratified on target x protected group so
    every fold keeps the group and outcome ratios. Deterministic (seed 42),
    so every experiment sees the same splits."""
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    data = cfg['data']
    df = pd.read_csv(_abs(cfg['paths']['preprocessed_data']), low_memory=False)
    df = df[df[data['sensitive_col']].isin(data['groups'])].reset_index(drop=True)
    strata = (df[data['target_col']].astype(str) + '|'
              + df[data['sensitive_col']].astype(str))
    splits = StratifiedKFold(N_FOLDS, shuffle=True, random_state=42).split(df, strata)
    train_idx, test_idx = list(splits)[fold]
    os.makedirs(fold_dir, exist_ok=True)
    paths = (os.path.join(fold_dir, 'train.csv'), os.path.join(fold_dir, 'test.csv'))
    df.iloc[train_idx].to_csv(paths[0], index=False)
    df.iloc[test_idx].to_csv(paths[1], index=False)
    return paths


def run_eval(cfg, out_dir):
    """Paper metrics for every synthetic CSV in a fold directory."""
    import json
    import pandas as pd
    import paper_eval
    train = pd.read_csv(os.path.join(out_dir, 'train.csv'), low_memory=False)
    test = pd.read_csv(os.path.join(out_dir, 'test.csv'), low_memory=False)
    results = {}
    for name, fname in (('full', 'flip_full_synthetic.csv'),
                        ('demo', 'flip_synthetic.csv')):
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            print(f'Evaluating {fname} ...')
            results[name] = paper_eval.evaluate(train, test, pd.read_csv(path),
                                                cfg['data'])
            print(json.dumps({k: v for k, v in results[name].items()
                              if k in SUMMARY_METRICS}, indent=2))
    if not results:
        raise SystemExit(f'No synthetic CSV in {out_dir}: run --stages full first.')
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)


def summarize(cfg, exp_dir):
    """Mean +/- std over the folds' metrics.json, beside the paper."""
    import json
    import numpy as np
    folds = sorted(d for d in os.listdir(exp_dir) if d.startswith('fold')
                   and os.path.exists(os.path.join(exp_dir, d, 'metrics.json')))
    if not folds:
        raise SystemExit(f'No fold*/metrics.json under {exp_dir}')
    runs = [json.load(open(os.path.join(exp_dir, d, 'metrics.json'))) for d in folds]
    paper = {}
    if cfg['data'].get('paper_reference') == 'adult':
        key = (float(cfg['stage1']['fairness']['lambda_fair']),
               bool(cfg['stage1']['privacy']['use_dp']))
        paper = PAPER_ADULT.get(key, {})
    for pipeline in ('full', 'demo'):
        rows = [r[pipeline] for r in runs if pipeline in r]
        if not rows:
            continue
        print(f'\n{pipeline} pipeline, {len(rows)} fold(s): {", ".join(folds)}')
        print(f'  {"metric":<22}{"ours (mean ± std)":>20}{"paper":>8}')
        for m in SUMMARY_METRICS:
            vals = np.array([r[m] for r in rows if m in r], dtype=float)
            ref = f'{paper[m]:.2f}' if m in paper else ''
            print(f'  {m:<22}{np.nanmean(vals):>11.3f} ± {np.nanstd(vals):.3f}{ref:>8}')


def apply_config(cfg):
    import run_flip_demo as demo
    import run_flip_full_pipeline as full

    paths, data = cfg['paths'], cfg['data']
    out_dir = _abs(paths['output_dir'])
    prep = _abs(paths['preprocessed_data'])

    tradeoff = _to_params(_flatten_stage1(cfg['stage1']),
                          demo.TRADEOFF_PARAMS, 'stage1')
    tradeoff['DEVICE'] = _resolve_device(tradeoff['DEVICE'])
    diffusion = _to_params(cfg['diffusion'], full.DIFFUSION_PARAMS,
                           'diffusion')

    categorical = ([data['sensitive_col'], data['target_col']]
                   + list(data['categorical_cols']))
    for mod in (demo, full):
        mod.DATA_PATH = prep
        mod.SENSITIVE_COL = data['sensitive_col']
        mod.TARGET_COL = data['target_col']
        mod.RACE_GROUPS = list(data['groups'])
        mod.NUMERIC_COLS = list(data['numeric_cols'])
        mod.CATEGORICAL_COLS = categorical
    # Mutate in place: run_flip_full_pipeline imported the same dict object.
    demo.TRADEOFF_PARAMS.clear()
    demo.TRADEOFF_PARAMS.update(tradeoff)
    full.DIFFUSION_PARAMS.clear()
    full.DIFFUSION_PARAMS.update(diffusion)

    demo.REPORT_PATH = os.path.join(out_dir, 'flip_report.html')
    demo.SYNTHETIC_DATA_PATH = os.path.join(out_dir, 'flip_synthetic.csv')
    full.REPORT_PATH = os.path.join(out_dir, 'flip_full_report.html')
    full.SYNTHETIC_DATA_PATH = os.path.join(out_dir, 'flip_full_synthetic.csv')
    return demo, full, out_dir, prep


# ==========================================================================
# Privacy-budget prediction (no training)
# ==========================================================================
def predict_eps(demo, full):
    """Composed ε each pipeline will report, from the same accountants and
    step counts the training loops use. Matches the recorded HMDA run."""
    import numpy as np
    import torch
    from tabsyn.vae.flip_fairness import (RDPAccountant, GroupLevelDPMechanism,
                                          dp_release_mean_std,
                                          compose_rdp_budgets)

    p, d = demo.TRADEOFF_PARAMS, full.DIFFUSION_PARAMS
    if not p['USE_DP']:
        print('stage1.privacy.use_dp is false: no formal privacy guarantee.')
        return
    if not os.path.exists(demo.DATA_PATH):
        raise SystemExit(f'{demo.DATA_PATH} not found: run --stages preprocess first.')

    _, X_cat, _, s_idx, *_ = demo.load_sample(p)
    counts = np.bincount(X_cat[:, s_idx].numpy())
    n = len(X_cat)
    per_group = p['BATCH_SIZE'] // len(counts)
    q = max(min(1.0, per_group / c) for c in counts)
    steps_per_epoch = (p['POISSON_STEPS_PER_EPOCH']
                       or math.ceil(counts.min() / per_group))

    acc = RDPAccountant(p['NOISE_MULTIPLIER'], q, p['DP_DELTA'])
    acc.steps = (p['PHASE1_EPOCHS'] + p['PHASE2_EPOCHS']) * steps_per_epoch
    group = GroupLevelDPMechanism(
        p['MAX_GROUP_GRAD_NORM'], p['GROUP_NOISE_MULTIPLIER'],
        sample_rate=q if p['SAMPLER'] == 'poisson' else None)
    # The batch-mean L_S is batch-coupled, so the group mechanism also runs
    # in Phase 1 (train_flip's pass 2).
    group_epochs = (p['PHASE1_EPOCHS'] + p['PHASE2_EPOCHS']
                    if p.get('LS_VARIANT') == 'batch_mean' else p['PHASE2_EPOCHS'])
    group.steps = group_epochs * steps_per_epoch
    _, _, stats = dp_release_mean_std(torch.zeros(2, 2),
                                      p['STATS_NOISE_MULTIPLIER'],
                                      p['STATS_MAX_NORM'], 2)
    stage1 = [acc.get_rdp_curve()]
    if group.steps:
        stage1.append(group.get_rdp_curve())

    print(f'rows={n}  group sizes={counts.tolist()}  '
          f'max inclusion rate={q:.3f}  steps/epoch={steps_per_epoch}')
    eps_demo, _ = compose_rdp_budgets(stage1 + [stats, stats], p['DP_DELTA'])
    print(f'demo  composed ε ≈ {eps_demo:.1f}  (δ={p["DP_DELTA"]})')

    full_budgets = stage1 + [stats]
    if d['USE_DP']:
        diff = RDPAccountant(d['NOISE_MULTIPLIER'], d['BATCH_SIZE'] / n,
                             d['DP_DELTA'])
        diff.steps = d['EPOCHS'] * math.ceil(n / d['BATCH_SIZE'])
        full_budgets.append(diff.get_rdp_curve())
    eps_full, _ = compose_rdp_budgets(full_budgets, p['DP_DELTA'])
    note = '' if d['USE_DP'] else '  (diffusion NOT private: diffusion.use_dp is false)'
    print(f'full  composed ε ≈ {eps_full:.1f}  (δ={p["DP_DELTA"]}){note}')
    print('Excludes statistics computed outside the accounting '
          '(see ARCHITECTURE.md, known issue #9).')


# ==========================================================================
# Stages
# ==========================================================================
def run_stage(stage, cfg, demo, full, out_dir, prep):
    audit_cols = list(cfg['data']['audit_sensitive_cols'])
    if stage == 'preprocess':
        import preprocess
        preprocess.DATA_PATH = _abs(cfg['paths']['raw_data'])
        preprocess.OUTPUT_PATH = prep
        preprocess.SENSITIVE_COLS = audit_cols
        os.makedirs(os.path.dirname(prep), exist_ok=True)
        preprocess.main()
    elif stage == 'audit':
        import fairness_audit
        fairness_audit.DATA_PATH = prep
        fairness_audit.SENSITIVE_COLS = audit_cols
        fairness_audit.main()
    elif stage == 'report':
        import fairness_report
        fairness_report.DATA_PATH = prep
        fairness_report.REPORT_PATH = os.path.join(out_dir, 'fairness_report.html')
        fairness_report.SENSITIVE_COLS = audit_cols
        fairness_report.main()
    elif stage == 'demo':
        demo.main()
    elif stage == 'full':
        argv, sys.argv = sys.argv, [sys.argv[0]]   # full.main() parses argv
        try:
            full.main()
        finally:
            sys.argv = argv
    elif stage == 'eval':
        if not os.path.exists(os.path.join(out_dir, 'test.csv')):
            raise SystemExit('The eval stage needs a held-out test fold: '
                             'pass --fold 0|1|2.')
        run_eval(cfg, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=os.path.join(HERE, 'configs', 'flip.yaml'))
    parser.add_argument('--experiment', default='baseline')
    parser.add_argument('--stages', default=','.join(s for s in STAGES if s != 'eval'),
                        help=f'comma-separated subset of {STAGES}')
    parser.add_argument('--predict-eps', action='store_true',
                        help='print the composed ε this experiment will report, then exit')
    parser.add_argument('--list', action='store_true', help='list experiments and exit')
    parser.add_argument('--fold', type=int, choices=range(N_FOLDS),
                        help='train on the other folds and write outputs to '
                             'runs/<experiment>/fold<k>/ (the eval stage needs this)')
    parser.add_argument('--summarize', action='store_true',
                        help='mean ± std of every fold\'s metrics.json, then exit')
    args = parser.parse_args()

    cfg, experiments = load_config(args.config, args.experiment)
    if args.list:
        for name, exp in experiments.items():
            parent = (exp or {}).get('extends')
            print(f'{name}' + (f'  (extends {parent})' if parent else ''))
        return
    if args.summarize:
        summarize(cfg, _abs(cfg['paths']['output_dir']))
        return
    if args.fold is not None:
        fold_dir = os.path.join(_abs(cfg['paths']['output_dir']), f'fold{args.fold}')
        train_path, _ = make_fold(cfg, args.fold, fold_dir)
        cfg['paths']['output_dir'] = fold_dir
        cfg['paths']['preprocessed_data'] = train_path
        if 'preprocess' in args.stages:
            raise SystemExit('--fold cannot be combined with the preprocess stage')

    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f'Unknown stage(s) {bad}; choose from {STAGES}')

    demo, full, out_dir, prep = apply_config(cfg)
    if args.predict_eps:
        predict_eps(demo, full)
        return

    os.makedirs(out_dir, exist_ok=True)
    resolved = dict(cfg, resolved_stage1=dict(demo.TRADEOFF_PARAMS),
                    resolved_diffusion=dict(full.DIFFUSION_PARAMS))
    with open(os.path.join(out_dir, 'resolved_config.yaml'), 'w') as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    print(f'Experiment {args.experiment!r} -> {out_dir}')

    # A failing stage does not stop later ones (e.g. the demo's metrics can
    # crash after its synthetic CSV is already saved - known issue #3).
    failed = []
    for stage in stages:
        print(f'\n===== stage: {stage} =====')
        try:
            run_stage(stage, cfg, demo, full, out_dir, prep)
        except Exception:
            traceback.print_exc()
            failed.append(stage)
            print(f'!!! stage {stage!r} failed; continuing')

    print(f'\nExperiment {args.experiment!r} finished. Outputs: {out_dir}')
    if failed:
        raise SystemExit(f'Failed stage(s): {failed}')


if __name__ == '__main__':
    main()
