"""Fast Phase-2 (disentanglement) experiments on one fold, no DP.

Phase 1 is trained once and checkpointed (runs/<experiment>/fold<k>/
phase1.pt); each lambda then trains only Phase 2 from that checkpoint and
is probed directly on the VAE, without diffusion:

  - latent BER: balanced error rate of predicting the protected attribute
    from the encoder means mu (3-fold out of fold), with all tokens and
    without the protected attribute's own token. 0.5 = no group
    information, which is what FLIP's L_fair targets.
  - CKA^T per stage (latent, detokenizer, decoder): what the loss sees.

Usage:
  python phase2_sweep.py --experiment adult_paper --fold 0 --lambdas 0 4 12 32
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tabsyn-main', 'tabsyn-main'))

import run_experiment
from fairness_eval import _clf
from tabsyn.vae.flip_fairness import disentanglement_loss

PROBE_ROWS_PER_GROUP = 2000  # balanced sample for the CKA^T probe


@torch.no_grad()
def _stages(model, X_num, X_cat, device, batch=4096):
    """Encoder means for all rows, batched."""
    mus = []
    for i in range(0, len(X_num), batch):
        st = model.forward_with_stages(X_num[i:i + batch].to(device),
                                       X_cat[i:i + batch].to(device))
        mus.append(st['mu_z'].cpu())
    return torch.cat(mus).numpy()  # (n, tokens, d_token)


def latent_ber(mu, groups, seed=0):
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    pred = cross_val_predict(_clf(seed), mu, groups, cv=cv)
    return float(1 - balanced_accuracy_score(groups, pred))


@torch.no_grad()
def stage_cka(model, X_num, X_cat, s_idx, device, seed=0):
    """CKA^T between groups at each intervention stage, on a balanced
    sample (CKA = -disentanglement_loss)."""
    rng = np.random.default_rng(seed)
    g = X_cat[:, s_idx].numpy()
    idx = np.concatenate([rng.choice(np.where(g == v)[0],
                                     min(PROBE_ROWS_PER_GROUP, (g == v).sum()),
                                     replace=False) for v in np.unique(g)])
    st = model.forward_with_stages(X_num[idx].to(device), X_cat[idx].to(device))
    group = X_cat[idx, s_idx].long().to(device)
    n = len(idx)
    out = {'latent': -disentanglement_loss(st['mu_z'].reshape(n, -1), group),
           'decoder': -disentanglement_loss(st['h'].reshape(n, -1), group)}
    feats = []
    if st.get('recon_num_pre') is not None:
        feats += [st['recon_num_pre'][:, j, :] for j in range(st['recon_num_pre'].shape[1])]
    feats += [c for i, c in enumerate(st['recon_x_cat']) if c is not None and i != s_idx]
    vals = [v for v in (disentanglement_loss(f, group) for f in feats) if v is not None]
    if vals:
        out['detokenizer'] = -torch.stack(vals).mean()
    return {k: float(v) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=os.path.join(HERE, 'configs', 'flip.yaml'))
    parser.add_argument('--experiment', default='adult_paper')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0, 4, 12, 32])
    args = parser.parse_args()

    cfg, _ = run_experiment.load_config(args.config, args.experiment)
    fold_dir = os.path.join(run_experiment._abs(cfg['paths']['output_dir']),
                            f'fold{args.fold}')
    train_path, _ = run_experiment.make_fold(cfg, args.fold, fold_dir)
    cfg['paths']['output_dir'] = fold_dir
    cfg['paths']['preprocessed_data'] = train_path
    demo, _, _, _ = run_experiment.apply_config(cfg)

    base = copy.deepcopy(demo.TRADEOFF_PARAMS)
    if base['USE_DP']:
        raise SystemExit('phase2_sweep runs without DP: pick a non-DP experiment')
    base['PHASE1_CHECKPOINT'] = os.path.join(fold_dir, 'phase1.pt')
    device = torch.device(base['DEVICE'])
    X_num, X_cat, categories, s_idx, *_ = demo.load_sample(base)
    groups = X_cat[:, s_idx].numpy()

    # Tokens: [constant, numerics..., categoricals...]. The protected
    # attribute is itself an encoder input, so its own token trivially
    # reveals it; 'latent_ber_without_s' drops that token.
    s_token = 1 + X_num.shape[1] + s_idx

    def probe(model):
        model.eval()
        mu = _stages(model, X_num, X_cat, device)
        rest = np.delete(mu, s_token, axis=1)
        return {'latent_ber': latent_ber(mu.reshape(len(mu), -1), groups),
                'latent_ber_without_s': latent_ber(rest.reshape(len(rest), -1), groups),
                'cka': stage_cka(model, X_num, X_cat, s_idx, device)}

    results = {}
    # Phase 1 only: trains and saves the checkpoint, or loads it.
    model, _, _ = demo.train_flip(X_num, X_cat, categories, s_idx,
                                  dict(base, PHASE2_EPOCHS=0), device)
    results['phase1'] = probe(model)
    print(f'phase1  {results["phase1"]}', flush=True)

    for lam in args.lambdas:
        p = dict(base, LAMBDA_FAIR=lam)
        model, _, _ = demo.train_flip(X_num, X_cat, categories, s_idx, p, device)
        results[f'lambda_{lam:g}'] = probe(model)
        print(f'lambda={lam:g}  {results[f"lambda_{lam:g}"]}', flush=True)

    with open(os.path.join(fold_dir, 'phase2_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n{"run":<12}{"BER all":>9}{"BER w/o S":>11}{"CKA latent":>12}'
          f'{"CKA detok":>11}{"CKA decoder":>13}')
    for name, r in results.items():
        c = r['cka']
        print(f'{name:<12}{r["latent_ber"]:>9.3f}{r["latent_ber_without_s"]:>11.3f}'
              f'{c.get("latent", np.nan):>12.3f}{c.get("detokenizer", np.nan):>11.3f}'
              f'{c.get("decoder", np.nan):>13.3f}')


if __name__ == '__main__':
    main()
