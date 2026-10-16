"""AIM (McKenna et al., PVLDB 2022) with fairness and realism applied to its
noisy measurements - the "AIM hybrid" (see WRITEUP.md).

AIM measures 1- and 2-way marginals under zCDP (Gaussian noise on the count
tables) and fits a graphical model to them with Mirror Descent (Private-PGM).
FairAIM keeps AIM's measuring loop unchanged and replaces the final fit and
sampling with (FairAIM.synthesize):

1. proxy masking - category levels that are near-deterministic for one sex
   (P(sex | level) >= 0.95, >= 1% of rows, pointing to different sexes) are
   merged into one level in every noisy table (detect_proxy_classes_model,
   merge_categories);
2. repair (Feldman et al., KDD 2015, on the marginals) - per sex group, a
   transport plan moves P(X | sex) to the pooled P(X) for every column but
   sex (monotone coupling for binned numerics, minimal-move coupling for
   categories); tables with sex are moved exactly, the others with the plans
   mixed by P(sex | x); then the model is refitted (repair_plans,
   repair_measurements);
3. consistency rules - a pre-specified test on the noisy non-sex 2-way tables
   (tested_pair_rules);
4. pair redraw - rows breaking a rule get only the broken pair redrawn from
   the model conditional, then drifted 1-way marginals are re-matched
   (gibbs_fix, match_marginals).

Every step reads only the noisy measurements or the model fitted to them:
post-processing, so none of it costs privacy budget.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'vendor'))
from aim_reference import AIM, filter_candidates, compile_workload  # noqa: E402
from mbi import Domain, LinearMeasurement, estimation  # noqa: E402


def coupling(p, q, ordinal):
    """Row-stochastic transport plan T[x, x'] from distribution p to q."""
    p = np.clip(np.asarray(p, np.float64), 0, None)
    q = np.clip(np.asarray(q, np.float64), 0, None)
    p, q = p / p.sum(), q / q.sum()
    if ordinal:   # monotone (north-west corner) coupling = 1-D optimal transport
        cp, cq = np.cumsum(p), np.cumsum(q)
        hi = np.minimum(cp[:, None], cq[None, :])
        lo = np.maximum((cp - p)[:, None], (cq - q)[None, :])
        mass = np.clip(hi - lo, 0, None)
    else:         # 0-1 cost: keep min(p, q), move the surplus to the deficits
        keep = np.minimum(p, q)
        r, d = p - keep, q - keep
        mass = np.diag(keep)
        if d.sum() > 1e-12:
            mass = mass + np.outer(r, d) / d.sum()
    T = np.eye(len(p))
    ok = p > 1e-12
    T[ok] = mass[ok] / p[ok, None]
    return T


def repair_plans(model, domain, cols, ordinal, sex_col='sex', tau=1.0,
                 target='pooled'):
    """{col: (T [S, k, k] per group, M [k, k] mixed)} from the model's P(S, X).
    target 'average': equal-weight average of the group conditionals
    (Feldman's median-style target); 'pooled': the overall P(X), so each
    column's overall distribution is kept (e.g. the 24% income rate)."""
    plans, target_kind = {}, target
    for c in cols:
        f = model.project((sex_col, c))
        if tuple(f.domain.attributes) != (sex_col, c):
            f = f.transpose((sex_col, c))
        pj = np.clip(np.asarray(f.datavector(flatten=False), np.float64), 0, None)
        pj = pj / pj.sum()                                  # [S, k]
        cond = pj / np.maximum(pj.sum(1, keepdims=True), 1e-12)
        target = pj.sum(0) if target_kind == 'pooled' else cond.mean(0)
        eye = np.eye(domain[c])
        T = np.stack([(1 - tau) * eye + tau * coupling(cond[s], target, c in ordinal)
                      for s in range(pj.shape[0])])
        s_given_x = pj / np.maximum(pj.sum(0, keepdims=True), 1e-12)   # [S, k]
        M = np.einsum('sx,sxy->xy', s_given_x, T)
        plans[c] = (T, M)
    return plans


def _apply(y, axis, K):
    """y'[..., x', ...] = sum_x y[..., x, ...] K[x, x'] along `axis`."""
    return np.moveaxis(np.tensordot(y, K, axes=([axis], [0])), -1, axis)


def repair_measurements(measurements, domain, plans, sex_col='sex'):
    """Push every noisy measurement through the repair plans (see docstring)."""
    out = []
    for m in measurements:
        axes = [(i, c) for i, c in enumerate(m.clique) if c in plans]
        if not axes:
            out.append(m)
            continue
        y = np.asarray(m.noisy_measurement, np.float64).reshape(
            [domain[c] for c in m.clique])
        if sex_col in m.clique:
            ax = m.clique.index(sex_col)
            parts = []
            for s in range(y.shape[ax]):
                ys = np.take(y, s, axis=ax)
                for i, c in axes:
                    ys = _apply(ys, i - (i > ax), plans[c][0][s])
                parts.append(ys)
            y = np.stack(parts, axis=ax)
        else:
            for i, c in axes:
                y = _apply(y, i, plans[c][1])
        out.append(LinearMeasurement(y.reshape(-1), m.clique, stddev=m.stddev))
    return out


def _measured_sex_table(measurements, domain, sex_col, c):
    """Inverse-variance average of the noisy [sex, c] tables, as [S, k], or
    None if AIM never measured that pair."""
    ys, ws = [], []
    for m in measurements:
        if set(m.clique) == {sex_col, c} and len(m.clique) == 2:
            y = np.asarray(m.noisy_measurement, np.float64).reshape(
                [domain[a] for a in m.clique])
            ys.append(y if m.clique[0] == sex_col else y.T)
            ws.append(1.0 / m.stddev ** 2)
    if not ys:
        return None
    return sum(w * y for w, y in zip(ws, ys)) / sum(ws)


def detect_proxy_classes_model(model, domain, sex_col='sex', threshold=0.95,
                               min_support=0.01, exclude=(),
                               measurements=None):
    """Sex-specific category levels, found from the NOISY measurements
    (post-processing) with the same dataset-agnostic rule as
    tabsyn/vae/proxy_decouple.detect_proxy_classes: a level is flagged
    when it has >= min_support of the rows and max_s P(S = s | level) >=
    threshold; a column's flagged levels are merged only if they point to
    at least two different groups (Adult: Husband -> male, Wife -> female).
    P(S, X) is read from the measured noisy [sex, X] table when AIM
    measured it (the fitted model smooths near-zero cells), else from the
    model fitted to the measurements. Returns [(column, [codes...]), ...]."""
    out = []
    for c in domain.attributes:
        if c == sex_col or c in exclude:
            continue
        pj = (_measured_sex_table(measurements, domain, sex_col, c)
              if measurements is not None else None)
        if pj is None:
            f = model.project((sex_col, c))
            if tuple(f.domain.attributes) != (sex_col, c):
                f = f.transpose((sex_col, c))
            pj = np.asarray(f.datavector(flatten=False), np.float64)
        pj = np.clip(pj, 0, None)
        pj = pj / pj.sum()
        share = pj.sum(0)
        purity = pj / np.maximum(share, 1e-12)
        flagged = {k: int(purity[:, k].argmax()) for k in range(len(share))
                   if share[k] >= min_support
                   and purity[:, k].max() >= threshold}
        if len(flagged) >= 2 and len(set(flagged.values())) >= 2:
            out.append((c, sorted(flagged)))
    return out


def merge_categories(measurements, domain, merges):
    """Merge each detected class into ONE new level, the last code of its
    column, by summing the merged cells of every noisy table that contains
    the column (a linear post-processing of the counts). Returns the new
    measurements and domain."""
    if not merges:
        return measurements, domain
    maps, sizes = {}, dict(zip(domain.attributes, domain.shape))
    for c, members in merges:
        keep = [k for k in range(sizes[c]) if k not in members]
        M = np.zeros((sizes[c], len(keep) + 1))
        for new, old in enumerate(keep):
            M[old, new] = 1
        M[members, len(keep)] = 1
        maps[c], sizes[c] = M, len(keep) + 1
    new_dom = Domain(list(domain.attributes),
                     [sizes[c] for c in domain.attributes])
    out = []
    for m in measurements:
        y = np.asarray(m.noisy_measurement, np.float64).reshape(
            [domain[c] for c in m.clique])
        for i, c in enumerate(m.clique):
            if c in maps:
                y = _apply(y, i, maps[c])
        out.append(LinearMeasurement(y.reshape(-1), m.clique,
                                     stddev=m.stddev))
    return out, new_dom


def tested_pair_rules(measurements, domain, sex_col='sex', alpha=0.05,
                      kappa=0.1):
    """Consistency rules from the noisy measurements. A cell (a, b) of a
    measured noisy non-sex 2-way table is impossible when, with
    family-wise confidence 1 - alpha (Bonferroni over every cell tested),
    its true count is below kappa times what independence of the table's
    own noisy margins predicts: y + z * sigma < kappa * E with
    z = Phi^-1(1 - alpha / m). alpha and kappa are fixed in advance and
    do not depend on the dataset. Returns [(col_a, col_b, a, b), ...]."""
    from scipy.stats import norm
    tables, seen = [], set()
    for m in measurements:
        if len(m.clique) != 2 or sex_col in m.clique or m.clique in seen:
            continue
        seen.add(m.clique)
        a, b = m.clique
        y = np.asarray(m.noisy_measurement, np.float64).reshape(
            domain[a], domain[b])
        tables.append((a, b, y, m.stddev))
    n_cells = sum(y.size for _, _, y, _ in tables)
    z = norm.ppf(1 - alpha / max(n_cells, 1))
    rules = []
    for a, b, y, sd in tables:
        r, c = np.clip(y.sum(1), 0, None), np.clip(y.sum(0), 0, None)
        exp = np.outer(r, c) / max(y.sum(), 1e-12)
        bad = y + z * sd < kappa * exp
        rules += [(a, b, int(i), int(j)) for i, j in zip(*np.nonzero(bad))]
    return rules


def violates(codes, rules):
    """Boolean mask of rows of {attribute: codes} breaking any pair rule."""
    n = len(next(iter(codes.values())))
    bad = np.zeros(n, bool)
    for a, b, i, j in rules:
        bad |= (np.asarray(codes[a]) == i) & (np.asarray(codes[b]) == j)
    return bad


def _log_conditional(model, codes, cols, rows):
    """Unnormalized log P(cols | rest of the row) for `rows`, from the
    model's log-potentials: the sum of every clique potential that touches
    `cols`, evaluated at the row's other values. Shape [len(rows), k_1, ...]
    (one axis per column in `cols`, in order)."""
    sizes = [model.domain[c] for c in cols]
    out = np.zeros([len(rows)] + sizes)
    for cl, f in model.potentials.tables.items():
        if not set(cl) & set(cols):
            continue
        attrs = list(f.domain.attributes)
        vals = np.asarray(f.values, np.float64)
        idx = []
        for a in attrs:
            if a in cols:
                shape = [1] * (len(cols) + 1)
                shape[1 + cols.index(a)] = model.domain[a]
                idx.append(np.arange(model.domain[a]).reshape(shape))
            else:
                idx.append(np.asarray(codes[a])[rows].reshape(
                    [-1] + [1] * len(cols)))
        out = out + vals[tuple(idx)]
    return out


def _banned(rules, codes, cols, rows, domain):
    """Boolean [len(rows), k_1, ...]: value combinations of `cols` that
    would break a rule given each row's other values."""
    sizes = [domain[c] for c in cols]
    ban = np.zeros([len(rows)] + sizes, bool)
    for a, b, i, j in rules:
        for (x, vx), (y, vy) in (((a, i), (b, j)), ((b, j), (a, i))):
            if x not in cols:
                continue
            sel = [slice(None)] * (len(cols) + 1)
            sel[1 + cols.index(x)] = vx
            if y in cols:
                sel[1 + cols.index(y)] = vy
                ban[tuple(sel)] = True
            else:
                hit = np.asarray(codes[y])[rows] == vy
                sel[0] = hit
                ban[tuple(sel)] = True
            if y in cols:
                break     # pair handled once when both columns are redrawn
    return ban


def _draw(logp, ban, rng, weight=None):
    """Draw one flat cell per row from exp(logp), banned cells excluded,
    optionally reweighted. Rows with nothing allowed keep -1."""
    n = logp.shape[0]
    lp = logp.reshape(n, -1).copy()
    lp[ban.reshape(n, -1)] = -np.inf
    if weight is not None:
        with np.errstate(divide='ignore'):
            lp = lp + np.log(weight.reshape(1, -1))
    m = lp.max(1, keepdims=True)
    ok = np.isfinite(m[:, 0])
    p = np.exp(lp - np.where(ok[:, None], m, 0))
    p[~ok] = 0
    cum = p.cumsum(1)
    u = rng.random(n) * np.where(ok, cum[:, -1], 1)
    pick = (cum < u[:, None]).sum(1)
    return np.where(ok, np.minimum(pick, lp.shape[1] - 1), -1)


def gibbs_fix(model, codes, rules, rng, max_rounds=10):
    """Repair rule-breaking rows in place instead of discarding them: for
    each broken rule (A = a, B = b), redraw the PAIR (A, B) jointly from the
    model's conditional given the rest of the row, excluding every value
    combination that would break a rule. The rest of the row (and its
    repaired values) is kept, and no column has to be singled out as the
    offender. Returns the number of rows changed."""
    changed = np.zeros(len(next(iter(codes.values()))), bool)
    pairs = sorted({(a, b) for a, b, _, _ in rules})
    for _ in range(max_rounds):
        any_bad = False
        for a, b in pairs:
            sub = [r for r in rules if (r[0], r[1]) == (a, b)]
            rows = np.nonzero(violates(codes, sub))[0]
            if not len(rows):
                continue
            any_bad = True
            logp = _log_conditional(model, codes, [a, b], rows)
            ban = _banned(rules, codes, [a, b], rows, model.domain)
            cell = _draw(logp, ban, rng)
            ok = cell >= 0
            kb = model.domain[b]
            codes[a][rows[ok]] = cell[ok] // kb
            codes[b][rows[ok]] = cell[ok] % kb
            changed[rows[ok]] = True
        if not any_bad:
            break
    return int(changed.sum())


def match_marginals(model, codes, rules, rng, cols=None, tol=0.002):
    """Marginal check after the redraws: for every column whose 1-way
    distribution drifted from the model's by more than `tol` (total
    variation), move the excess rows of over-represented values to the
    under-represented ones, redrawing just that column from its conditional
    restricted to the deficit values (banned values excluded). Returns
    {column: (TV before, TV after)}."""
    report = {}
    n = len(next(iter(codes.values())))
    for c in cols or model.domain.attributes:
        k = model.domain[c]
        target = np.clip(np.asarray(model.project((c,)).datavector(),
                                    np.float64), 0, None)
        target = target / target.sum() * n
        cur = np.bincount(codes[c], minlength=k).astype(float)
        tv0 = 0.5 * np.abs(cur - target).sum() / n
        if tv0 <= tol:
            continue
        excess = np.floor(np.clip(cur - target, 0, None)).astype(int)
        picks = [rng.choice(np.nonzero(codes[c] == v)[0], e, replace=False)
                 for v, e in enumerate(excess) if e > 0]
        if not picks:
            continue
        rows = np.concatenate(picks)
        deficit = np.clip(target - cur, 0, None)
        logp = _log_conditional(model, codes, [c], rows)
        ban = _banned(rules, codes, [c], rows, model.domain)
        new = _draw(logp, ban, rng, weight=deficit / deficit.sum())
        ok = new >= 0
        codes[c][rows[ok]] = new[ok]
        cur = np.bincount(codes[c], minlength=k).astype(float)
        report[c] = (round(float(tv0), 4),
                     round(float(0.5 * np.abs(cur - target).sum() / n), 4))
    return report


class FairAIM(AIM):
    """AIM whose final fit and sampling use the hybrid pipeline above."""

    def __init__(self, epsilon, delta, sex_col='sex', **kw):
        super().__init__(epsilon, delta, **kw)
        self.sex_col = sex_col

    def run(self, data, workload, num_synth_rows=None, initial_cliques=None):
        # identical to aim_reference.AIM.run except that the noisy
        # measurements are kept for synthesize() instead of being sampled
        rounds = self.rounds or 16 * len(data.domain)
        candidates = compile_workload(workload)
        answers = {cl: data.project(cl).datavector() for cl in candidates}
        if not initial_cliques:
            initial_cliques = [cl for cl in candidates if len(cl) == 1]
        oneway = [cl for cl in candidates if len(cl) == 1]
        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)
        measurements = []
        rho_used = len(oneway) * 0.5 / sigma ** 2
        for cl in initial_cliques:
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, x.size)
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))
        model = estimation.MirrorDescent().estimate(
            data.domain, measurements, iters=self.max_iters)
        potentials = None
        terminate = False
        while not terminate:
            if self.rho - rho_used < 2 * (0.5 / sigma ** 2 + 1.0 / 8 * epsilon ** 2):
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True
            rho_used += 1.0 / 8 * epsilon ** 2 + 0.5 / sigma ** 2
            size_limit = self.max_model_size * rho_used / self.rho
            small = filter_candidates(candidates, model, size_limit)
            cl = self.worst_approximated(small, answers, model, epsilon, sigma)
            n = data.domain.size(cl)
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))
            z = model.project(cl).datavector()
            pcliques = list(set(M.clique for M in measurements))
            potentials = model.potentials.expand(pcliques)
            model = estimation.MirrorDescent().estimate(
                data.domain, measurements, iters=self.max_iters,
                warm_start=potentials)
            w = model.project(cl).datavector()
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                sigma /= 2
                epsilon *= 2
            print(f'  budget {rho_used / self.rho:.2f}  measured {cl}',
                  flush=True)
        self.measured_cliques = [M.clique for M in measurements]
        self.measurements, self.potentials = measurements, potentials

    def _fit(self, domain, measurements, warm=True):
        return estimation.MirrorDescent().estimate(
            domain, measurements, iters=self.max_iters,
            warm_start=self.potentials if warm else None)

    def synthesize(self, domain, num_synth_rows, repair_cols, ordinal=(),
                   tau=1.0, mask=True, fix=True, exclude=()):
        """Masking, repair, refit, sampling and pair redraw on the noisy
        measurements of the last run() (or of saved ones). Post-processing:
        any number of variants share one run() at no extra privacy cost.
        Returns (model, {attribute: codes}); the domain, merges, rules and
        fix statistics are left in self.last_info."""
        meas, dom, merges = self.measurements, domain, []
        if mask:
            raw = self._fit(domain, meas)
            merges = detect_proxy_classes_model(
                raw, domain, self.sex_col, exclude=exclude, measurements=meas)
            meas, dom = merge_categories(meas, domain, merges)
        rules = tested_pair_rules(meas, dom, self.sex_col) if fix else []
        fair = meas
        if tau > 0 and repair_cols:
            base = self._fit(dom, meas, warm=not merges)
            plans = repair_plans(base, dom, repair_cols, ordinal,
                                 self.sex_col, tau)
            fair = repair_measurements(fair, dom, plans, self.sex_col)
        model = self._fit(dom, fair, warm=not merges)
        codes = {k: np.asarray(v).astype(np.int64) for k, v in
                 model.synthetic_data(rows=num_synth_rows).to_dict().items()}
        info = {'domain': dom, 'merges': merges, 'rules': rules,
                'broken_before_fix': float(violates(codes, rules).mean())
                if rules else 0.0}
        if rules:
            rng = np.random.default_rng(self.prng.randint(2 ** 31))
            info['rows_redrawn'] = gibbs_fix(model, codes, rules, rng)
            info['marginal_fix'] = match_marginals(model, codes, rules, rng)
            info['broken_after_fix'] = float(violates(codes, rules).mean())
        self.last_info = info
        return model, codes
