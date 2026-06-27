# MarketDesk — Quant Research & Architecture Blueprint

**Status:** design document · **Audience:** maintainers + future research collaborators
**Scope:** how to evolve MarketDesk from "a model that produces scores" into "a
research system that proves a signal is real before any capital touches it."

This document is opinionated. Where the brief that prompted it proposed something
premature or risky for this system's actual scale, that is called out explicitly
rather than rubber-stamped. The job of this doc is to sequence work by
*evidence produced per unit of effort*, not to maximize the number of techniques
in flight.

---

## 0. Executive summary — read this first

Three findings reframe everything below. If you only act on this section, act on these.

### Finding 1 — There are two disconnected systems, and the wrong one has the rigor

| | Serving path | Research path |
|---|---|---|
| Module | `marketdesk/model.py`, `features.py`, `service.py` | `marketdesk/quant/*` |
| Shape | **Cross-sectional** — ranks ~500 names each day | **Per-symbol** — one OHLCV series at a time |
| What it produces | the `score`, `ml_pct`, `signal` users see | research artifacts, never served |
| Backtest | none of its own | `quant/backtest.py` — **single-asset** position series |
| Validation | Rank IC / AUC / accuracy in the model card | purged walk-forward, drift, HPO, intervals |

The production model — the one whose numbers users act on — has the *least*
validation. The research toolkit's purged CV, cost-aware backtest, drift
detection, and HPO operate on a **different problem shape** (one stock's
time series) than the thing in production (a daily cross-sectional rank). You
cannot today answer "does the served signal survive costs?" because the backtest
engine literally cannot consume a cross-sectional ranker's output.

**This is the highest-leverage gap in the system.** Every metric and model idea
below is gated on closing it.

### Finding 2 — The missing primitive is a cross-sectional portfolio backtest

`run_backtest(prices: Series, signals: Series)` is a single-instrument engine. A
universe ranker needs a **portfolio** backtest: each rebalance, form a book
(e.g. long top-decile / short bottom-decile, or long-only top-N), size positions,
charge per-name turnover costs, and roll the equity curve of the *book*, not of
one ticker. Until this exists, "Phase 2/3" (regime models, meta-ensembles, alt
data) optimize a number (Rank IC) that is not yet known to convert into
after-cost return. **Build the bridge before building more models.**

### Finding 3 — Statistical honesty is unaccounted, and you already have the exposure

`quant/hpo.py` runs Optuna with many trials, objective = purged-CV Sharpe. Picking
the best of *N* trials inflates the in-sample Sharpe by construction. Nothing in
the system currently deflates for that. Before trusting any HPO-selected config,
you need **Deflated Sharpe Ratio** (penalize for trial count) and **PBO**
(probability of backtest overfitting). This is cheap to add and is the single
most important "is this real?" guardrail.

### The re-sequenced first move

> Do **not** start with regime models, meta-ensembles, or alternative data.
> Start by wrapping the **existing serving model** in a cross-sectional
> evaluation harness: portfolio backtest + IC decomposition + deflated Sharpe +
> cost breakeven. ~2-3 weeks. It will tell you whether you have a business before
> you spend months making the model fancier.

Everything else is downstream of having that harness.

---

## 1. System grounding — what actually exists

Verified against the codebase (not the brief):

**Serving model (`marketdesk/model.py`)** — cross-sectional, rank-normalized
features (market-relative momentum, risk-adjusted momentum, vol-regime ratio,
normalized MACD histogram), isotonic-calibrated 3-learner ensemble (RF + ExtraTrees
+ HistGBM), purged/embargoed walk-forward CV (21d embargo), recency-weighted.
Model card: `test_auc`, `test_accuracy`, `test_ic`, `cv_auc`, `cv_ic`,
`feature_importances`.

**Research toolkit (`marketdesk/quant/`)** — public API in `quant/__init__.py`:
- `validation.py` — `DataValidator` (completeness, outliers, OHLC consistency, gaps)
- `features.py` — `FeatureEngineer` (winsorize, corr-prune, train-only scaling)
- `targets.py` — `build_targets` (direction, return, vol, risk-adjusted, quantile, regime)
- `splits.py` — `PurgedWalkForward`, `time_series_split`
- `backtest.py` — `run_backtest` (**single-asset**; costs, slippage, Sharpe/Sortino/Calmar/MDD/VaR/CVaR/profit-factor/turnover)
- `drift.py` — `population_stability_index`, `concept_drift`
- `models.py` — `model_zoo` (quantile GBM, RF, conformal ElasticNet/SVR; XGB/LGBM/CatBoost auto-register), every model emits intervals
- `hpo.py` — `optimize`/`optimize_zoo` (Optuna TPE, objective = purged-CV Sharpe)
- `metrics.py` — **regression only**: rmse, mae, r2, directional_accuracy, interval_coverage/width

**The metric gap is concrete:** `quant/metrics.py` has no IC family, no
calibration/Brier, no deflated/probabilistic Sharpe, no factor attribution. Those
are the genuine net-new work, not "more features."

---

## 2. Metrics dictionary

Prioritized, not exhaustive. Each metric: **what it captures · formula ·
target · when it lies.** Skip anything not on this list until these are live —
a dashboard of 60 metrics nobody acts on is worse than 8 that gate decisions.

### Tier 1 — Signal quality (build first, in a new `quant/ic.py`)

**Rank IC** *(have it; formalize per-period)*
`IC_t = spearman(score_{i,t}, fwd_ret_{i,t})` across names `i` on date `t`.
Target: mean monthly-horizon Rank IC **> 0.03** is usable, **> 0.05** is strong.
Lies when: a few mega-caps dominate the cross-section → decompose (below).

**IC Information Ratio (IC-IR)** — *consistency, the metric that matters most*
`IC_IR = mean_t(IC_t) / std_t(IC_t)`; significance `t = IC_IR · sqrt(N_periods)`.
Target: IC-IR **> 0.5** good, **> 1.0** institutional; require `t > 2`.
Lies when: periods are autocorrelated (overlapping 21d horizons) → use
Newey-West / sample every `horizon` days. **This is the headline number.**

**IC decay curve** — *how fast the edge dies, sets rebalance frequency*
Compute Rank IC at horizons `h ∈ {1,5,10,21,63}`; report half-life.
Interpretation: if IC at 5d ≈ IC at 21d, you are rebalancing too slowly and
paying for it. Lies when: ignored alongside turnover cost.

**IC decomposition** — *where the model works*: Rank IC sliced by sector,
market-cap quintile, and VIX tercile. The deliverable is a **heatmap**. If IC is
strongly positive in low-vol large-caps and ~0 in high-vol small-caps, you have a
liquidity-conditioned signal — which changes universe construction, not the model.

### Tier 2 — Statistical honesty (build second, in `quant/honesty.py`)

**Probabilistic Sharpe Ratio (PSR)**
`PSR(SR*) = Φ( (SR − SR*)·sqrt(n−1) / sqrt(1 − γ₃·SR + ((γ₄−1)/4)·SR²) )`
where γ₃, γ₄ are skew/kurtosis of returns. Probability the true SR exceeds a
threshold SR*. Target: `PSR(0) > 0.95`.

**Deflated Sharpe Ratio (DSR)** — *the multiple-testing fix for your Optuna loop*
`SR* = sqrt(Var(ŜR)) · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]`, γ ≈ 0.5772,
N = number of independent trials tried; then `DSR = PSR(SR*)`.
Target: **DSR > 0.95** before believing any HPO-selected config.
Lies when: you under-count N. **Count every Optuna trial and every feature-set
variant** as a trial — `hpo.py` knows N; thread it through.

**PBO — Probability of Backtest Overfitting** (CSCV, Bailey et al.)
Split the trial matrix into combinatorial train/test partitions; PBO = fraction
where the in-sample-best config lands below the median out-of-sample.
Target: **PBO < 0.1**. PBO ≈ 0.5 means your selection is noise.

**Minimum Track Record Length (MinTRL)** — months of live data needed to confirm
SR > 0 at 95%. Sets honest expectations for the paper-trading gate (§4).

### Tier 3 — After-cost economics (needs the portfolio backtest, §3)

**Cost breakeven** — `c* = gross_alpha_bps / turnover`. The transaction cost (bps)
at which net alpha hits zero. If `c*` is below your realistic all-in cost
(commission + half-spread + impact ≈ 5–15 bps for liquid US names), **the signal
is not investable.** This is the single most decisive economic test. Many
academically "significant" Rank ICs die here.

**Deflated factor alpha** — regress strategy excess returns on **FF5 + momentum**;
alpha = annualized intercept, with t-stat. Captures: is this genuine selection
skill, or repackaged exposure to market/size/value/momentum you could buy for
3 bps in an ETF? Target: alpha t-stat **> 2** after costs. Lies when: you skip it
and call beta "alpha."

**Capacity** — AUM at which `Σ impact(orderᵢ) ≥ gross alpha`. Square-root impact
proxy: `impact_bps ≈ k · σ_daily · sqrt(order$ / ADV$)`. You don't need precision;
you need the order of magnitude (is it \$1M or \$1B?) because it dictates whether
turnover-heavy signals are even worth pursuing.

**CVaR₉₅/₉₉, Omega, Gain-to-Pain** — tail and asymmetry. `backtest.py` already has
VaR/CVaR₉₅; add Omega(threshold) and Gain-to-Pain = ΣR⁺ / |ΣR⁻|.

### Tier 4 — Calibration & operations (ongoing)

**Brier score + reliability diagram** — for the served `ml_pct` probability.
`Brier = mean((p − y)²)`; Murphy decomposition = reliability − resolution +
uncertainty. You isotonic-calibrate per learner; **verify it held out-of-sample**
with a reliability diagram. A miscalibrated `ml_pct` is worse than none — users
read it as a probability.

**Operational (already partially have via `/health`, drift):** prediction
turnover (day-over-day rank churn — high churn = cost you haven't priced),
prediction entropy/spread (is the model differentiating or punting?), feature PSI
per feature (have `population_stability_index`), inference p50/p95/p99,
provider-agreement trend.

> **Metrics that mislead — do not lead with these:** raw accuracy (a
> cross-sectional ranker can have mediocre accuracy and excellent IC); aggregate
> Sharpe without DSR (selection-inflated); single-number IC without IC-IR
> (consistency is the asset); R² on returns (returns are ~unpredictable in level;
> rank is the game).

---

## 3. The bridge: cross-sectional portfolio backtest (new `quant/portfolio.py`)

This is the missing primitive from Finding 2. Spec:

```
run_portfolio_backtest(
    scores:   DataFrame[date × ticker],   # the model's daily scores
    forward_returns: DataFrame[date × ticker],
    construction: 'long_short_decile' | 'long_top_n' | 'rank_weighted',
    sizing:    'equal' | 'vol_target' | 'risk_parity',
    cost_bps, slippage_bps, rebalance='daily'|'weekly',
    liquidity_filter,                      # min ADV$ to be eligible
) -> PortfolioResult(equity, returns, weights, turnover, metrics, attribution)
```

Mechanics that must be correct:
- **One-bar execution lag** (decide on close `t`, earn `t→t+1`) — `backtest.py`
  already models this for one asset; preserve it per name.
- **Per-name turnover costs** on the weight delta, summed across the book.
- **Neutralization**: market-neutral (Σw=0) for long/short; optionally
  sector-neutral to isolate stock selection from sector bets.
- **Reuse `BacktestConfig`** semantics so costs are consistent with the per-symbol engine.

Reuse for metrics, don't reinvent: feed the resulting **book return series** into
the existing `_metrics()` for Sharpe/Sortino/Calmar/MDD/VaR/CVaR. The new code is
only the cross-sectional construction layer.

This single module unlocks Tier-3 metrics, the validation protocol (§4), and any
honest claim about the served model.

---

## 4. Validation protocol — the gate before capital

A model is `champion` only after passing every gate **in order**. Implement as
`quant/protocol.py` returning a pass/fail report per layer; wire into CI so a
candidate that regresses a gate cannot be promoted.

**Gate 0 — Leakage & honesty (fast, run every train)**
- Purged + embargoed splits (have `PurgedWalkForward`); extend to **Combinatorial
  Purged CV** *only* once Tier-2 metrics exist (CPCV is C(N,k) backtests — budget it).
- **Noise test:** shuffle labels / feed pure-noise targets → IC must collapse to ~0.
  If the pipeline "finds signal" in noise, the pipeline is broken. Non-negotiable.
- IC-IR `t > 2` and stable across ≥3 random seeds.

**Gate 1 — Economic validity (needs §3)**
- Net IC-IR and net Sharpe **after** realistic costs.
- **Cost breakeven `c*` > realistic cost** with margin (≥2×).
- FF5+MOM alpha t-stat > 2 (it's selection, not beta).
- Capacity order-of-magnitude documented.

**Gate 2 — Robustness**
- **Feature ablation:** drop each feature group; a feature group that doesn't move
  purged-CV IC is removed (Hick's law for models — fewer, load-bearing features).
- Universe sensitivity (S&P 500 vs Russell 1000 if obtainable), period-by-period
  (per-year, not just aggregate), parameter sensitivity (small HPO perturbations).
- DSR > 0.95, PBO < 0.1.

**Gate 3 — Paper trading (real time, ≥ MinTRL, floor 30 trading days)**
- Serve predictions, execute nothing. Log score-at-compute vs score-at-would-be-fill
  → **signal decay from latency**. Compare live IC to backtest IC; a large gap is
  look-ahead you didn't catch.

Only after Gate 3 → small capital. This sequence is the product.

---

## 5. Model evolution blueprint — three iterations

Each iteration has a **single success criterion** measured by §4. Do not start an
iteration until the prior one's harness can score it.

### Iteration A — "Measure the model you have" (no new model)
Wrap the current ensemble in the §3 portfolio backtest + §2 Tier-1/2/3 metrics.
**Success:** you can state, with DSR and after-cost numbers, whether the live
signal is economically real. Likely outcome: it works in a subset (e.g. liquid
large-caps, low-vol regime) — which *defines* iteration B.

### Iteration B — Regime conditioning, the cheap way first
The brief proposes an HMM. **Push back:** a 2–3 state HMM trained on ~20y (which is
mostly one secular bull) overfits regime labels and adds a latent-state failure
mode. Start with **rule-based regimes** (VIX tercile × SPY-above-200DMA × ADX
trend/MR) — transparent, no training, and §2's IC-decomposition already tells you
which regimes the signal lives in. Route/weight: `0.7·regime_model + 0.3·global`
for smooth transitions; weight base learners by **recent rolling IC-IR** rather
than a learned meta-learner at first (stacking is leakage-prone — the OOF
predictions must come from the *same* purged splits, and that is easy to get
subtly wrong).
**Success:** regime-conditioned net IC-IR beats the global model out-of-sample
across ≥2 distinct regimes, DSR-deflated for the extra trials.

### Iteration C — Uncertainty-adjusted scoring + (only if warranted) a learned meta-ensemble
You already emit prediction intervals in the zoo. Score = `predicted_edge /
prediction_uncertainty` (ensemble disagreement or interval width as the
denominator) → naturally downweights low-conviction names and feeds position
sizing. Add a learned meta-learner over OOF base predictions **only if** dynamic
IC-weighting (Iteration B) shows base models are genuinely regime-specialized and
a meta-learner beats the simple weighting on PBO/DSR — not before.
**Success:** conviction-weighted book improves after-cost Sharpe and reduces
drawdown vs equal-weight, net of added trials.

**Explicitly deferred (with reasons), not in the 3 iterations:**
- **Deep learning / TabNet / MLP:** tree ensemble is nowhere near its limit; DL on
  ~500×daily tabular data rarely beats GBM and imports a large explainability +
  ops burden. Revisit only if ablation shows tree models plateau *and* you have
  a feature-rich (alt-data) input space. Agrees with the brief, stated firmer.
- **RL for position sizing:** needs a validated signal + a faithful simulator
  first; otherwise it optimizes a fiction. Long-term, conditional on everything above.
- **Alt-data Phases 2–3 (13F, NLP, satellite):** **do not** integrate alt data into
  an unproven signal — it adds DOF and trial count, inflating overfitting exactly
  when you can least afford it. Phase-1 alt data (VIX term structure, put/call,
  intermarket) is cheap and regime-relevant; gate even those through ablation.

---

## 6. Research infrastructure

### Experiment tracking — lightweight, not MLflow-server
The brief suggests MLflow. **Push back for this stack:** a hosted MLflow tracking
server is operational weight that a Lambda + S3 system shouldn't carry yet. Use
**MLflow's file/SQLite backend on S3**, or simpler, an append-only experiment log
you already have the primitives for (`ArtifactStore`):

```
s3://…/experiments/{exp_id}/  ->  meta.json  (git sha, data range, universe hash,
                                              feature-set hash, params, seed, N_trials)
                                  metrics.json (ALL gate metrics, not just the best)
                                  oof_preds.parquet (for later stacking)
                                  report.html
```
Tag lifecycle `candidate → challenger → champion → production`. **Champion-challenger
rule:** production stays until a challenger beats it on the *promotion metric*
(net IC-IR or after-cost Sharpe, DSR-deflated) for **N consecutive refreshes** —
not a single lucky window. `model_history.json` already exists; extend it into this.

### Feature lifecycle (formalize what `FeatureEngineer` half-does)
Hypothesis → look-ahead audit → univariate IC/monotonicity → corr < 0.7 vs
existing (you already corr-prune) → marginal purged-CV contribution → stability
across regimes → survives costs → version bump. **Each feature carries a card**
(formula, rationale, coverage, IC by regime, failure modes) — mirror the existing
model-card discipline at the feature level.

### Reproducibility
Seed everything; hash data/feature/model versions into `meta.json`; pinned deps
(have `requirements.txt`); the monthly **replication test** — fresh clone
reproduces a frozen backtest bit-for-bit — is the cheapest insurance against
silent pipeline drift.

---

## 7. Infrastructure evolution — triggers, not speculation

The single-Lambda design is **correct for now**; don't pre-scale. Concrete trigger points:

| Trigger (measured) | Action |
|---|---|
| CPCV / portfolio backtests blow the **15-min Lambda budget** | Move the *training/research* job to **Fargate/ECS** (or a larger-memory, longer-timeout Lambda); keep serving Lambda lightweight. The handler already dispatches by event — this is a clean split. |
| Universe > ~1000 names *and* train time > budget | Same Fargate split; chunked feature compute. |
| You need intraday features | Feature store (Feast/Redis) + streaming ingest. **Not before** capacity/cost are understood (§2). |
| > 1 model in production | The S3-versioned registry + champion-challenger from §6; add shadow deployment (challenger predicts, doesn't serve) before canary. |

**Monitoring/alerts worth wiring now** (you have the data): *Critical* — no refresh
2+ weekdays, artifact load failure, API error > 5%, PSI > 0.25 on a core feature.
*Warning* — PSI > 0.15, rolling IC < 0 for 10 sessions, data staleness > 30 min in
market hours, one provider persistently disagreeing. Map these onto the existing
`/health` + drift outputs.

---

## 8. Architectural decisions

The brief's five, answered — plus the one that dominates them.

**0. (Added) Unify the research and serving paths.** The cross-sectional serving
model and the per-symbol research toolkit must meet at a **cross-sectional
portfolio backtest** (§3). *Recommendation:* this is decision #1; the other five
are downstream.

**1. Complexity vs interpretability** — *Keep tree-based; push it to its limit.*
SHAP works, ops are simple, and you have no evidence trees plateaued. Earn DL with
an ablation that shows diminishing returns.

**2. Cross-sectional vs time-series** — *Cross-sectional primary + a market-timing
overlay.* The hierarchical Market→Sector→Stock framing is right, but implement it
as **neutralization in the portfolio backtest** (sector-neutral books) first, not
as three separate model stacks. Cheaper, and it directly measures whether sector
bets vs stock selection drive returns.

**3. Point vs distributional** — *Build distributional for risk, point for ranking.*
You already emit intervals; route uncertainty into **scoring and position sizing**
(Iteration C), not into a full distributional rewrite of the ranker.

**4. Daily vs intraday** — *Stay daily.* Intraday multiplies data/infra cost and is
unjustified until capacity (§2) shows daily turnover is the binding constraint.

**5. Single model vs zoo** — *Zoo for research, promote a small ensemble to serving.*
Exactly the current split; formalize promotion via champion-challenger (§6). Don't
serve the whole zoo — serve the few that survive the gates.

---

## 9. Prioritization matrix — effort × impact (re-sequenced)

Impact = evidence-toward-"is-this-real" + decision value. Effort = build days.
This **re-orders the brief's roadmap** to front-load the harness over new models.

| # | Work | Effort | Impact | Why this rank |
|---|------|:---:|:---:|---|
| 1 | **Portfolio backtest** (`quant/portfolio.py`, §3) | M | ★★★★★ | Unlocks every economic metric; without it nothing downstream is measurable |
| 2 | **IC decomposition + IC-IR + decay** (`quant/ic.py`) | S | ★★★★★ | The headline signal-quality numbers; cheap |
| 3 | **DSR + PBO + cost breakeven** (`quant/honesty.py`) | S | ★★★★★ | Statistical/economic honesty on what you already run; you're exposed today |
| 4 | **Validation protocol gates** (`quant/protocol.py`, §4) | M | ★★★★☆ | Turns metrics into a promotion decision; CI-enforceable |
| 5 | **Feature ablation study** | S | ★★★★☆ | Removes dead features; shrinks overfitting surface |
| 6 | **Experiment log + champion-challenger** (§6) | M | ★★★★☆ | Makes results reproducible/comparable; prerequisite for any model bake-off |
| 7 | **Rule-based regime conditioning** (Iter B) | M | ★★★☆☆ | Real upside, but only meaningful once 1–5 quantify *where* the signal lives |
| 8 | **Uncertainty-adjusted scoring** (Iter C) | S | ★★★☆☆ | Good risk/return polish; needs the harness to prove it helps |
| 9 | **Phase-1 alt data** (VIX term, put/call, intermarket) | M | ★★☆☆☆ | Gate through ablation; easy to add DOF that overfits |
| 10 | **Paper-trading harness** (Gate 3) | M | ★★★★☆ | Required before capital; can run in parallel from week 1 |
| — | HMM regimes / DL / TabNet / RL / 13F / NLP / satellite | L | ★☆☆☆☆ *(now)* | Deferred — see §5; revisit only after 1–6 prove a business |

**Immediate term (2–4 wks):** items **1, 2, 3, 5** + start the paper-trading log (10).
**Short term (1–2 mo):** **4, 6, 7**. **Medium:** **8, 9** + sector-neutral
hierarchical books. **Long:** the deferred list, each gated on the harness.

---

## 10. Concrete module plan

New code lives beside the existing toolkit so the research/serving unification is physical, not notional:

```
marketdesk/quant/
  ic.py          # NEW  rank/Pearson IC, IC-IR, decay curve, decomposition (sector/cap/VIX)
  honesty.py     # NEW  PSR, DSR (reads N_trials from hpo), PBO/CSCV, MinTRL
  portfolio.py   # NEW  cross-sectional book construction + backtest (the bridge, §3)
  protocol.py    # NEW  gated validation runner (§4) → pass/fail report
  attribution.py # NEW  FF5+MOM regression, sector vs selection attribution
  backtest.py    #      reuse _metrics() from portfolio book returns
  metrics.py     #      extend: add calibration/Brier + reliability-diagram data
marketdesk/research/
  experiments.py # NEW  append-only experiment log over ArtifactStore (§6)
  features_registry.py # NEW  per-feature cards + lifecycle status
```

**First PR (the keystone):** `quant/portfolio.py` + `quant/ic.py` + a script that
runs the **current** serving model through both and prints IC-IR, after-cost
Sharpe, DSR, and cost-breakeven. That one PR converts MarketDesk from "produces
scores" to "knows whether the scores are worth acting on."

---

*This document should evolve with the system. When a gate threshold proves wrong
in practice, change it here and say why — the reasoning is the asset, not the number.*
