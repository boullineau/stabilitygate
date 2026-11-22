# stabilitygate

A three-outcome test-retest stability screen for repeated-call LLM experiments.

## Status

This is a research prototype at version 0.1.0, released alongside *The Screen
Decides the Verdict: Operating Characteristics of Model-Exclusion Rules in LLM
Evaluation* ([doi 10.5281/zenodo.21366659](https://doi.org/10.5281/zenodo.21366659)).
The package follows the decision order of the paper's reference screen, which
checks the design before either test, and differs from it in test statistics and
calibration. "Reading this against the defaults" sets out the differences. The
screen verdicts for the four models in "The worked case" are a new worked
example and are not reported in the paper.
A browser version that screens the four models and your own two runs is
at [judgmentalism.org/stabilitygate](https://judgmentalism.org/stabilitygate/). The package implements the screen and the design calculator that the paper's
section "Design requirements" describes, and it reports its own operating
characteristics in full, favourable results and unfavourable results together,
under "Operating characteristics" below. The admitting verdict is a failure to
reject a no-drift null and not a rejection of consequential drift. Read "What
this is not" before adopting it as a gate in a pipeline of your own.

## Why this exists

A correlation-based test-retest gate asks whether the *profile* across matched
conditions was preserved. Its resolution therefore depends on how much reliable
between-condition spread the suite supplies. Many evaluation suites contain
conditions on which response rates sit near a boundary, and a saturated suite has
little spread. In that regime the correlation is reading sampling noise, and a
fixed threshold on it has design-dependent error rates rather than nominal ones.

Across 252 simulated designs an `r > .80` gate excludes models generated under no
drift at a median rate of **.763**. In the worked case below it passes the model
with the largest mean absolute change in response rate (.129) and fails the model
with the smallest (.018).

The screen replaces the one-coefficient gate with three questions and returns one
of three verdicts.

- **INDETERMINATE** if the design carried too little reliable between-cell
  information to rule. The screen declines rather than guessing.
- **FAIL** if the design was informative and either the model's overall
  propensity shifted, a signed run effect on the log-odds scale that a
  rank-order correlation cannot see, or particular cells moved differently.
- **no_drift_detected** if the design was informative and neither a run effect
  nor a cell-by-run interaction was detected.

That third verdict is deliberately not called PASS. Both component tests are
calibrated against the design's own no-drift null, so the verdict records a
failure to reject no drift rather than a rejection of consequential drift. Those
two criteria carry different error-control properties and this package implements
the weaker one.

## Install

```bash
pip install git+https://github.com/boullineau/stabilitygate
# or, from a local checkout
pip install .
```

The package is not on a package index.

The only runtime dependency is `numpy`. `scipy` is optional and used by one test.
Python 3.9 and later.

## Quickstart

### 1. Screen data you already have

```python
from stabilitygate import screen

# per-cell response rates for the two runs, one entry per matched cell
main   = [0.90, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 1.00]
retest = [1.00, 1.00, 1.00, 1.00, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]

result = screen(main, retest, n_calls=20)
print(result)          # the verdict and every diagnostic behind it
print(result.verdict)  # 'no_drift_detected' | 'FAIL' | 'INDETERMINATE'
```

`screen` accepts rates in `[0, 1]` or integer success counts. Counts are
auto-detected only when a value exceeds 1. For 0/1 count data pass
`input_kind="count"` explicitly, `"rate"` and `"auto"` being the other options.
It returns a `ScreenResult` dataclass carrying the verdict, the reliability
`lambda` with a bootstrap confidence interval, the run-effect and interaction
statistics with bootstrap p-values, the decision rule that produced the verdict,
drift on the probability and log-odds scales, and, for reference only, the
Pearson and Spearman correlations and what a bare correlation gate would have
said.

### 2. Calibrate a design before you collect data

```python
from stabilitygate import power, recommend_design

# operating characteristics of a 14-cell, 20-call design near a ceiling
for row in power(K=14, n_calls=20, baseline=0.95):
    print(row.mode, row.final_pass, row.final_fail, row.final_indeterminate)

# smallest design keeping false exclusion at or below 5% while detecting a
# half-log-odds shift at least 80% of the time
rec = recommend_design(
    baseline=0.90,
    drift_logit_that_matters=0.5,
    target_false_exclusion=0.05,
    target_power=0.80,
)
print(rec)
```

`recommend_design` is a search over a small set of candidate designs, each
evaluated by simulation. At its default replicate counts the design it returns
can change with the seed. Run it with several seeds or a larger `n_rep`, and
check the design it returns with `power` before relying on it.

### 3. Command line

```bash
python -m stabilitygate demo                       # reproduce the four-model case
python -m stabilitygate screen mydata.csv          # long-format CSV, one row per cell
python -m stabilitygate power --K 14 --n 20 --baseline 0.9 0.95 0.99
python -m stabilitygate recommend --baseline 0.9 --drift 0.5
```

Every subcommand takes `--seed` and writes an optional results CSV with `--out`.
The decision-rule flags of "The decision rule" below are available on `screen`,
`power` and `recommend`.

## The worked case

`python -m stabilitygate demo` reproduces the paper's figures for the four models,
the signal fraction, the correlation and the gate's result, and adds the screen's
results, which the paper does not report.

| Model | lambda | z_run | Q | Pearson r | Bare gate | Screen verdict |
|---|---:|---:|---:|---:|:--:|:--:|
| Claude Opus 4.7 | 0.98 | -3.48 | 14.6 | +0.886 | PASS | **FAIL** |
| Gemini 3.1 Pro | 0.94 | -0.75 | 11.7 | +0.928 | PASS | **no_drift_detected** |
| DeepSeek-V4-Flash | 0.82 | +1.21 | 2.6 | +0.907 | PASS | **no_drift_detected** |
| GPT-5.5 | 0.29 | +0.55 | 2.3 | -0.135 | FAIL | **INDETERMINATE** |

The `Bare gate` column keeps the labels PASS and FAIL because it reports what the
incumbent correlation gate would have said, not what this screen returns.

The bare gate passes Claude, whose mean absolute change in response rate was the
largest of the four (.129, a fall of half a log-odds), and fails GPT-5.5, whose
mean absolute change was the smallest (.018). GPT-5.5 sat at a ceiling, where the
correlation is uninformative and where a small change in rate is a large change
on the log-odds scale (+1.40). On GPT-5.5's design a perfectly stable model fails
that gate about 84% of the time. The screen fails Claude on its run effect and returns indeterminate for
GPT-5.5, whose design could not resolve the question.

Reproduce it, and every other number here, with `bash repro/reproduce.sh`.

## Interpreting the verdicts

An **INDETERMINATE** verdict is information, not failure. It says the design, not
the model, fell short. There was too little reliable between-cell variance to
detect the drift that would matter. The remedy is a stronger design, which is
what `power` and `recommend_design` help you find before you spend calls.

By default the information floor is evaluated first, so INDETERMINATE dominates
and no test rules on a design below the floor. This is the rule behind the
measured operating characteristics below. When the run-effect test rejects on
such a design, the result carries `run_detected_below_floor=True` and says so in
`reason`. Set `precedence="test_first"` to let a detected run effect return FAIL
below the floor instead. The choice moves false exclusion by a factor of two or more
at a saturated baseline, so it is worth making deliberately.

A **no_drift_detected** verdict means neither component test rejected. It does
not mean the model is stable. See "What this is not".

## Operating characteristics

A screen is an instrument. Publishing one without its error rates repeats the
failure the accompanying paper is about, so this section reports what
`stabilitygate` does on models whose true state is known, favourable results and
unfavourable results together.

### How these were measured

A simulation grid of 252 configurations, crossing seven baselines from .50 to
.99, four between-condition spreads from .25 to 1.5 logits, three condition
counts in {14, 28, 56}, and three call counts in {20, 50, 200}, at 4,000
replicates per configuration. Condition propensities are drawn on the logit
scale, responses are binomial, and the re-run is either identical to the first
run (stable), or shifted by one logit in either direction. All runs use seed
20260101. Reproduce them with `bash repro/reproduce.sh`, which writes
`results/extension_design.csv`.

These numbers describe the configuration named `stabilitygate.presets.OC_2026_08_28`.
It shares the default decision rule and differs from the defaults in its
statistics and its null. The differences are listed under "Reading this against
the defaults" at the end of this section, followed by the same grid measured
under the defaults.

### Two denominators, declared once

The screen has three outcomes and the correlation gate it replaces has two, so
the two cannot be compared on one denominator without a choice. A *resolvable*
draw is one on which the screen did not return indeterminate. Rates called
**conditional** are computed on resolvable draws. Rates called **unconditional**
are computed on all draws. Both are given wherever they differ, because
conditioning is not neutral. Resolvable draws are the high-variance draws where a
correlation works best, so conditioning helps the incumbent.

### Size under no drift

Pooled across the grid, conditional size is **.0501** on 645,061 resolvable
draws, against a 5 per cent target. That is the favourable headline and it does
not hold uniformly.

Per-configuration conditional size reaches **.295** (13 of 44 resolvable draws,
or .294 from the four-decimal rates in `extension_design.csv`). It exceeds .10 in **41 of
the 252 configurations**, and differs from nominal at a Bonferroni-corrected
threshold in a minority of them, almost all on the high side.

The non-uniformity has a cause rather than being noise. Critical values are
calibrated on all stable draws while size is reported on the resolvable subset,
and conditioning on the information floor selects the high-observed-variance
draws most likely to trip the heterogeneity test. The pattern follows the split
exactly.

| Configurations | Count | Pooled conditional size |
|---|---:|---:|
| Well resolved, indeterminate under .10 | 124 | .048 |
| Poorly resolved | 128 | .056 |
| All | 252 | .0501 |

The aggregate figure is carried by the well-resolved configurations, which hold
76.2 per cent of resolvable draws. **Under this preset the screen holds its
nominal size where the design resolves the drift and does not hold it reliably
where the design does not.**

### Power where the design resolves the drift

In the 124 well-resolved configurations, median conditional power against a
positive one-logit shift is **1.000**, against **.162** for a fixed `r > .80`
gate on the same draws, at matched median realised size of **.048** against
**.058**.

That comparison is restricted to the well-resolved subset and the restriction
matters. Over all 252 configurations the fixed gate's medians are **.857**
against a positive shift and **.713** against a negative one, so the gate is not
weak in absolute terms across the whole grid and this comparison rests on the
well-resolved subset.

Pooled on the resolvable denominator across the whole grid, the fixed gate
excludes a stable model on **.348** of draws against the screen's **.0501**. That
figure needs the joint distribution of the two instruments on the same draws,
which `results/extension_joint_counts.csv` supplies.

### Direction symmetry

Median conditional power is **1.000** against a positive one-logit shift and
**1.000** against a negative one. The screen does not have a preferred direction.
The fixed correlation gate does, and in the wrong direction at the ceiling.

### The cost, and the boundary

Two unfavourable results, both on both denominators.

**Abstention.** Under stability the screen returns indeterminate on a median of
**10.8 per cent** of draws across the grid, rising to **99.9 per cent** in the
worst-resolved configurations. That is the price of declining to rule when the
design cannot support a ruling. A binary screen returns a confident verdict on
those same draws, and on the evidence of the accompanying paper that verdict is
wrong most of the time.

**False admission of a shifted model.** On the resolvable denominator, **50 of
252** configurations admit a model carrying a full one-logit shift on more than 5
per cent of draws, with a maximum of **91.9 per cent**. Unconditionally the same
figures are **19 of 252** and **41.0 per cent**. These sit where the design is
poorly resolved and the level test has little to work with. The screen as
released is not a universal repair and should not be treated as one.

### What these numbers do and do not establish

They establish that, under the preset, the screen holds its nominal size and has
complete power against a one-logit shift in the regime where a screen can work
at all, that it declines
to answer outside that regime, and that it is not reliably sized in the region
where it declines. They do not establish that it is sound everywhere, and the two
boundary results above are the reason.

The admission criterion is a failure to reject no drift, calibrated against the
design's own no-drift null. It is not an equivalence test and does not require
rejecting the presence of consequential drift. That is the weaker criterion and
it is why the screen still admits a shifted model in a minority of poorly
resolved configurations.

### Reading this against the defaults

| Setting | `OC_2026_08_28` | Package default |
|---|---|---|
| Per-component level | `alpha / 2` | `alpha / 2` |
| Lambda rule | Point estimate | Point estimate |
| Level statistic | Pooled mean log-odds difference | Woolf weighted mean per-cell log odds ratio |
| Heterogeneity statistic | Pearson chi-square on per-condition differences | Cochran's Q |
| Null | Oracle, from the true design parameters | Parametric bootstrap from the observed pooled rates |
| Precedence | Resolution first, indeterminate dominates | Resolution first, indeterminate dominates |

The first two rows and the last are equal. The oracle null is available only in
simulation, since it requires knowing the true design parameters, and a user with
data in hand gets the bootstrap null.

**The three middle rows matter, so the grid was measured again under the
defaults.** See the next subsection.

Under `precedence="test_first"`, at baseline .90, sigma .50, K = 14 and n = 20,
the screen excludes a stable model more than twice as often as the figures above
and detects a positive one-logit shift more than three times as often, because a detected
run effect rules on designs the default would decline. `tests/test_presets.py`
measures both.

### The same grid under the defaults

`repro/run_default_grid.py` runs the same 252 configurations through `power` at the
package defaults, which are the Woolf z, Cochran's Q, the bootstrap null and resolution
first. It uses 2,000 replicates per configuration, 499 null draws per replicate and seed
20260101. Each replicate is decided by the rule `screen` applies to observed data. With 499
null draws each component test runs at 12 / 500 = .024, where `screen` with 20,000 draws
runs it at .025. The outcome counts are in `repro/default_grid.csv`. A rate from one
configuration carries a Monte Carlo standard error of at most .011, and conditional rates
that rest on few resolvable draws carry more.

| Quantity under no drift or a one-logit shift | `OC_2026_08_28` | Package defaults |
|---|---:|---:|
| Replicates per configuration | 4,000 | 2,000 |
| Pooled conditional size | .0501 on 645,061 draws | .0576 on 322,593 draws |
| Pooled conditional size, well-resolved configurations | .048 in 124 | .052 in 125 |
| Pooled conditional size, poorly resolved configurations | .056 in 128 | .076 in 127 |
| Median conditional size across configurations | .051 | .058 |
| Maximum conditional size in one configuration | .295 (13 of 44 draws) | 1.000 (1 of 1 draws) |
| Configurations with conditional size above .10 | 41 | 61 |
| The same two figures over configurations with at least 50 resolvable draws | .222 and 26 of 232 | .326 and 40 of 227 |
| False exclusion on all draws, median and maximum | .040 and .059 | .043 and .086 |
| Indeterminate under no drift, median and maximum | 10.8% and 99.9% | 10.3% and 99.9% |
| Median conditional power against +1 logit, well resolved | 1.000 | 1.000 |
| Median conditional power against −1 logit, well resolved | 1.000 | 1.000 |
| Configurations admitting a +1 logit shift on over 5% of resolvable draws, and the maximum | 50 and 91.9% | 62 and 100.0% |
| The same on all draws | 19 and 41.0% | 29 and 35.5% |

A well-resolved configuration is one that returns indeterminate on under 10 per cent of
stable draws under the column's own configuration. The preset column is recomputed from
`extension_design.csv` with the definitions used for the defaults column, so both columns
are like for like.

The two columns differ in test statistics, in the null and in the number of replicates
at once, so this comparison shows the size of the gap and not which difference produces it.
Under the defaults the screen holds its size less well than under the preset. Pooled
conditional size is .058 against .050. The excess sits in the poorly resolved configurations,
at .076 against .056, and in the well-resolved configurations it is .052. More
configurations exceed .10, 61 against 41, and half as many replicates make each
configuration's rate noisier, which by itself raises that count. Abstention and power in
the well-resolved configurations are about the same in both columns. False admission of
a one-logit shift is more widespread under the defaults, in 62 configurations against 50 on
resolvable draws and in 29 against 19 on all draws, with a lower maximum on all draws. The bootstrap null
treats the pooled cell rates as the truth and the floor then selects the draws with the
most observed variance, so the nominal level is not guaranteed on the draws that receive a
verdict. Under either configuration, read a verdict from a poorly resolved design with
that in mind.

## Controlling false exclusion

Because the screen excludes when either component test rejects, two tests each
run at the full `alpha` put the union near `2 * alpha`. The default
`alpha_split="bonferroni"` runs each at `alpha / 2` so the union sits near
`alpha` instead. `recommend_design` reports the
achieved false exclusion, power and abstention rate for the design it returns, so
the trade is explicit. The grid figures under "Operating characteristics" are the
authority on what the screen actually does.

## Choosing the information floor

`lambda_floor = 0.50` is an illustrative default, not a universal constant.
Choose it for your setting with `power`, which shows how often a design of a
given size clears the floor at your expected baseline and spread.

The floor is a *resolution* criterion, not a power criterion. The accompanying
paper motivates the indeterminate verdict by what a design can resolve, and this
release operationalises that with a preregistered floor on `lambda` chosen at
design time with `recommend_design`, which is simpler and computable from data in
hand. A real power criterion behind `resolution_rule="power"` is later work.

`lambda` is computed from the main run alone.

Two rules handle the sampling uncertainty in `lambda` itself.
`lambda_rule="point"` compares the point estimate to the floor, which is the
paper's headline and the default. `lambda_rule="lower"` compares the lower bound
of `lambda`'s bootstrap interval to the floor, which is conservative and returns
indeterminate more often. `ScreenResult` reports the interval under both.

## What this is not

- **Not an equivalence test.** Admission is a failure to reject a no-drift null,
  not a rejection of consequential drift. A `no_drift_detected` verdict is
  evidence that this design did not detect drift, not evidence that the model is
  stable.
- **Not calibrated outside its assumptions.** The null is logistic-normal with
  independent binomial calls and equal calls per cell. Prompt-template effects,
  call ordering, caching and within-cell dependence are outside it, and
  within-cell dependence makes the screen reject too often. See "Scope".
- **Not a claim about why a model changed.** The screen separates a common shift
  from cell-specific movement. It says nothing about provider updates, sampling
  temperature or anything else that might have caused either.
- **Not a substitute for a design calculation.** Running the screen on a design
  that cannot resolve the drift that matters returns INDETERMINATE, which is the
  correct answer and not a useful one. Use `recommend_design` first.

## Faithfulness and reproducibility

Every point estimator matches the method paper's reproducibility package.
Reliability `lambda` is one minus the ratio of mean binomial sampling variance to
observed between-cell variance. The signed run effect is an inverse-variance
weighted (Woolf) mean per-cell log odds ratio, asymptotically Mantel-Haenszel.
The interaction is Cochran's Q. Log-odds are Haldane corrected so boundary cells
stay finite. The package reproduces the four-model verdicts and reliabilities
exactly.

All p-values and critical values come from each design's own perfectly-stable
parametric bootstrap, so each test is referred to the design's own pooled rates
and not to a fixed reference distribution. The bootstrap treats those pooled
rates as the truth, which is why size is measured by simulation under "Operating
characteristics" and is not guaranteed. Those bootstrap p-values are Monte Carlo estimates and agree with the
paper within Monte Carlo error rather than bit for bit. They use add-one
smoothing, so the smallest reportable value is `1 / (n_boot + 1)` and never
exactly zero. Increase `n_boot` for tighter small-tail estimates. Given a seed,
every function is deterministic and independent of the order in which models,
baselines or designs are evaluated.

### One-command reproduction

```bash
pip install -e .
pip install -r repro/requirements-repro.txt
bash repro/reproduce.sh
```

`repro/requirements-repro.txt` pins numpy 2.2.6, the version the published
numbers were produced with. numpy 2.5 changed how large binomials are drawn, so
from that release rates agree within Monte Carlo error and a few exact counts
differ. The counts quoted under "Operating characteristics", such as 41 of 252
configurations and the maximum of .295, belong to that random stream.

Seven stages, from the worked case to the figures, ending in a verification that
recomputes every published number and prints `0 FAIL`. Measured wall clock on the
authors' machine is **1 minute 14 seconds**, on one core, with the MD-Agreement
corpus present. `SG_QUICK=1` runs every stage at reduced replicate counts in
about 15 seconds as a smoke test. `repro/README.md` documents the stages, the
environment variables and the tolerances. Stages 2, 4 and 5 rewrite their
deposited reference outputs byte for byte.

The MD-Agreement corpus is distributed by its own authors and is not
redistributed here. Point `SG_MDAGREEMENT` at a local copy to enable stage 5. It
skips cleanly otherwise.

### Tests

```bash
pip install -e ".[test]" pytest-cov
pytest -q                                          # the fast suite
pytest -q -m "not slow"                            # skipping the recomputing variants
pytest --cov=stabilitygate --cov-fail-under=90     # with coverage
```

The suite reproduces the four-model verdicts and reliabilities, pins every number
in the paper and in "Operating characteristics" above, pins the decision logic,
input handling, determinism and design calibration, and asserts that each
decision-rule parameter changes at least one verdict away from its default.

## API

```
screen(main_rates, retest_rates, n_calls, *, input_kind="auto",
       lambda_floor=0.50, alpha=0.05, alpha_split="bonferroni",
       run_sided="two", combine="or", precedence="resolution_first",
       level_statistic="woolf_z", heterogeneity_statistic="cochran_q",
       null="bootstrap", lambda_rule="point", gate_threshold=0.80,
       n_boot=20_000, n_boot_lambda=10_000,
       compute_gate_false_exclusion=False, seed=20260101) -> ScreenResult

screen_dataframe(rows, *, group_key="model", main_col="main_rate",
                 retest_col="stability_rate", n_col="main_n", **kwargs)
                 -> dict[str, ScreenResult]

power(K, n_calls, *, baseline=0.90, sigma_logit=0.50, modes=DEFAULT_MODES,
      lambda_floor=0.50, alpha=0.05, alpha_split="bonferroni", run_sided="two",
      combine="or", precedence="resolution_first", level_statistic="woolf_z",
      heterogeneity_statistic="cochran_q", null="bootstrap",
      lambda_rule="point", n_rep=1500, n_boot=300, n_boot_lambda=1_000,
      seed=20260101) -> list[PowerRow]

recommend_design(*, baseline, drift_logit_that_matters, sigma_logit=0.50,
                 target_false_exclusion=0.05, target_power=0.80,
                 lambda_floor=0.50, alpha=0.05, ...the same decision-rule
                 parameters as power..., candidate_K=(10,14,20,28,40),
                 candidate_n=(20,30,50,80), n_rep=1200, n_boot=250,
                 seed=20260101) -> DesignRecommendation
```

The verdict labels are exported as `NO_DRIFT_DETECTED`, `FAIL` and
`INDETERMINATE`, with `VERDICTS` holding all three, so downstream code can
compare against a name rather than a literal. `stabilitygate.presets` holds
`OC_2026_08_28`, the configuration under which `power` reproduces the published
operating characteristics.

### The decision rule

The screen runs two detection tests, a signed run effect and a cell-by-run
interaction, and excludes when either rejects. Four parameters expose every
choice that combination involves. All four are accepted by `screen`, `power`,
`recommend_design` and the matching command-line subcommands.

| Parameter | Default | Other values | What it decides |
|---|---|---|---|
| `alpha_split` | `"bonferroni"` | `"sidak"`, `"none"` | The level each component test runs at, given the union budget `alpha`. `"bonferroni"` runs each at `alpha / 2`, `"sidak"` at `1 - (1 - alpha) ** 0.5`, `"none"` at the full `alpha`. |
| `run_sided` | `"two"` | `"greater"`, `"less"` | The tail of the run-effect test. One-sided settings are the right choice when the direction that would matter is fixed in advance. |
| `combine` | `"or"` | `"and"` | Whether one rejecting component excludes the model, or both must. `"and"` lowers false exclusion and lowers power. |
| `precedence` | `"resolution_first"` | `"test_first"` | Whether a detected run effect may rule on a design below the information floor. The default says it may not, so INDETERMINATE dominates. |

Two further parameters select the statistics, `level_statistic` (`"woolf_z"` or
`"pooled_logit"`) and `heterogeneity_statistic` (`"cochran_q"` or
`"pearson_chi2"`). `power` and `recommend_design` additionally accept
`null="oracle"`, which calibrates from the true design parameters rather than
from a bootstrap of the observed rates and is available only in simulation.

## Scope

The screen is designed for matched-cell, repeated-call binary or proportion
responses with equal calls per cell, the setting of the worked case. It assumes
independent calls within a cell. Positive within-cell dependence (overdispersion)
makes two runs of a stable model differ by more than binomial sampling allows, so
the screen rejects too often. In a simulation at K = 14, n = 20, baseline .50 and a spread of one logit, a stable model failed in .047 of 2,000 draws with independent calls, in .548 with an intraclass correlation of .05 and in .859 with .10 (`repro/run_dependence_check.py`). Do not run the screen on calls
that share a seed, a cache or a conversation. It is a decision aid for study
design and inclusion, not a claim about why a model changed.

## Citation

Cite the software and the paper that measures it.

> Boullineau, E., & Muñoz Arciniegas, J. D. (2026). *The Screen Decides the
> Verdict: Operating Characteristics of Model-Exclusion Rules in LLM Evaluation*.
> Can We Trust AI Evaluation? (TAE) Workshop at NeurIPS 2026.
> https://doi.org/10.5281/zenodo.21366659

`CITATION.cff` carries the machine-readable form.

## License

Code is released under the MIT License, in `LICENSE`. The bundled data file
`stabilitygate/data/four_model_stability.csv` is released under CC BY 4.0, in
`stabilitygate/data/LICENSE`.
