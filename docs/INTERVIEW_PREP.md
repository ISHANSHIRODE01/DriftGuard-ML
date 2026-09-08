# DriftGuard — Interview Preparation

Read this until you can answer every question without looking. If you can
explain the reasoning in your own words, the project is yours regardless of
who typed the code. If you cannot, no amount of polish will survive round 2.

Start here: **the one-line summary.**

> "I built a drift monitoring system that decides when to retrain a model, and
> I measured it against a no-retraining baseline on the ELEC2 benchmark. F1
> improved 28% while accuracy only moved 1% — the stale model was collapsing
> toward the majority class, which accuracy alone completely hid."

That last sentence is the whole interview. It shows you measured something,
noticed something counter-intuitive, and understood why.

---

## Part 1 — The five questions you will definitely get

### Q1. "Why did F1 improve 28% but accuracy only 1%?"

**Answer:** ELEC2's label is roughly 42% positive, so a model can score ~70%
accuracy by leaning heavily toward the majority class. As the data drifted,
the frozen model's decision boundary became increasingly wrong for the new
regime, and its safest behaviour was to predict the majority class more often.
That keeps accuracy propped up while destroying minority-class recall — F1
bottomed out near 0.05 around batches 20–23, meaning it had almost stopped
identifying the positive class at all.

Retraining on recent data restored a boundary appropriate to the current
regime, which recovered recall. So the metric that moved is the metric that
was actually failing.

**The point to land:** if I had reported only accuracy, this system would look
useless (+1%) and the failure would have been invisible. Metric choice is not
cosmetic — it determines whether you can see the problem.

### Q2. "Why both KS and PSI? Isn't one test enough?"

**Answer:** They answer different questions.

- **KS test** → "did the distribution change?" Gives a p-value. Its weakness
  is sample-size sensitivity: with 45,000 rows, a mean shift of 0.02 comes
  back p < 0.001. Statistically significant, operationally meaningless.
- **PSI** → "did it change *enough to matter*?" An effect size, independent of
  n. Convention: <0.10 negligible, 0.10–0.25 moderate, >0.25 major.

I require **both** — significant AND material. Without the PSI floor the
detector fires on essentially every batch at this sample size, and an alarm
that always fires is the same as no alarm.

There's a test for exactly this in `tests/test_drift_detector.py::
test_psi_gate_suppresses_trivial_but_significant_shift` — 60,000 rows with a
0.02 shift, asserting the gated detector never reports more drift than the
ungated one.

### Q3. "Why won't a retrained model auto-deploy?"

**Answer:** Because retraining does not guarantee improvement. In my run, **5
of 12 retrains were rejected by the gate.** Auto-promoting those would have
swapped the production model 5 extra times for no gain.

The cost isn't just wasted compute. Every unnecessary swap destroys
attributability — when performance drops next week, you can't tell whether it
was drift or one of your own unvalidated swaps. Stable model lineage is an
operational asset.

The gate: challenger must beat champion by ≥1 point **on the same unseen
batch**. Both scored on identical data. Comparing a challenger's fresh
validation score against the champion's months-old score would be rigged —
the challenger's validation data is from the current regime, the champion's is
from the old one.

### Q4. "Why is your validation split temporal instead of random?"

**Answer:** Random splitting on time-series data leaks the future into
training. If a row from March is in train and a row from February is in test,
the model has seen information that wouldn't exist at prediction time. The
score comes back inflated and the model fails in production.

Concretely for this project: ELEC2 is half-hourly, so adjacent rows are highly
autocorrelated. A random split puts near-duplicate rows in both train and
test, and the model gets credit for near-memorization.

My split is the last 20% chronologically — `_temporal_split()` in
`src/models/trainer.py`. Same reason batches are never shuffled.

**Bonus point to make:** this is exactly the bug I'd look for if someone showed
me a suspiciously high score on time-series data.

### Q5. "Walk me through what happens when a batch arrives."

```
1. Detect  — KS + Chi² + PSI on 5 monitored features vs the reference window
2. Score   — champion predicts on the batch; accuracy/F1 logged to Postgres
3. Decide  — trigger if (>50% features drifted) OR (accuracy fell >5 points)
             suppressed if within the 2-batch cool-down
4. Retrain — challenger trained on the last 6 batches (sliding window)
5. Gate    — challenger vs champion on this same batch; promote only if +1pt
6. Persist — drift_runs + feature_drift + predictions rows written
7. Append  — batch joins history AFTER scoring, never before
```

Step 7 matters: if the batch entered history before scoring, the model would
be evaluated on data it trained on. That ordering is deliberate.

---

## Part 2 — Harder follow-ups

**"Why a sliding window instead of all history?"**
Under concept drift, old data is actively harmful — it encodes a
feature→label relationship that no longer holds. Training on everything drags
the model toward a regime that no longer exists. 6 batches ≈ 6 months here.
It's a hand-tuned constant, which is a real limitation; ADWIN would adapt it
automatically.

**"What's the cool-down for?"**
During sustained drift, every batch trips the gate. Without a cool-down you
retrain constantly, thrash the champion, and burn compute. 2 batches is the
minimum spacing. You can see it working: retrains at 15, 17, 19 not 15, 16, 17.

**"Why exclude `day` and `period` from monitoring?"**
They're cyclical calendar features. They shift between batches by
construction — a batch spanning different weekdays will always "drift" on
`day`. That's a guaranteed false positive. They stay available to the *model*
as features; they're just not *monitored*. Distinguishing "feature the model
uses" from "feature worth alerting on" is a real design decision.

**"Why Postgres rather than files?"**
The question "which model was serving when `nswprice` drifted?" is one SQL
join across `model_versions`, `drift_runs`, `feature_drift`. With JSON files
it's a filesystem hunt. Also: concurrent writes from the API and pipeline,
and the champion flag needs a real transaction so two models can never both
be champion — that invariant is unit-tested.

**"How does this fail?"**
Several ways, and I'd flag them before deploying:
1. **Labels assumed instant.** Biggest gap. Real systems get labels late or
   never. You'd need proxy metrics or delayed-label evaluation.
2. **Reference window follows the champion.** After promotion the reference
   resets to the new training window, so gradual drift can be silently
   absorbed — each step looks small relative to a moving reference.
3. **Single dataset.** ELEC2 has abrupt regime change. Gradual or recurring
   drift may behave differently.
4. **The 1-point promotion margin is arbitrary.** Should be derived from the
   variance of the batch score, not picked by hand.

**"What would you do with more time?"**
ADWIN/KSWIN adaptive windowing benchmarked against my fixed window;
delayed-label simulation; shadow deployment so challengers see live traffic
before promotion; per-feature attribution of *performance* loss rather than
just distribution shift.

---

## Part 3 — Concepts to actually understand

Don't memorize these definitions — be able to explain them to a
non-specialist.

**Data drift vs concept drift.** Data drift = P(X) changes, the input
distribution moves. Concept drift = P(y|X) changes, the *relationship* moves.
You can have either without the other. Detecting only data drift misses the
case where inputs look normal but the mapping broke — which is why I monitor
performance too.

**Kolmogorov-Smirnov.** Compares two empirical CDFs; the statistic is the
maximum vertical distance between them. Non-parametric — no normality
assumption. Sensitive to n.

**PSI.** `Σ (curr% − ref%) × ln(curr% / ref%)` over bins taken from the
*reference* quantiles. Reference bins matter: the question is how the new data
redistributes across the buckets the model was trained on.

**Champion-challenger.** Incumbent serves traffic; candidate is evaluated
against it on identical data; promotion only on a demonstrated win. Standard
in credit risk and fraud, where unvalidated swaps are a compliance problem.

**Why F1 for imbalanced data.** Accuracy is dominated by the majority class.
F1 is the harmonic mean of precision and recall, so it collapses if either
collapses — which is what makes it sensitive to the majority-class collapse
that accuracy hides.

---

## Part 4 — Run it yourself before any interview

Do this at least once. If you've never seen it run, it shows.

```bash
python scripts/download_data.py
python experiments/run_experiment.py      # watch the retrain/reject decisions
pytest tests/ -v                          # 29 tests
docker compose up --build                 # API :8000, dashboard :8501
```

Then deliberately break things and watch what happens — this is where real
understanding comes from:

```bash
# 1. Remove the PSI gate. Drift should fire far more often.
#    src/drift/detector.py -> psi_threshold=0.0

# 2. Remove the promotion gate. More swaps, no better results.
#    src/pipeline/monitor.py -> promotion_margin=0.0

# 3. Make the retrain window huge (e.g. 25 batches). Performance should
#    DROP under drift, because stale data dominates. Confirms the sliding
#    window is doing real work.

# 4. Switch to a random split in trainer.py. Validation accuracy will jump.
#    That jump is the leakage. Understand why it's fake.
```

Experiment 4 is the most valuable one you can run. It teaches you what data
leakage feels like from the inside — and it's the exact bug currently sitting
in your Customer-Churn repo, where `reports/model_metrics.json` claims 0.95
ROC-AUC on a dataset whose documented ceiling is ~0.85.

---

## Part 5 — Numbers to know cold

| Fact | Value |
|---|---|
| Dataset | ELEC2 (Harries, 1999), NSW electricity market |
| Size | 45,312 records, 8 features |
| Batches | 31 total, 1,440 records each (≈30 days) |
| Bootstrap | first 3 batches, 28 evaluated |
| Static F1 → DriftGuard F1 | 0.4515 → 0.5789 (+28.2%) |
| Static acc → DriftGuard acc | 0.6988 → 0.7064 (+1.1%) |
| Retrains triggered | 12 |
| Promotions | 7 (5 rejected by gate) |
| Final model version | v13 |
| Most drifted feature | `nswprice`, 28/28 batches |
| Tests | 29 passing |
| Drift gate | KS p<0.05 AND PSI≥0.10, >50% of features |
| Performance gate | accuracy drop >5 points |
| Promotion margin | +1 point on the same unseen batch |

If someone asks a number you don't remember, say "let me pull it up" and open
`reports/experiment_summary.json`. Checking your own artifacts is far better
than guessing — guessing wrong on your own project is fatal, looking it up is
normal engineering.
