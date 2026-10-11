# Bias-variant comparison findings

**Date**: 2026-06-28
**Goal**: Test whether the negative half of the production logit-bias map (the -2.5 entries on tokens containing HSK 4+ multi-char words) is actually doing useful work, or if it could be removed for simpler tuning.

## Three variants tested

All sharing the same positive map (`+3` on tokens whose decoded text is all-HSK-1-N chars), differing only in how compound HSK 4+ tokens are treated:

| Variant | Compound HSK 4+ token bias | Implementation |
|---|---|---|
| **A** (production) | `-2.5` | Substring check → assign NEGATIVE bias, push model away |
| **B1** (naive remove) | `+3` | No substring check → tokens fall through to positive char rule |
| **B2** (suppress) | `0` (neutral) | Substring check still runs but assigns nothing instead of negative |

Test chapter: Solo Leveling Chapter 1, run in both Chinese-source (KuaiKan, `comic-next/185037`) and English-source (existing homepage demo at `outputs/hsk-comparison`) at HSK 3 target.

Production-equivalent pipeline: full batching (`SIMPLIFY_BATCH_TARGET_BLOCKS=15-30`) with per-batch plot summaries pulled from `chapter-context.json`.

## Quantitative findings

### Chinese chapter (CN input, target HSK 3)

| Variant | Total chars | KEEP occurrences | KEEP unique | non-KEEP HSK 4+ occurrences | Approx % above-target |
|---|---|---|---|---|---|
| A | 1000 | 59 | 22 | 5 | 1.0% |
| B1 | 1002 | 59 | 22 | 7 | 1.4% |
| B2 | 1000 | 59 | 22 | 4 | 0.8% |

**Verdict for CN input**: differences are noise. All three variants land at <1.5% above-target — well within independent-reading range for an HSK 3 learner. **Ship A (no change needed).**

### English chapter (EN input, target HSK 3, single-pass translate+simplify)

| Variant | Total chars | KEEP occurrences | KEEP unique | non-KEEP HSK 4+ occurrences | Approx % above-target |
|---|---|---|---|---|---|
| **A** | 1030 | 27 | 9 | **38** | **7.4%** |
| B1 | 1039 | 29 | **8** | 63 | 12.1% |
| B2 | 1043 | 25 | 9 | 51 | 9.8% |

**Verdict for EN input (quantitative)**: A wins by a lot.
- Removing the negative bias (B1) jumps non-KEEP HSK 4+ usage by 66%
- B2 sits in between (+34%)
- For HSK 3 readability thresholds (<5% independent, 5-10% instructional, >10% frustration), only A stays in instructional zone

## Qualitative findings (HSK 3 learner perspective)

Reading the actual outputs end-to-end as if I were the target learner:

- **Chinese chapter**: all three variants felt the same. ~1-2 lines per chapter where A used a slightly harder word (e.g. `丢命`) vs B1/B2's wordier-but-simpler paraphrase (`没命了`). Neither pattern is clearly better for a learner. Wash.

- **English chapter**:
  - **A**: Most words I knew (HSK 1-3 compliant), but occasional weird phrasings like `做房子` (awkward Chinese — you don't "make" a house). Also tended to drop KEEP-term context: `金相植` → `金` (which makes character tracking harder mid-chapter). Sometimes uses idioms I haven't learned (`丢脸`, `真该死`).
  - **B1**: Beautiful, faithful Chinese — but vocabulary wall every few lines: `协会`, `成员`, `等级`, `恢复`, `怀孕`, `糟糕`. As an HSK 3 reader I'd hit ~5 unknown words per page. Reading flow dies.
  - **B2**: Best of both worlds in many spots. Uses `公会` (HSK 3-friendly) like A, but keeps `地城突袭` (KEEP term) like B1. Reads smoothly more often. The 9.8% above-target rate is borderline but workable.

**Qualitative verdict diverges slightly from quantitative**: B2 actually felt the best for EN input as a learner, even though A has the lower above-target count. The quantitative metric undercounts:
- Sentence-level difficulty (long sentences with multi-clause structure even when individual words are simple)
- Idiomatic vs literal vocabulary (`丢脸` is technically HSK 4+ once and counts the same as `恢复`, but `丢脸` is far harder to decode)
- KEEP-term consistency for character tracking

## Recommendation

| Input | Recommended | Why |
|---|---|---|
| Chinese | **A** (no change) | All three variants effectively identical (<1.5% above-target). Production behavior is fine. |
| English | **A**, with **B2** as a defensible alternative | Quantitatively A wins (lower HSK 4+ count). Qualitatively B2 felt smoother to read because it preserves KEEP-term context and avoids A's awkward simplification patterns. Worth a real-learner A/B test before changing production. |

**Most useful insight**: the bias-map negative half is doing real work for English input but almost nothing for Chinese input. The negative bias prevents `~25` extra non-KEEP HSK 4+ word occurrences per chapter in EN input — significant for hitting the HSK target. For CN input the source is already mostly HSK 3, so there's nothing to push against.

## What was NOT tested

- **Other chapter genres**. Solo Leveling Ch1 is action manga with short bubbles. Different domains (educational manhua, slice-of-life, news-style) might show different patterns.
- **Other HSK target levels**. Only tested HSK 3. HSK 1 with B-variants is theoretically much more impactful (huge bias-map difference) but wasn't run on EN input.
- **B3** (exact-dictionary-word positive only). Implemented and bias maps built but not run on chapter outputs.
- **Per-language production change**. The cleanest fix would route different bias variants based on `inputLanguage`, but no production change shipped — keeping current A for all cases.
- **Real-learner evaluation**. My HSK 3 reactions are simulated. A real beginner-intermediate learner test would settle the A-vs-B2 EN call.

## Data + scripts

- Bias maps (all variants): `data/hsk-token-bias-{1,1to2,1to3,1to4}.deepseek-v4{,.b1,.b2,.b3}.json`
- Builder: `build-bias-map.js` (supports `MODE=b1|b2|b3` env var)
- Chinese chapter outputs: `outputs/bias-comparison/{A-current,B1-naive,B2-suppress}.json`
- English chapter outputs: `outputs/bias-comparison-en/{A-current,B1-naive,B2-suppress}.json`
- Side-by-side comparisons: `outputs/bias-comparison{,-en}/bias-comparison.{md,html,txt}`
- Runner: `scripts/bias-variant-run.js` (uses `BIAS_MAP_VARIANT` env var, supports `INPUT_LANG=english`)
- Quantitative HSK 4+ counter: `scripts/bias-hsk-count.js`
- Comparison report generator: `scripts/bias-variant-compare.js`

To re-run: see the scripts' usage docstrings. Each variant is one process with the `BIAS_MAP_VARIANT` env var set differently. Total LLM cost across all three variants on both chapters: ~$0.20.

## How to integrate variant-switching if you ever ship it

The infrastructure is already in place:

```ts
// src/simplifier.ts:unifiedBiasFilePath
const variant = process.env.BIAS_MAP_VARIANT || '';
return path.join(process.cwd(), 'data', `hsk-token-bias-${range}${tokenFileSuffix()}${variant}.json`);
```

If you ever decide per-language variant switching is worth it:

```ts
const variant = inputLanguage === 'english'
  ? (process.env.BIAS_MAP_VARIANT_EN || '')
  : (process.env.BIAS_MAP_VARIANT_ZH || '');
```

But empirically the current single-variant production behavior is fine for both inputs, so this isn't urgent.
