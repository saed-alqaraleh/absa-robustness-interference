# Stage-1 Protocol — Scope-Contrastive Contract Training (SCCT)

**Protocol version:** 1.0 (FROZEN)  
**Date frozen:** 7 October 2026  
**Project:** Reciprocal Robustness Interference in Aspect-Based Sentiment Analysis  
**Purpose:** Prespecify the decisive Stage-1 method pilot before examining Stage-1 test outcomes.

---

## 1. Scope of this protocol

This document settles the seven open Stage-1 decisions:

1. the BAT-style comparison design;
2. Fei et al.'s exact loss and its RoBERTa-compatible implementation;
3. the exact SCCT construction and loss;
4. the Stage-1 pass/fail thresholds;
5. the P4 and P8 thresholds;
6. whether hyperparameters are tuned per domain;
7. how the protocol is frozen/registered.

The protected ARTS test sets are evaluation-only. No Stage-1 method, hyperparameter, checkpoint, or stopping decision may be selected from ARTS performance.

---

# DECISION 1 — BAT-style comparison design

## Decision

Use **BAT-Triplet**, not BAT-Pairwise, as the primary BAT comparator.

Chen, Ji, and Evans (EMNLP 2022) define Balanced Adversarial Training (BAT) by treating a label-preserving adversarial example as a positive and a label-changing adversarial example as a negative. Their triplet form was the stronger of the two reported BAT variants and is therefore the fairest primary comparator.

## ABSA adaptation

For a clean source/query pair \((x,a,y)\):

- **anchor:** original input \(x\);
- **positive \(x_f\):** a target-opinion *polarity-preserving intensification* generated inside the queried aspect's opinion scope; its gold label remains \(y\);
- **negative \(x_o\):** target-opinion reversal (RevTgt) from the same source; its gold label is \(y'\neq y\).

This design deliberately uses **different operators at the same target location**. It therefore tests whether generic balanced sensitivity/invariance pairing is sufficient, while SCCT tests the narrower hypothesis that holding the operator fixed and varying its location is important.

The representation is the L2-normalized final [CLS] representation \(h(\cdot)\). Define cosine distance

\[
d(u,v)=1-\cos(u,v).
\]

The adapted BAT triplet term is

\[
\mathcal L_{\mathrm{BAT-trip}}
=
\left[
d(h(x),h(x_f))
-
d(h(x),h(x_o))
+
m_{\mathrm{BAT}}
\right]_+ .
\]

The complete comparator objective is

\[
\mathcal L_{\mathrm{BAT}}
=
\mathcal L_{\mathrm{CE}}
+
\lambda_{\mathrm{BAT}}\mathcal L_{\mathrm{BAT-trip}} .
\]

### Fixed BAT choices

- \(m_{\mathrm{BAT}}=1.0\), following the original BAT paper.
- Tune only \(\lambda_{\mathrm{BAT}}\in\{0.25,0.5,1.0\}\).
- The original clean sentence is an anchor already present in the base training data and does **not** count as an added augmentation record.
- Both generated derivatives do count against the common Stage-1 added-record budget.
- BAT-Pairwise is optional secondary/appendix evidence only; it is not part of the Stage-1 go/no-go decision.

---

# DECISION 2 — Fei et al. exact contrastive loss

## Decision

Implement the **opinion-guided intra-aspect and inter-aspect contrastive losses** from Fei et al., *On the Robustness of Aspect-based Sentiment Analysis: Rethinking Model, Data, and Training* (TOIS), at RoBERTa's final [CLS] representation.

Do **not** attempt to reproduce Fei et al.'s syntax-fusion/structure-guided loss in Stage 1, because the Stage-1 backbone is RoBERTa rather than their syntax-aware USGCN. The comparator is therefore named **Fei-Opinion**, not "full Fei et al."

## Mapping of Fei et al.'s pseudo instances

For source \(X_i^o\):

- \(X_{i,j}^{a,+}\): target-opinion modification that preserves the target polarity — implemented here as target-opinion intensification;
- \(X_{i,k}^{a,-}\): target-opinion reversal — RevTgt;
- \(X_{i,k}^{m}\): non-target aspect addition — AddDiff.

Let \(r(\cdot)\) be the L2-normalized final [CLS] representation and

\[
\mathrm{Sim}(u,v)=\frac{u^\top v}{\lVert u\rVert\lVert v\rVert}.
\]

Fei et al.'s opinion-guided **intra-aspect** contrastive term is implemented as

\[
\mathcal L_{F}^{\mathrm{intra}}
=
-\sum_{i}\sum_{j}
\log
\frac{
\exp\!\left(\mathrm{Sim}(r(X_{i,j}^{a,+}),r(X_i^o))/\mu\right)
}{
\sum_k
\exp\!\left(\mathrm{Sim}(r(X_{i,k}^{a,-}),r(X_i^o))/\mu\right)
}.
\]

Their opinion-guided **inter-aspect** term is implemented as

\[
\mathcal L_{F}^{\mathrm{inter}}
=
-\sum_{i}\sum_{j}
\log
\frac{
\exp\!\left(\mathrm{Sim}(r(X_{i,j}^{a,+}),r(X_i^o))/\mu\right)
}{
\sum_k
\exp\!\left(\mathrm{Sim}(r(X_{i,k}^{m}),r(X_i^o))/\mu\right)
}.
\]

The RoBERTa-compatible Stage-1 objective is

\[
\mathcal L_{\mathrm{Fei}}
=
\mathcal L_{\mathrm{CE}}
+
\lambda_F
\left(
\mathcal L_F^{\mathrm{intra}}
+
\mathcal L_F^{\mathrm{inter}}
\right).
\]

### Fixed Fei-Opinion tuning grid

- \(\mu\in\{0.05,0.10,0.20\}\);
- \(\lambda_F\in\{0.25,0.50,1.00\}\).

This comparator is deliberately strong: it includes both the target-polarity contrast and the target/non-target contrast available at the opinion representation level.

---

# DECISION 3 — Exact SCCT design

## 3.1 Common source eligibility

Before outcome-bearing runs, build one Stage-1 common source pool per domain.

A source is eligible only if the frozen transformation pipeline can generate all four required derivatives:

1. **TgtInt:** target-opinion intensification, target label unchanged;
2. **RT:** target-opinion reversal, target label changes;
3. **RN-rev:** non-target-opinion reversal using the *same reversal operator* as RT, queried target label unchanged;
4. **AD:** AddDiff, queried target label unchanged.

This common eligibility rule prevents comparator-specific source-population advantages.

Let the resulting pool size be \(|P_D|\) for domain \(D\).

## 3.2 Common Stage-1 record budget

The historical budgets are

- Restaurant: \(B_R=1004\);
- Laptop: \(B_L=419\).

Because the competing constructions use pairs or triplets, define the Stage-1 budget as the largest multiple of six that is both no larger than the historical budget and feasible for every method from the common source pool:

\[
B_D^{(1)}
=
6
\left\lfloor
\frac{\min(B_D,\,2|P_D|)}{6}
\right\rfloor .
\]

If the common pool is large enough, this gives:

- Restaurant: \(B_R^{(1)}=1002\);
- Laptop: \(B_L^{(1)}=414\).

If eligibility makes either value smaller, the formula above—not observed model performance—determines the reduced budget. The final realized values and pool hashes are recorded before training.

All Stage-1 augmentation methods receive exactly \(B_D^{(1)}\) added records.

## 3.3 Mixed baseline

Mixed receives one third RT, one third RN-rev, and one third AD records:

\[
B_D^{(1)}/3
\]

from each family.

Sources are sampled independently by family from the common pool using the frozen seed. Pairing is **not enforced**.

## 3.4 Same-operator paired-data condition

Select

\[
N_D=B_D^{(1)}/3
\]

source sentences. For each selected source, add exactly:

- RT;
- RN-rev;
- AD.

Thus each source contributes a triplet and the total remains \(B_D^{(1)}\).

This condition uses ordinary cross-entropy only. It tests the effect of matched source allocation before adding contract-specific losses.

## 3.5 Full SCCT objective

The same triplet data are used.

Let

\[
p_x=f(x),\qquad
p_{RN}=f(x^{RN}),\qquad
p_{AD}=f(x^{AD}),
\]

with \(f(\cdot)\) the class-probability vector.

For invariance, use symmetric Jensen-Shannon divergence:

\[
JS(p,q)
=
\frac12 KL(p\|m)
+
\frac12 KL(q\|m),
\qquad
m=\frac12(p+q).
\]

The invariance term is

\[
\mathcal L_I
=
\frac12
\left[
JS(p_x,p_{RN})
+
JS(p_x,p_{AD})
\right].
\]

For target reversal, if \(y\) is the original label and \(y'\) is the reversed gold label,

\[
\mathcal L_S
=
\left[
m_S-
\left(
z_{y'}(x^{RT})-z_y(x^{RT})
\right)
\right]_+ .
\]

The full SCCT objective is

\[
\boxed{
\mathcal L_{\mathrm{SCCT}}
=
\mathcal L_{\mathrm{CE}}
+
\lambda_I\mathcal L_I
+
\lambda_S\mathcal L_S
}
\]

with

\[
m_S=1.0.
\]

### SCCT tuning grid

\[
\lambda_I,\lambda_S
\in
\{0.25,0.50,1.00\}.
\]

This gives nine SCCT candidates. \(m_S\) is fixed rather than tuned.

## 3.6 Batching

SCCT and paired comparators use a grouped sampler so that the original anchor and required derivatives for a source are available in the same optimization step.

The ordinary Mixed baseline uses the same batch size and optimizer schedule but no pair/group constraint.

## 3.7 Fixed training controls

Across Stage-1 methods, hold fixed:

- encoder checkpoint;
- optimizer and learning-rate schedule;
- batch size;
- maximum sequence length;
- epoch limit / early stopping;
- precision mode;
- clean base training records;
- total added-record budget;
- Stage-1 seeds;
- checkpoint-selection rule.

Only the augmentation construction and prespecified objective differ.

---

# DECISION 4 — Stage-1 pass-rule thresholds

## Primary endpoints

1. **Worst-family accuracy**

\[
W=\min(A_{RT},A_{RN},A_{AD}).
\]

2. **RT/RN joint correctness**

For a common source, both the RT sensitivity requirement and RN invariance requirement must be correct.

The two domains receive equal weight in pooled Stage-1 summaries.

## Core comparators for the decisive rule

The core comparator set is:

- Mixed;
- BAT-Triplet;
- Fei-Opinion.

For each primary endpoint, define the **strongest comparator** as the member of this set with the highest equal-domain mean.

## SCCT PASS rule

Stage 1 passes only if **all** of the following are met:

### A. Worst-family gain
Against the strongest comparator,

\[
\Delta W \ge +2.0\ \text{percentage points}
\]

on the equal-domain mean, and the 95% paired/source-cluster bootstrap CI for the pooled effect has lower bound \(>0\).

### B. Joint-correctness gain
Against the strongest comparator,

\[
\Delta J \ge +2.0\ \text{percentage points}
\]

on the equal-domain mean, and the 95% paired/source-cluster bootstrap CI for the pooled effect has lower bound \(>0\).

### C. Domain consistency
For neither Restaurant nor Laptop may SCCT be worse than the corresponding strongest comparator by more than

\[
1.0\ \text{percentage point}
\]

on either primary endpoint.

### D. Clean-performance guardrail
SCCT clean accuracy may not be more than

\[
1.0\ \text{percentage point}
\]

below Mixed.

### E. Direction across seeds
At least two of the three Stage-1 seeds must favor SCCT over the strongest comparator on **both** primary endpoints in the equal-domain summary.

If A–E are satisfied, proceed to the full multi-encoder confirmation.

If same-operator paired data beat the core comparators but adding the SCCT loss does not improve further, the simpler paired-data construction becomes the candidate method.

If the paired-data condition and full SCCT both fail the core comparator set, stop the method-expansion path and retain the project as an analysis/theory paper.

---

# DECISION 5 — P4 and P8 thresholds

## P4 — shared-operator mechanism

P4 asks whether the RevNon→RevTgt penalty is carried primarily by the **reversal** component rather than by non-target intensification.

For each domain define the magnitude of the original full RevNon→RevTgt penalty as

\[
P_{\mathrm{full}}
=
\left|
A_{RT}(\mathrm{RevNon\ train})
-
A_{RT}(\mathrm{CleanMatched})
\right|.
\]

Define corresponding magnitudes for:

- reversal-only non-target training: \(P_{\mathrm{rev}}\);
- intensification-only non-target training: \(P_{\mathrm{int}}\).

### P4 confirmatory threshold

P4 is considered confirmed only if, on the equal-domain mean:

\[
\frac{P_{\mathrm{rev}}}{P_{\mathrm{full}}}\ge 0.70,
\]

\[
\frac{P_{\mathrm{int}}}{P_{\mathrm{full}}}\le 0.30,
\]

and

\[
P_{\mathrm{rev}}-P_{\mathrm{int}}\ge 8.0\ \text{pp},
\]

with the 95% bootstrap CI for \(P_{\mathrm{rev}}-P_{\mathrm{int}}\) excluding zero.

If the direction holds but one magnitude threshold fails, report P4 as partial support rather than redefining the threshold.

---

## P8 — dose response

The original 0/25/50/75/100% wording is replaced by a design that preserves both contracts.

For each reciprocal direction, total added records remain fixed at \(B_D^{(1)}\).

Let \(q\in\{0,0.10,0.25,0.50\}\) be the **share of the total added-record budget assigned to the complementary contract**.

For an RT-dominant schedule:

- \(qB\) RN-rev records are added;
- \(qB\) RT records are drawn from the same sources, forming matched pairs;
- the remaining \((1-2q)B\) records are additional RT records.

Thus:

- \(q=0\): fully one-sided RT;
- \(q=0.25\): 75% RT / 25% RN, with the RN records source-paired to an equal number of RT records;
- \(q=0.50\): fully balanced RT/RN pairs.

The RN-dominant direction is constructed symmetrically.

Define cross-contract penalty magnitude \(P(q)\) relative to Clean-Matched and recovery

\[
R(q)
=
1-\frac{P(q)}{P(0)}.
\]

### P8 confirmatory threshold

P8 is confirmed if the equal-domain, equal-direction mean satisfies:

\[
R(0.25)\ge 0.50,
\]

and the 95% bootstrap CI for \(R(0.25)\) has lower bound \(>0.25\).

Strong supporting evidence is predefined as

\[
R(0.50)\ge 0.75.
\]

The observed \(q=0.10\) point is descriptive evidence about how early the curve bends; no success threshold is attached to it.

No post-hoc dose value may replace \(q=0.25\) as the confirmatory P8 test.

---

# DECISION 6 — Hyperparameter tuning across domains

## Decision

**Do not tune hyperparameters separately for Restaurant and Laptop.**

Every method receives one shared hyperparameter configuration across both domains.

This is deliberate: per-domain tuning would increase tuning opportunity, make the method less portable, and weaken the claim that the mechanism is domain-general.

## Tuning data

- Use only train/dev material.
- Generate robustness-dev derivatives from the development split using the training operators.
- ARTS is never used for tuning.
- Use tuning seed **101**, which is not one of the Stage-1 confirmatory seeds.

## Stage-1 confirmatory seeds

\[
13,\ 37,\ 73.
\]

## Shared selection score

For candidate hyperparameters compute, separately by domain:

- worst robustness-dev family accuracy;
- RT/RN robustness-dev joint correctness;
- clean-dev accuracy.

Discard any candidate whose equal-domain clean-dev accuracy is more than 1.0 pp below the corresponding CE-only matched control.

For remaining candidates define

\[
S_{\mathrm{dev}}
=
\frac12
\left[
\overline{W}_{dev}
+
\overline{J}_{dev}
\right],
\]

where the bars denote **equal weighting of Restaurant and Laptop**.

Choose the single candidate maximizing \(S_{\mathrm{dev}}\).

Tie-break order:

1. higher \(\overline{W}_{dev}\);
2. higher clean-dev accuracy;
3. smaller regularization weight(s).

The chosen setting is then frozen for all three confirmatory seeds and both domains.

---

# DECISION 7 — Protocol registration / freeze

## Decision

No OSF registration will be used.

This dated, versioned protocol records the internal Stage-1 experimental plan before outcome-bearing runs. It is not an externally registered protocol. The version history and amendments document subsequent changes.

The protocol file is **Stage1_SCCT_Protocol_v1.0_2026-10-07.md**.

## Amendment rule

After the first Stage-1 outcome-bearing run begins:

- v1.0 is never overwritten;
- any change becomes v1.1, v1.2, etc.;
- the amendment must state the date, exact change, and reason;
- any change made after viewing Stage-1 outcomes is labeled explicitly as **post-outcome exploratory** and cannot redefine the preregistered pass rule.

---

# Stage-1 execution table

| Condition | Added-data construction | Special loss | Included in decisive comparison? |
|---|---|---|---|
| Clean-Matched | matched clean exposure | CE | reference |
| Mixed | RT + RN-rev + AD, independently sampled | CE | **yes** |
| BAT-Triplet | TgtInt + RT paired by source | CE + BAT triplet | **yes** |
| Fei-Opinion | TgtInt + RT + AD | CE + Fei intra/inter contrast | **yes** |
| Same-operator paired data | RT + RN-rev + AD triplets | CE | **yes, method ablation** |
| SCCT | same RT + RN-rev + AD triplets | CE + JS invariance + sensitivity margin | **yes, proposed method** |
| Consistency baseline | matched invariant edits | CE + consistency | secondary strong baseline |
| GroupDRO | family groups | GroupDRO objective | secondary strong baseline |
| PCGrad | family gradients | projected gradients | secondary strong baseline |

CAGrad is not required on the critical path. It may be added later only if implementation is already validated before Stage-1 outcomes are inspected.

---

# Required pre-run audit

Before launching confirmatory Stage-1 runs, save a machine-readable manifest containing:

- realized Restaurant and Laptop common-pool sizes;
- realized \(B_D^{(1)}\);
- source IDs selected for every condition;
- transformation family/operator for every added record;
- gold label before and after edit;
- seed;
- hyperparameter configuration;
- SHA-256 of every generated training file;
- code commit/archive hash;
- protocol SHA-256.

If any common-pool or transformation-validity check fails, fix the data-generation pipeline **before** training. Do not alter the statistical pass thresholds.

---

# Literature basis used to settle the comparator decisions

- Chen, H., Ji, Y., & Evans, D. (2022). *Balanced Adversarial Training: Balancing Tradeoffs between Fickleness and Obstinacy in NLP Models.* EMNLP 2022. arXiv:2210.11498.
- Fei, H., Chua, T.-S., Li, C., Ji, D., Zhang, M., & Ren, Y. *On the Robustness of Aspect-based Sentiment Analysis: Rethinking Model, Data, and Training.* ACM Transactions on Information Systems, 41(2). DOI: 10.1145/3564281.
- The project's frozen strengthening plan and Gate-0 novelty audit, finalized 6–7 October 2026.

---

## Frozen Stage-1 thesis

> Holding the sentiment intervention fixed while varying only whether it falls inside or outside the queried aspect's opinion scope should produce a more balanced sensitivity–invariance model than unpaired mixing, generic label-preserving/label-changing pairing, or mixed-operator inter-aspect contrast under the same added-record budget.

**END OF FROZEN PROTOCOL v1.0**
