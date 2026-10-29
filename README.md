# Lending Club Credit Risk Analytics

An end-to-end credit analytics study on ~50,000 Lending Club loans from 2012–2013. It runs an A/B test of two approval policies, segments borrowers with PCA and k-means, and builds a deep-learning default model, with each part feeding the next.

**Context:** MSc Business Analytics, Warwick Business School · group project (team of 7) · R and Python

## 1. A/B test: conservative vs expanded approval policy

| | Policy A | Policy B |
|---|---|---|
| Rule | Grades A–B, DTI < 25 | Grades A–D, DTI < 25 |
| Approval rate | 44.3% | 74.9% |
| Mean realised profit per loan | £946 | £982 |
| Default rate | **9.7%** | 13.5% |
| Profit per £ lent | 7.67p | 7.50p |

- **Profit (primary OEC):** the Welch t-test shows no significant difference (t = −1.35, p = 0.18, Cohen's d = −0.01).
- **Default rate (secondary OEC):** significantly higher under Policy B (χ² = 189.4, p < 0.001, a 3–4 percentage-point gap).

**Conclusion:** widening approval to grades C–D takes on more risk without a per-loan profit gain.

→ [`notebooks/q1_ab_testing.Rmd`](notebooks/q1_ab_testing.Rmd)

## 2. Borrower segmentation: PCA + k-means

- 11 underwriting variables covering affordability, credit usage, risk behaviour and loan terms.
- Checked PCA suitability with KMO/MSA, and dropped `pub_rec` (MSA = 0.33) from the inputs but kept it for profiling.
- Removed 1,071 multivariate outliers with Mahalanobis distance (p < 0.001), standardised, ran PCA, then k-means on the components.
- Profiled 4 segments: **Highest-cost / weakest credit**, **Mainstream / lower-risk**, **Delinquency-history** and **High-inquiry / active credit**.

**Key insight:** the policy effect isn't uniform. Overall, Policy B wins on total profit, but that's carried by the mainstream segment (+£9.76m). In the weakest-credit segment, expanding approvals *loses* a further £0.86m, so a segment-aware policy beats both.

→ [`notebooks/q2_cluster_analysis.Rmd`](notebooks/q2_cluster_analysis.Rmd) · cluster labels in `data/cluster_labels.csv`

## 3. Default prediction: deep learning in PyTorch

- **Time-based split** (train May–Nov 2012, validation Nov–Dec 2012, test Dec 2012–Feb 2013), so the model is always tested on loans issued *after* the ones it learned from.
- `ColumnTransformer` preprocessing (median imputation, scaling, one-hot encoding) gives 122 features.
- MLP with layers 256 → 128 → 64 → 32, BatchNorm, ReLU and dropout. Class imbalance handled with a weighted loss (`pos_weight = 5.2`).
- Decision threshold chosen on validation, with a threshold-sensitivity table on test showing the approval vs risk trade-off.

| Test metric | Value |
|---|---|
| AUC-ROC | 0.66 |
| Recall (defaults caught) | 0.60 |
| Precision | 0.22 |

AUC is evaluated within each borrower segment from part 2 (0.63–0.67), so performance is consistent across segments. A modest AUC is realistic for application-time data only.

→ [`notebooks/q3_deep_learning_model.ipynb`](notebooks/q3_deep_learning_model.ipynb)

## Run it

```bash
# R parts (RStudio): open notebooks/*.Rmd and knit
#   needs tidyverse, broom, effectsize, psych, corrplot, cluster, factoextra
# Python part
pip install pandas numpy scikit-learn torch matplotlib
jupyter notebook notebooks/q3_deep_learning_model.ipynb
```

The notebooks read the CSVs from the working directory, so run them from `data/` or update the paths. Rendered HTML reports for parts 1 and 2 are in `notebooks/`.

## Data

`data/early_2012_2013_loan_sample_with_outcome.csv` is a sample of public Lending Club loan data with the realised outcome.

## Tech stack

R · tidyverse · PCA · k-means · hypothesis testing · Python · PyTorch · scikit-learn
