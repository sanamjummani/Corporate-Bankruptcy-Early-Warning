# Data

## Dataset — Corporate Bankruptcy Prediction

Two files used in this project, both provided as part of the MGMT 571 Kaggle competition at Purdue University.

---

### bankruptcy_Train.csv — Training Data

**10,000 firms × 65 columns** — known attrition outcomes included.

| Column | Description |
|---|---|
| `Attr1` – `Attr64` | 64 financial ratio features per firm (anonymised) |
| `class` | **Target variable** — 1 = bankrupt, 0 = solvent |

**Class distribution:**
- Bankrupt (class = 1): 473 firms — 4.73%
- Solvent (class = 0): 9,527 firms — 95.27%

This severe class imbalance is intentional — it reflects real-world bankruptcy rates. Accuracy is not a useful metric here. ROC-AUC is.

---

### bankruptcy_Test_X.csv — Test Data

**8,000 firms × 64 columns** — no target variable. These are the firms the model scores.

Same 64 financial ratio columns as the training set. The model generates a bankruptcy probability score for each firm, which becomes the submission file.

---

## Preprocessing Applied

Both files require the following before modelling:

```python
# 1. Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2. Apply arcsinh transformation to all 64 ratio columns
attr_cols = [f'Attr{i}' for i in range(1, 65)]
df[attr_cols] = np.arcsinh(df[attr_cols])
```

**Why arcsinh instead of log?** Financial ratios have extreme outliers, negative values, and zeros. Log transform fails on zeros and does not handle negative values. arcsinh works on the full real line — compressing extremes while preserving sign and direction.

---

## Missing Values

All 64 financial features have missing values. AutoGluon handles these internally during training — no imputation is required before passing data to the model.

---

## Source

Dataset provided by Prof. for the MGMT 571 Data Mining course, Purdue University, Fall 2025. Originally sourced from financial statement data for publicly traded firms. Feature names are anonymised as Attr1–Attr64.

Competition link: [https://www.kaggle.com/competitions/fall-2025-mgmt-571-final-project](https://www.kaggle.com/competitions/fall-2025-mgmt-571-final-project)
