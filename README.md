# Customer Analytics Pipeline — QLab Tea

A two-stage data science pipeline built for **QLab Tea**, a multi-market tea e-commerce brand operating across Singapore, Malaysia, and Hong Kong. The project covers customer segmentation (Case 1) and repurchase prediction with business value quantification (Case 2).

View the final presentation ppt [here](./pre1+2%20final.pptx)

---

## Overview


|              | Case 1 — Customer Segmentation                                 | Case 2 — Repurchase Prediction                             |
| ------------ | -------------------------------------------------------------- | ---------------------------------------------------------- |
| **Goal**     | Identify distinct customer groups to enable targeted marketing | Predict whether a customer will repurchase within 12 weeks |
| **Approach** | RFM feature engineering + K-Means clustering                   | Binary classification with SHAP explainability             |
| **Output**   | Segment profiles with behavioral and preference breakdowns     | Churn propensity scores + revenue impact scenarios         |
| **Markets**  | SGD, MYR, HKD                                                  | SGD                                                        |


---

## Pipeline

```
original_data/
    users.csv, orders.csv, subscriptions.csv,
    preferences.csv, events.csv, products.csv,
    vouchers.csv, voucher_applications.csv, ...
         │
         ▼
┌─────────────────────────────┐
│  Case 1 · main.ipynb        │  ──── main_hk.ipynb (HK variant)
│  Customer Segmentation      │
│  · Data cleaning            │
│  · Feature engineering (7   │
│    feature families)        │
│  · Temporal RFM scoring     │
│  · K-Means clustering       │
│  · t-SNE visualisation      │
│  → Labelled customer table  │
└────────────┬────────────────┘
             │ labelled customers (required dependency)
             ▼
┌─────────────────────────────┐
│  Case 2 · case2/            │
│  repurchase_pipeline.ipynb  │
│  · Order-level features     │
│  · Model training & eval    │
│  · SHAP analysis            │
│  · Business value scenarios │
│  → Propensity scores +      │
│    revenue impact reports   │
└─────────────────────────────┘
```

---

## Case 1 — Customer Segmentation

### Feature Engineering

Seven feature families are constructed per customer:


| Family               | Features                                               |
| -------------------- | ------------------------------------------------------ |
| **RFM**              | Recency, frequency, monetary value (temporal-weighted) |
| **Behavioural**      | Active days, session count, cart-to-order rate         |
| **Product category** | Spend share across teas, accessories, gift sets        |
| **Subscription**     | Subscription status, tenure, plan tier                 |
| **Engagement**       | Email open rate, app events, referral activity         |
| **Promotion**        | Voucher usage rate, discount sensitivity               |
| **Social**           | Referral count, user-generated content                 |


### Temporal RFM Weighting

Monetary values are weighted by recency year to reduce the influence of early-stage purchasing behaviour:


| Year | Weight |
| ---- | ------ |
| 2020 | 0.2    |
| 2021 | 0.4    |
| 2022 | 0.7    |
| 2023 | 1.0    |


### Segmentation Models

Two complementary models are produced:

- **Binary segmentation** — high-value vs. standard customers; simpler to operationalise
- **Ternary segmentation** — three-tier structure for more granular campaign targeting

Dimensionality reduction via **t-SNE** is used for visual validation of cluster separation.

### Hong Kong Variant (`main_hk.ipynb`)

A simplified analysis targeting the HK market in isolation (40 customers, 81 completed orders, Mar 2020 – Sep 2022). Given the small sample size, results should be interpreted directionally.

---

## Case 2 — Repurchase Prediction

### Problem Definition

> Given that a customer just made a purchase, will they purchase again within **12 weeks**?

This is framed as a **binary classification** problem at the order level, then aggregated to the customer level for actionable segment strategies.

### Models Compared


| Model               | Notes                                         |
| ------------------- | --------------------------------------------- |
| Logistic Regression | Baseline; interpretable coefficients          |
| Random Forest       | Ensemble; handles non-linearity               |
| XGBoost             | Gradient boosting; strong tabular performance |
| LightGBM            | Gradient boosting; fast                       |


Evaluation metrics: Accuracy, Precision, Recall, F1, AUC-ROC.

### Explainability (SHAP)

SHAP values are computed at both the order level and aggregated to the user level. Segment-specific SHAP summaries are produced for three business personas:

- **Golden Whales** — high-value, high-frequency
- **High Potential** — moderate engagement, growth opportunity
- **Casual Walk-in** — low retention, price-sensitive

### Business Value Scenarios

Three ROI scenarios are modelled to quantify the commercial impact of model deployment:


| Scenario | Lever                                            | Metric                        |
| -------- | ------------------------------------------------ | ----------------------------- |
| 1        | AOV uplift via upselling at-risk customers       | Revenue per retained customer |
| 2        | Churn reduction via targeted retention campaigns | Incremental GMV               |
| 3        | Subscription conversion from repurchase base     | Recurring revenue growth      |


All scenarios include sensitivity analysis and estimated ROI.

---

## Input / Output

### Inputs

All source data lives in `original_data/`:


| File                                        | Description                                   |
| ------------------------------------------- | --------------------------------------------- |
| `users.csv`                                 | Customer master (47,231 records)              |
| `orders.csv`                                | Transaction history (264,341 records)         |
| `subscriptions.csv`                         | Subscription plans and status                 |
| `preferences.csv`                           | Customer taste profiles                       |
| `events.csv`                                | Behavioural events (page views, cart actions) |
| `products.csv`                              | Product catalogue                             |
| `vouchers.csv` + `voucher_applications.csv` | Promotion tracking                            |
| `user_references.csv`                       | Referral relationships                        |
| `conditional_offers.csv`                    | Promotion rules                               |
| `postal_codes.csv`                          | Geographic mapping                            |
| `review.xlsx`                               | Customer feedback                             |
| `20250826_data_dictionary.xlsx`             | Full schema reference                         |


### Outputs


| Location                                     | Content                                                      |
| -------------------------------------------- | ------------------------------------------------------------ |
| In-notebook                                  | Case 1 cluster visualisations (K-Means, t-SNE, GMV profiles) |
| `case2/outputs/repurchase/visualizations/`   | Model performance charts, SHAP plots (18 PNG)                |
| `case2/outputs/repurchase/revenue_impact/`   | Revenue impact analysis (2 PNG)                              |
| `case2/outputs/repurchase/value_estimation/` | Business scenario ROI charts (8 PNG)                         |


---

## Setup

**Requirements:** Python 3.8+

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ensure the `original_data/` folder is present with all source files before running.

---

## Execution

**Case 1 must be run before Case 2** — Case 2 depends on the labelled customer table produced by Case 1.

```
# Step 1 — Customer Segmentation
main.ipynb

# Step 1b — Hong Kong variant (optional)
main_hk.ipynb

# Step 2 — Repurchase Prediction
case2/repurchase_pipeline.ipynb
```

> **Note:** The final section of `main.ipynb` uses the OpenAI API for automated secondary-segment analysis. This is supplementary and can be skipped — it is expected to error if no API key is configured.

---

## Dependencies


| Package                 | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| `pandas`, `numpy`       | Data manipulation                          |
| `scikit-learn`          | Preprocessing, clustering, baseline models |
| `xgboost`, `lightgbm`   | Gradient boosting classifiers              |
| `shap`                  | Model explainability                       |
| `matplotlib`, `seaborn` | Visualisation                              |
| `joblib`                | Model persistence                          |
| `tqdm`                  | Progress tracking                          |


Full pinned versions: `requirements.txt`

---

## Repository Structure

```
.
├── main.ipynb                          # Case 1: customer segmentation (all markets)
├── main_hk.ipynb                       # Case 1: Hong Kong variant
├── case2/
│   └── repurchase_pipeline.ipynb       # Case 2: repurchase prediction pipeline
├── pre1+2 final.pptx                   # Presentation slides (Case 1 & 2)
└── requirements.txt                    # Python dependencies
```

