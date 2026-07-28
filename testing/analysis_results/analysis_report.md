# Face Recognition Embedding Analysis Report

## Executive Summary
- **Total Samples Analyzed**: 1275
- **Total Identities (Students)**: 43
- **Total Data Points**: 425
- **Optimal Threshold (EER Intersect)**: `0.2660` (Equal Error Rate: `0.81%`)
- **Max F1-Score Threshold**: `0.4070` (F1 Score: `0.9959`)

---

## 1. Similarity Distribution Statistics

| Match Type | Count | Mean | Median | Std Dev | Min | Max |
|---|---|---|---|---|---|---|
| **Genuine** | 47,100 | 0.7709 | 0.7796 | 0.1158 | -0.0854 | 0.9915 |
| **Impostor** | 1,577,250 | 0.0618 | 0.0580 | 0.0820 | -0.2438 | 0.4184 |

---

## 2. Top-K Retrieval Performance

- **Top-1 Accuracy**: `99.92%`
- **Top-3 Accuracy**: `99.92%`
- **Top-5 Accuracy**: `99.92%`
- **Top-10 Accuracy**: `99.92%`

---

## 3. Data Point Consistency Analysis

- **Within-Data-Point Mean Similarity**: `0.8431` (Std: `0.0941`)
- **Across-Data-Point Mean Similarity**: `0.7780` (Std: `0.0785`)

---

## 4. Hardest and Easiest Identities

### Easiest Identities (Highest Margin)
| kerberos_id   |   avg_genuine_margin |   avg_genuine_sim |   highest_impostor_sim |
|:--------------|---------------------:|------------------:|-----------------------:|
| csz258227     |             0.709409 |          0.94755  |               0.276551 |
| mcs252104     |             0.6932   |          0.853691 |               0.252552 |
| csz258238     |             0.686939 |          0.770826 |               0.283089 |
| mcs252089     |             0.680504 |          0.84201  |               0.373825 |
| mcs252108     |             0.665682 |          0.812112 |               0.294635 |

### Hardest Identities (Lowest Margin)
| kerberos_id   |   avg_genuine_margin |   avg_genuine_sim |   highest_impostor_sim |
|:--------------|---------------------:|------------------:|-----------------------:|
| mcs252100     |             0.558197 |          0.71782  |               0.34708  |
| mcs242004     |             0.55536  |          0.76012  |               0.418434 |
| mcs252094     |             0.538559 |          0.715704 |               0.38016  |
| mcs252102     |             0.535029 |          0.732156 |               0.378173 |
| mcs252112     |             0.525183 |          0.749561 |               0.345271 |

---

## 5. Failure Diagnostics & Negative Margin Samples

- **Total Samples with Negative Margin (Impostor closer than Genuine)**: `1`

### Top Hardest Impostor Matches
| query_kerberos   |   query_dp |   query_sample | hard_negative_kerberos   |   impostor_similarity |
|:-----------------|-----------:|---------------:|:-------------------------|----------------------:|
| mcs242004        |         16 |              1 | csy257616                |              0.418434 |
| csy257616        |          1 |              1 | mcs242004                |              0.418434 |
| mcs252091        |          2 |              1 | mcs252744                |              0.406925 |
| mcs252744        |         14 |              1 | mcs252091                |              0.406925 |
| csy257616        |          3 |              1 | csz238003                |              0.404903 |
| csz238003        |         20 |              2 | csy257616                |              0.404903 |
| csy257616        |          1 |              3 | mcs242004                |              0.403774 |
| mcs242004        |          1 |              3 | csy257616                |              0.403448 |

---

## 6. Generated Visualizations
- `similarity_distributions.png`: Density plots of genuine vs. impostor scores.
- `threshold_sweeps.png`: FAR, FRR, and F1 score curves across thresholds.
- `roc_curve.png`: Receiver Operating Characteristic curve.
- `confusion_matrix.png`: Top hard-negative identity confusion matrix.
