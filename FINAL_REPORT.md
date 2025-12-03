# Cross-Sector Stock Market Analysis: Final Report
**CMPT 459 - Data Mining Course Project**  
Team: Sleepycats | Members: Yogya Agrawal, Ananya Singh

---

## 1. Methodology

This project analyzes cross-sector stock market relationships using a cluster-based classification approach on 9,015 trading days with 44 principal components and 11 sector binary targets.

**Phase 1: Clustering** - K-Means with k=3 identified distinct market regimes (Normal, Volatility, Momentum) validated by elbow method and silhouette scores.

**Phase 2: Outlier Detection** - Isolation Forest and LOF identified ~5% anomalous days representing crisis periods, which were retained for analysis.

**Phase 3: Feature Selection** - Mutual Information reduced features from 44 to 20, achieving 2.8x speedup while maintaining 98% accuracy.

**Phase 4: Classification** - Random Forest achieved **83.1% accuracy** (tuned) vs SVM's 79.4%, significantly outperforming random baseline (33.3%).

---

## 2. Key Results

### Market Regimes Identified

- **Cluster 0 (Normal)**: 7,523 days (83.4%) - Balanced performance, all sectors ~52-54% win rate
- **Cluster 1 (Volatility)**: 1,282 days (14.2%) - Tech/Industrials lead (54%), Energy/Real Estate lag (50-51%)
- **Cluster 2 (Momentum)**: 210 days (2.3%) - Tech dominates (60.5%), growth sectors strong (Consumer Disc 59%, Energy 58.6%)

### Cross-Sector Insights

**Growth vs Defensive Divergence**: In momentum markets (Cluster 2), growth sectors (Tech, Consumer Discretionary) outperform defensive sectors (Utilities, Financials) by 7-8 percentage points.

**Energy Volatility**: Weakest in normal (52.1%) and volatile (50.3%) conditions but rebounds strongly in momentum markets (58.6%), demonstrating cyclical regime-dependency.

**Technology Leadership**: Consistently top performer across all regimes, with exceptional 60.5% win rate in momentum conditions.

**Sector Correlations**: Tech and Consumer Discretionary move together (correlation ~0.65), while defensive sectors show lower correlation with growth sectors.

### Model Performance

Hyperparameter tuning via GridSearchCV improved accuracy by ~0.5% for both models:
- **Random Forest**: 82.7% → 83.1% (n_estimators=200, max_depth=20)
- **SVM**: 78.9% → 79.4% (C=10, gamma=0.01)
- ROC-AUC scores consistently above 0.90 for all classes

---

## 3. Domain Applications

**Portfolio Strategy**: Dynamic allocation based on predicted regime - balanced approach in normal markets, favor growth sectors in momentum periods, reduce Energy exposure during volatility.

**Risk Management**: Outlier analysis identified 450+ crisis days correlating with known events (2008 crash, COVID-19), validating regime-aware risk models.

---

## 4. Challenges & Solutions

1. **High Dimensionality** - Mutual Information feature selection reduced 44→20 features, maintaining accuracy
2. **Class Imbalance** - Cluster 2 only 2.3% of data; mitigated via stratified splits and cross-validation
3. **Interpretability** - PCA features lack business meaning; addressed via feature importance analysis and regime-specific sector performance

---

## 5. Conclusions

Successfully identified **3 distinct market regimes** with **83.1% predictive accuracy**, revealing actionable cross-sector relationships. Growth and defensive sectors diverge significantly in momentum markets, while Technology consistently leads across all conditions.

**Key Achievement**: Built a reproducible pipeline demonstrating that machine learning can predict market regimes and inform portfolio decisions.

**Limitations**: PCA obscures direct feature interpretation.

**Future Work**: Incorporate temporal analysis (LSTM/GRU), macroeconomic indicators, and ensemble methods for regime transition prediction.
