# Cross-Sector Stock Market Analysis

## Project Overview
This project analyzes cross-sector relationships in the stock market using data mining techniques including clustering, outlier detection, feature selection, and classification.

## Setup Instructions

### Installation

1. Clone or navigate to the project directory

2. (Optional) Create a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## How to Run

### Running the Jupyter Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to `notebooks/Final_Analysis.ipynb` and run the cells sequentially.

## Dataset

- **File**: `preprocessed_data_pca.csv`
- **Rows**: ~9000 stock market trading days
- **Features**: 44 principal components (PC1-PC44) derived from sector prices and market indicators
- **Targets**: 11 binary targets (one per GICS sector) indicating next-day price movement (1 = up, 0 = down)
- **Sectors**: Technology, Healthcare, Financials, Consumer Discretionary, Communication Services, Industrials, Consumer Staples, Energy, Utilities, Real Estate, Materials

## Methodology

1. **Clustering**: Identify market regimes using K-Means and Hierarchical clustering
2. **Outlier Detection**: Find anomalous market days using Isolation Forest and LOF
3. **Feature Selection**: Reduce dimensionality using Mutual Information and Lasso
4. **Classification**: Predict market regimes using Random Forest and SVM/k-NN
5. **Cross-Sector Analysis**: Analyze sector performance patterns across different market regimes

## Team
CMPT 459 Final Project - Fall 2025

## License
Academic use only - SFU CMPT 459 course project
