# Customer Loyalty Propensity Prediction Pipeline

## 📋 Overview

This project implements a comprehensive machine learning pipeline for customer loyalty prediction. The system analyzes customer behavior data to predict whether customers will continue making purchases, enabling targeted retention strategies and personalized marketing campaigns.

## 🎯 Business Problem

In retail and e-commerce, understanding customer loyalty is crucial for:
- **Customer Retention**: Identifying customers at risk of churning
- **Personalized Marketing**: Targeting high-value customers with appropriate strategies
- **Resource Allocation**: Focusing retention efforts where they matter most
- **Revenue Optimization**: Maximizing customer lifetime value

## 🏗️ Project Architecture

The pipeline consists of 6 main steps orchestrated through `main.py`:

### Step 1: Feature Engineering (`feature_engineering.py`)
- **Input**: Raw customer transaction and behavior data
- **Process**: Creates comprehensive feature sets including:
  - Recency, Frequency, Monetary (RFM) metrics
  - Temporal patterns and seasonality
  - Customer segmentation features
  - Behavioral indicators
- **Output**: Processed feature matrices ready for modeling

### Step 2: Model Training (`model_training.py`)
- **Algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Networks
- **Techniques**: Cross-validation, hyperparameter tuning, class balancing
- **Output**: Trained models with performance metrics

### Step 3: Model Evaluation
- **Metrics**: ROC-AUC, Precision-Recall AUC, Precision, Recall, F1-Score
- **Analysis**: Confusion matrices, classification reports
- **Selection**: Automated best model selection based on performance

### Step 4: SHAP Analysis (`shap_analysis.py`)
- **Explainability**: SHAP (SHapley Additive exPlanations) values
- **Insights**: Feature importance, customer segment analysis
- **Output**: Interpretable model explanations

### Step 5: Visualizations (`visualizations.py`)
- **Performance Plots**: ROC curves, Precision-Recall curves, confusion matrices
- **Feature Analysis**: SHAP summary plots, feature importance charts
- **Business Insights**: Segment-specific propensity distributions

### Step 6: Business Insights Generation
- **Strategic Recommendations**: Actionable insights for different customer segments
- **Propensity Scoring**: Customer segmentation by loyalty likelihood
- **Marketing Guidance**: Targeted campaign recommendations

## 📁 Directory Structure

```
case2/
├── main.py                      # Main pipeline orchestrator
├── feature_engineering.py       # Feature engineering pipeline
├── model_training.py           # Model training and evaluation
├── shap_analysis.py            # SHAP explainability analysis
├── visualizations.py           # Plotting and visualization functions
├── order_based_features.py     # Order-specific feature engineering
├── order_based_model.py        # Order-based modeling approaches
├── outputs/                    # Generated outputs
│   ├── features/              # Processed feature files
│   ├── models/                # Trained models and metadata
│   ├── plots/                 # Visualization plots
│   └── predictions/           # Propensity scores and predictions
└── __pycache__/               # Python cache files
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, xgboost, lightgbm, shap, matplotlib, seaborn

### Installation
```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn

# Additional dependencies for neural networks
pip install tensorflow keras
```

### Running the Pipeline
```bash
# Execute the complete pipeline
python main.py
```

The pipeline will automatically:
1. Create necessary output directories
2. Process raw data through feature engineering
3. Train multiple models with cross-validation
4. Evaluate and select the best performing model
5. Generate SHAP explanations
6. Create comprehensive visualizations
7. Produce business insights and recommendations

## 📊 Key Features

### Advanced Feature Engineering
- **Temporal Features**: Recency, frequency, monetary analysis
- **Behavioral Patterns**: Purchase sequences and seasonality
- **Customer Segmentation**: Automated clustering and profiling
- **Domain-Specific Features**: Retail and e-commerce focused metrics

### Multi-Model Approach
- **Ensemble Methods**: Random Forest, XGBoost, LightGBM
- **Traditional ML**: Logistic Regression with regularization
- **Deep Learning**: Neural network implementations
- **Automated Selection**: Data-driven best model selection

### Explainable AI
- **SHAP Integration**: Complete model interpretability
- **Feature Importance**: Global and local explanations
- **Segment Analysis**: Customer group-specific insights

### Business Intelligence
- **Propensity Scores**: Customer loyalty probability predictions
- **Strategic Segments**: "Golden Whales", "High Potential", "Drifting Risk"
- **Actionable Recommendations**: Targeted marketing strategies

## 📈 Output Artifacts

### Model Outputs (`outputs/models/`)
- `*_best_model.pkl`: Serialized best performing model
- `*_model_comparison.csv`: Performance comparison across all models
- `*_feature_importance.csv`: Feature importance rankings
- `*_business_insights.txt`: Strategic recommendations

### Visualizations (`outputs/plots/`)
- ROC and Precision-Recall curves
- Confusion matrices and metric comparisons
- SHAP summary and bar plots
- Segment-specific propensity distributions

### Predictions (`outputs/predictions/`)
- Customer propensity scores
- Segment classifications
- Win-back campaign targeting recommendations

## 🔧 Configuration

Key parameters can be adjusted in `main.py`:

```python
RANDOM_STATE = 42          # Reproducibility seed
TEST_SIZE = 0.2            # Train/test split ratio
CV_FOLDS = 5               # Cross-validation folds
TARGET_NAME = 'will_purchase'  # Prediction target
```

## 📋 Dependencies

Core dependencies (install via pip):
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `scikit-learn>=1.0.0` - Machine learning algorithms
- `xgboost>=1.5.0` - Gradient boosting
- `lightgbm>=3.3.0` - Microsoft LightGBM
- `shap>=0.40.0` - Model explainability
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualization

## 🤝 Contributing

### Development Workflow
1. **Feature Engineering**: Add new features in `feature_engineering.py`
2. **Model Development**: Implement new algorithms in `model_training.py`
3. **Analysis**: Extend SHAP analysis in `shap_analysis.py`
4. **Visualization**: Add new plots in `visualizations.py`
5. **Testing**: Validate changes with cross-validation metrics

### Code Standards
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include logging for pipeline steps
- Test new features on validation sets

## 📊 Performance Metrics

The pipeline optimizes for:
- **ROC-AUC**: Overall classification performance
- **PR-AUC**: Performance on imbalanced datasets
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to identify all positive cases

## 🔍 Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce dataset size or use sampling
2. **Encoding Issues**: Ensure UTF-8 encoding for text features
3. **Missing Dependencies**: Install all required packages
4. **Directory Permissions**: Ensure write access to outputs folder

### Logs and Debugging
- Check `outputs/pipeline.log` for detailed execution logs
- Review model comparison files for performance issues
- Examine SHAP plots for feature engineering problems

## 📝 License

This project is developed for academic and research purposes in the BS6206 course.

## 👥 Team

This project was developed as part of a collaborative machine learning initiative, combining expertise in:
- Data Engineering
- Machine Learning
- Business Intelligence
- Customer Analytics

---

For questions or contributions, please refer to the detailed logging output and documentation within each module.
