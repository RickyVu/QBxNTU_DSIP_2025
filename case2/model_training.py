# ---
# Model Training Pipeline for Case 2: Propensity Prediction
# ---

import pandas as pd
import numpy as np
import os
import json
import logging
import joblib
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# XGBoost and LightGBM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
FEATURES_PATH = "outputs/features"
MODELS_PATH = "outputs/models"
RANDOM_STATE = 42

# ============================================================================
# DATA LOADING
# ============================================================================

def load_processed_features(target_name='will_purchase', features_path=FEATURES_PATH):
    """
    Load processed features from CSV file.
    
    Args:
        target_name: Name of target column ('will_purchase' or 'will_be_high_value')
        features_path: Path to features directory
        
    Returns:
        tuple: (X, y, feature_names, user_ids)
    """
    logger.info("="*80)
    logger.info("LOADING PROCESSED FEATURES")
    logger.info("="*80)
    
    file_path = f"{features_path}/{target_name}_features.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Features file not found: {file_path}. Run feature_engineering.py first.")
    
    df = pd.read_csv(file_path)
    
    logger.info(f"Loaded features from: {file_path}")
    logger.info(f"  Shape: {df.shape}")
    
    # Separate user_id, features, and target
    user_ids = df['user_id']
    y = df[target_name]
    X = df.drop(columns=['user_id', target_name])
    
    feature_names = X.columns.tolist()
    
    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  Samples: {len(X)}")
    logger.info(f"  Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"  Target rate: {y.mean():.2%}")
    
    return X, y, feature_names, user_ids


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """
    Split data into training and test sets with stratification.
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info("\nSplitting data into train/test sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"  Training set: {len(X_train):,} samples")
    logger.info(f"  Test set: {len(X_test):,} samples")
    logger.info(f"  Train target rate: {y_train.mean():.2%}")
    logger.info(f"  Test target rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, feature_names):
    """
    Scale features using StandardScaler (for Logistic Regression).
    
    Args:
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    logger.info("\nScaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for consistency
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
    
    logger.info(f"  Scaled features: mean ≈ {X_train_scaled.mean().mean():.4f}, std ≈ {X_train_scaled.std().mean():.4f}")
    
    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_model_configs(n_pos, n_neg):
    """
    Get model configurations for all 4 models.
    
    Args:
        n_pos: Number of positive samples
        n_neg: Number of negative samples
        
    Returns:
        dict: Model configurations with hyperparameter grids
    """
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
    
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                solver='lbfgs'
            ),
            'param_grid': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l2']
            },
            'requires_scaling': True
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            ),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5]
            },
            'requires_scaling': False
        },
        'XGBoost': {
            'model': XGBClassifier(
                random_state=RANDOM_STATE,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1
            ),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1]
            },
            'requires_scaling': False
        },
        'LightGBM': {
            'model': LGBMClassifier(
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1,
                verbose=-1
            ),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1]
            },
            'requires_scaling': False
        }
    }
    
    return models


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_single_model(model_name, model_config, X_train, y_train, cv=5):
    """
    Train a single model with hyperparameter tuning.
    
    Args:
        model_name: Name of the model
        model_config: Model configuration dict
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        
    Returns:
        tuple: (best_model, best_params, cv_score)
    """
    logger.info(f"\n  Training {model_name}...")
    
    model = model_config['model']
    param_grid = model_config['param_grid']
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = grid_search.best_score_
    
    logger.info(f"    Best params: {best_params}")
    logger.info(f"    CV ROC-AUC: {cv_score:.4f}")
    
    return best_model, best_params, cv_score


def train_all_models(X_train, X_train_scaled, y_train, n_pos, n_neg, cv=5):
    """
    Train all 4 models with hyperparameter tuning.
    
    Args:
        X_train: Unscaled training features (for tree models)
        X_train_scaled: Scaled training features (for Logistic Regression)
        y_train: Training target
        n_pos: Number of positive samples
        n_neg: Number of negative samples
        cv: Number of cross-validation folds
        
    Returns:
        dict: Trained models with their configurations
    """
    logger.info("="*80)
    logger.info("MODEL TRAINING")
    logger.info("="*80)
    
    model_configs = get_model_configs(n_pos, n_neg)
    trained_models = {}
    
    for model_name, config in model_configs.items():
        # Use scaled data for Logistic Regression, unscaled for tree models
        X_data = X_train_scaled if config['requires_scaling'] else X_train
        
        best_model, best_params, cv_score = train_single_model(
            model_name, config, X_data, y_train, cv
        )
        
        trained_models[model_name] = {
            'model': best_model,
            'best_params': best_params,
            'cv_score': cv_score,
            'requires_scaling': config['requires_scaling']
        }
    
    logger.info("\n✓ All models trained successfully")
    
    return trained_models


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_single_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        
    Returns:
        dict: Evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm
    
    # ROC curve data
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds_roc}
    
    # Precision-Recall curve data
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    metrics['pr_curve'] = {'precision': precision_curve, 'recall': recall_curve, 'thresholds': thresholds_pr}
    
    return metrics


def evaluate_all_models(trained_models, X_test, X_test_scaled, y_test):
    """
    Evaluate all trained models on test set.
    
    Args:
        trained_models: Dictionary of trained models
        X_test: Unscaled test features
        X_test_scaled: Scaled test features
        y_test: Test target
        
    Returns:
        dict: Evaluation results for all models
    """
    logger.info("="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)
    
    results = {}
    
    for model_name, model_info in trained_models.items():
        logger.info(f"\n  Evaluating {model_name}...")
        
        # Use scaled data for Logistic Regression
        X_data = X_test_scaled if model_info['requires_scaling'] else X_test
        
        metrics = evaluate_single_model(model_info['model'], X_data, y_test, model_name)
        
        results[model_name] = {
            **metrics,
            'cv_score': model_info['cv_score'],
            'best_params': model_info['best_params']
        }
        
        logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall:    {metrics['recall']:.4f}")
        logger.info(f"    F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"    PR-AUC:    {metrics['pr_auc']:.4f}")
    
    return results


def create_comparison_table(results):
    """
    Create a comparison table of all models.
    
    Args:
        results: Dictionary of evaluation results
        
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'CV ROC-AUC': metrics['cv_score'],
            'Test Accuracy': metrics['accuracy'],
            'Test Precision': metrics['precision'],
            'Test Recall': metrics['recall'],
            'Test F1': metrics['f1_score'],
            'Test ROC-AUC': metrics['roc_auc'],
            'Test PR-AUC': metrics['pr_auc']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test ROC-AUC', ascending=False)
    
    return comparison_df


# ============================================================================
# MODEL SELECTION
# ============================================================================

def select_best_model(results, trained_models, metric='roc_auc'):
    """
    Select the best model based on specified metric.
    
    Args:
        results: Dictionary of evaluation results
        trained_models: Dictionary of trained models
        metric: Metric to use for selection (default: 'roc_auc')
        
    Returns:
        tuple: (best_model_name, best_model, best_metrics)
    """
    logger.info("="*80)
    logger.info("MODEL SELECTION")
    logger.info("="*80)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x][metric])
    best_metrics = results[best_model_name]
    best_model = trained_models[best_model_name]['model']
    requires_scaling = trained_models[best_model_name]['requires_scaling']
    
    logger.info(f"\n🏆 Best Model: {best_model_name}")
    logger.info(f"   Selection metric: {metric} = {best_metrics[metric]:.4f}")
    logger.info(f"   Requires scaling: {requires_scaling}")
    logger.info(f"\n   All metrics:")
    logger.info(f"     Accuracy:  {best_metrics['accuracy']:.4f}")
    logger.info(f"     Precision: {best_metrics['precision']:.4f}")
    logger.info(f"     Recall:    {best_metrics['recall']:.4f}")
    logger.info(f"     F1 Score:  {best_metrics['f1_score']:.4f}")
    logger.info(f"     ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    logger.info(f"     PR-AUC:    {best_metrics['pr_auc']:.4f}")
    
    return best_model_name, best_model, best_metrics, requires_scaling


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model_artifacts(best_model, best_model_name, best_metrics, scaler, 
                         feature_names, requires_scaling, target_name='will_purchase',
                         models_path=MODELS_PATH):
    """
    Save model artifacts to disk.
    
    Args:
        best_model: Best trained model
        best_model_name: Name of the best model
        best_metrics: Evaluation metrics
        scaler: StandardScaler (if used)
        feature_names: List of feature names
        requires_scaling: Whether model requires scaling
        target_name: Name of target variable
        models_path: Path to save models
    """
    logger.info("="*80)
    logger.info("SAVING MODEL ARTIFACTS")
    logger.info("="*80)
    
    # Create output directory
    os.makedirs(models_path, exist_ok=True)
    
    # Save model
    model_path = f"{models_path}/{target_name}_best_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"  ✓ Saved model to: {model_path}")
    
    # Save scaler (if used)
    if requires_scaling and scaler is not None:
        scaler_path = f"{models_path}/{target_name}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"  ✓ Saved scaler to: {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'target_name': target_name,
        'requires_scaling': requires_scaling,
        'feature_names': feature_names,
        'metrics': {
            'accuracy': float(best_metrics['accuracy']),
            'precision': float(best_metrics['precision']),
            'recall': float(best_metrics['recall']),
            'f1_score': float(best_metrics['f1_score']),
            'roc_auc': float(best_metrics['roc_auc']),
            'pr_auc': float(best_metrics['pr_auc'])
        },
        'best_params': best_metrics['best_params'],
        'cv_score': float(best_metrics['cv_score']),
        'created_at': datetime.now().isoformat()
    }
    
    metadata_path = f"{models_path}/{target_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  ✓ Saved metadata to: {metadata_path}")
    
    logger.info("\n✓ All artifacts saved successfully")
    
    return model_path, metadata_path


def save_comparison_results(comparison_df, results, target_name='will_purchase', models_path=MODELS_PATH):
    """
    Save model comparison results.
    
    Args:
        comparison_df: Comparison DataFrame
        results: Full results dictionary
        target_name: Name of target variable
        models_path: Path to save results
    """
    # Save comparison table
    comparison_path = f"{models_path}/{target_name}_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"  ✓ Saved comparison table to: {comparison_path}")
    
    # Save full results (excluding non-serializable items)
    results_serializable = {}
    for model_name, metrics in results.items():
        results_serializable[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc']),
            'pr_auc': float(metrics['pr_auc']),
            'cv_score': float(metrics['cv_score']),
            'best_params': metrics['best_params']
        }
    
    results_path = f"{models_path}/{target_name}_all_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    logger.info(f"  ✓ Saved full results to: {results_path}")


# ============================================================================
# COMPLETE TRAINING PIPELINE
# ============================================================================

def run_model_training_pipeline(target_name='will_purchase', test_size=0.2, cv=5):
    """
    Run the complete model training pipeline.
    
    Args:
        target_name: Target variable name ('will_purchase' or 'will_be_high_value')
        test_size: Proportion of data for testing
        cv: Number of cross-validation folds
        
    Returns:
        dict: Pipeline results including best model and metrics
    """
    logger.info("="*80)
    logger.info(f"CASE 2: MODEL TRAINING PIPELINE - {target_name.upper()}")
    logger.info("="*80)
    
    # Step 1: Load processed features
    X, y, feature_names, user_ids = load_processed_features(target_name)
    
    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y, test_size)
    
    # Step 3: Scale features (for Logistic Regression)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, feature_names)
    
    # Step 4: Get class counts for imbalance handling
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    logger.info(f"\nClass distribution in training set:")
    logger.info(f"  Positive (target=1): {n_pos:,} ({n_pos/len(y_train):.2%})")
    logger.info(f"  Negative (target=0): {n_neg:,} ({n_neg/len(y_train):.2%})")
    
    # Step 5: Train all models
    trained_models = train_all_models(X_train, X_train_scaled, y_train, n_pos, n_neg, cv)
    
    # Step 6: Evaluate all models
    results = evaluate_all_models(trained_models, X_test, X_test_scaled, y_test)
    
    # Step 7: Create comparison table
    comparison_df = create_comparison_table(results)
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)
    print(comparison_df.to_string(index=False))
    
    # Step 8: Select best model
    # Use PR-AUC if class imbalance is severe (positive rate < 30%)
    selection_metric = 'pr_auc' if y.mean() < 0.3 else 'roc_auc'
    best_model_name, best_model, best_metrics, requires_scaling = select_best_model(
        results, trained_models, metric=selection_metric
    )
    
    # Step 9: Save model artifacts
    save_model_artifacts(
        best_model, best_model_name, best_metrics, scaler,
        feature_names, requires_scaling, target_name
    )
    
    # Step 10: Save comparison results
    save_comparison_results(comparison_df, results, target_name)
    
    logger.info("\n" + "="*80)
    logger.info("✓ MODEL TRAINING PIPELINE COMPLETE")
    logger.info("="*80)
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_metrics': best_metrics,
        'requires_scaling': requires_scaling,
        'scaler': scaler,
        'feature_names': feature_names,
        'comparison_df': comparison_df,
        'all_results': results,
        'trained_models': trained_models,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'user_ids_test': user_ids.iloc[X_test.index]
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Run the training pipeline for loyalty model
    results = run_model_training_pipeline(target_name='will_purchase')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Model: {results['best_model_name']}")
    print(f"ROC-AUC: {results['best_metrics']['roc_auc']:.4f}")
    print(f"PR-AUC: {results['best_metrics']['pr_auc']:.4f}")
