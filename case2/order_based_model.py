# ---
# Order-Based Model Training
# Trains models on order-based features (post-purchase repurchase prediction)
# ---

import pandas as pd
import numpy as np
import os
import json
import logging
import joblib
from datetime import datetime

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
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
TRAIN_PATH = "outputs/order_based/order_based_train.csv"
TEST_PATH = "outputs/order_based/order_based_test.csv"
OUTPUT_DIR = "outputs/order_based/models"
RANDOM_STATE = 42

# ============================================================================
# DATA LOADING
# ============================================================================

def load_train_test_data():
    """Load training and test data."""
    logger.info("="*80)
    logger.info("LOADING TRAIN/TEST DATA")
    logger.info("="*80)
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    logger.info(f"Training set: {len(train_df):,} orders")
    logger.info(f"  Positive rate: {train_df['will_repurchase'].mean():.1%}")
    
    logger.info(f"Test set: {len(test_df):,} orders")
    logger.info(f"  Positive rate: {test_df['will_repurchase'].mean():.1%}")
    
    return train_df, test_df


def prepare_features(df, exclude_cols=['user_id', 'order_id', 'order_date', 'week_start', 
                                        'current_shipping_method', 'will_repurchase']):
    """
    Prepare features for modeling.
    
    Args:
        df: DataFrame with features
        exclude_cols: Columns to exclude from features
        
    Returns:
        tuple: (X, y, feature_names)
    """
    # Separate target
    y = df['will_repurchase'].values
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    
    # Fill any remaining NaN
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    feature_names = X.columns.tolist()
    
    return X.values, y, feature_names


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_model_configs(n_pos, n_neg):
    """
    Get model configurations.
    
    With good class balance (66:34), we can use standard settings.
    
    Args:
        n_pos: Number of positive samples
        n_neg: Number of negative samples
        
    Returns:
        dict: Model configurations
    """
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
    
    logger.info(f"Class balance: {n_pos:,} positive, {n_neg:,} negative")
    logger.info(f"Positive rate: {n_pos/(n_pos+n_neg):.1%}")
    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                solver='lbfgs',
                n_jobs=1
            ),
            'requires_scaling': True
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=1
            ),
            'requires_scaling': False
        },
        'XGBoost': {
            'model': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                n_jobs=1
            ),
            'requires_scaling': False
        },
        'LightGBM': {
            'model': LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=1,
                verbose=-1
            ),
            'requires_scaling': False
        }
    }
    
    return models


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_all_models(X_train, y_train):
    """
    Train all models.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        dict: Trained models
    """
    logger.info("="*80)
    logger.info("MODEL TRAINING")
    logger.info("="*80)
    
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    
    model_configs = get_model_configs(n_pos, n_neg)
    trained_models = {}
    
    for model_name, config in model_configs.items():
        logger.info(f"\n  Training {model_name}...")
        
        model = config['model']
        requires_scaling = config['requires_scaling']
        
        # Scale if needed
        scaler = None
        if requires_scaling:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        
        logger.info(f"    ✓ {model_name} trained")
        
        trained_models[model_name] = {
            'model': model,
            'scaler': scaler,
            'requires_scaling': requires_scaling
        }
    
    logger.info("\n✓ All models trained successfully")
    
    return trained_models


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, scaler=None):
    """
    Evaluate a single model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        scaler: Fitted scaler (if needed)
        
    Returns:
        dict: Evaluation metrics
    """
    if scaler is not None:
        X_test = scaler.transform(X_test)
    
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
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['true_positives'] = int(cm[1, 1])
    
    return metrics


def evaluate_all_models(trained_models, X_test, y_test):
    """
    Evaluate all trained models on test set.
    
    Args:
        trained_models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Evaluation results for all models
    """
    logger.info("="*80)
    logger.info("MODEL EVALUATION ON TEST SET")
    logger.info("="*80)
    
    results = {}
    
    for model_name, model_info in trained_models.items():
        logger.info(f"\n  Evaluating {model_name}...")
        
        metrics = evaluate_model(
            model_info['model'], 
            X_test, 
            y_test,
            model_info['scaler']
        )
        
        results[model_name] = metrics
        
        logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall:    {metrics['recall']:.4f}")
        logger.info(f"    F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"    PR-AUC:    {metrics['pr_auc']:.4f}")
        logger.info(f"    Confusion Matrix:")
        logger.info(f"      TN: {metrics['true_negatives']:,}  FP: {metrics['false_positives']:,}")
        logger.info(f"      FN: {metrics['false_negatives']:,}  TP: {metrics['true_positives']:,}")
    
    return results


def create_comparison_table(results):
    """Create a comparison table of all models."""
    
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc'],
            'FP': metrics['false_positives'],
            'FN': metrics['false_negatives']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('PR-AUC', ascending=False)
    
    return comparison_df


# ============================================================================
# MODEL SELECTION & SAVING
# ============================================================================

def select_best_model(results, trained_models, metric='pr_auc'):
    """
    Select the best model based on specified metric.
    
    Args:
        results: Dictionary of evaluation results
        trained_models: Dictionary of trained models
        metric: Metric to use for selection
        
    Returns:
        tuple: (best_model_name, best_model, best_metrics, scaler)
    """
    logger.info("="*80)
    logger.info("MODEL SELECTION")
    logger.info("="*80)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x][metric])
    best_metrics = results[best_model_name]
    best_model_info = trained_models[best_model_name]
    
    logger.info(f"\n🏆 Best Model: {best_model_name}")
    logger.info(f"   Selection metric: {metric.upper()} = {best_metrics[metric]:.4f}")
    logger.info(f"\n   All metrics:")
    logger.info(f"     Accuracy:  {best_metrics['accuracy']:.4f}")
    logger.info(f"     Precision: {best_metrics['precision']:.4f}")
    logger.info(f"     Recall:    {best_metrics['recall']:.4f}")
    logger.info(f"     F1 Score:  {best_metrics['f1_score']:.4f}")
    logger.info(f"     ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    logger.info(f"     PR-AUC:    {best_metrics['pr_auc']:.4f}")
    
    return best_model_name, best_model_info, best_metrics


def save_model_artifacts(best_model_name, best_model_info, best_metrics, 
                          feature_names, output_dir=OUTPUT_DIR):
    """
    Save model artifacts to disk.
    
    Args:
        best_model_name: Name of the best model
        best_model_info: Best model info dict
        best_metrics: Evaluation metrics
        feature_names: List of feature names
        output_dir: Path to save models
    """
    logger.info("="*80)
    logger.info("SAVING MODEL ARTIFACTS")
    logger.info("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = f"{output_dir}/order_based_best_model.pkl"
    joblib.dump(best_model_info['model'], model_path)
    logger.info(f"  ✓ Saved model to: {model_path}")
    
    # Save scaler if needed
    if best_model_info['scaler'] is not None:
        scaler_path = f"{output_dir}/order_based_scaler.pkl"
        joblib.dump(best_model_info['scaler'], scaler_path)
        logger.info(f"  ✓ Saved scaler to: {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'requires_scaling': best_model_info['requires_scaling'],
        'feature_names': feature_names,
        'approach': 'order_based',
        'description': 'Post-purchase repurchase prediction (one entry per order)',
        'metrics': {
            'accuracy': float(best_metrics['accuracy']),
            'precision': float(best_metrics['precision']),
            'recall': float(best_metrics['recall']),
            'f1_score': float(best_metrics['f1_score']),
            'roc_auc': float(best_metrics['roc_auc']),
            'pr_auc': float(best_metrics['pr_auc']),
            'true_negatives': best_metrics['true_negatives'],
            'false_positives': best_metrics['false_positives'],
            'false_negatives': best_metrics['false_negatives'],
            'true_positives': best_metrics['true_positives']
        },
        'created_at': datetime.now().isoformat()
    }
    
    metadata_path = f"{output_dir}/order_based_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  ✓ Saved metadata to: {metadata_path}")
    
    logger.info("\n✓ All artifacts saved successfully")
    
    return model_path, metadata_path


def save_comparison_results(comparison_df, results, output_dir=OUTPUT_DIR):
    """Save model comparison results."""
    
    # Save comparison table
    comparison_path = f"{output_dir}/order_based_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"  ✓ Saved comparison table to: {comparison_path}")
    
    # Save full results
    results_serializable = {}
    for model_name, metrics in results.items():
        results_serializable[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc']),
            'pr_auc': float(metrics['pr_auc']),
            'true_negatives': metrics['true_negatives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives'],
            'true_positives': metrics['true_positives']
        }
    
    results_path = f"{output_dir}/order_based_all_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    logger.info(f"  ✓ Saved full results to: {results_path}")


# ============================================================================
# COMPLETE TRAINING PIPELINE
# ============================================================================

def run_order_based_training_pipeline():
    """
    Run the complete order-based model training pipeline.
    
    Returns:
        dict: Pipeline results including best model and metrics
    """
    logger.info("="*80)
    logger.info("ORDER-BASED MODEL TRAINING PIPELINE")
    logger.info("="*80)
    logger.info("Approach: Post-purchase repurchase prediction")
    logger.info("Question: Given a user just ordered, will they order again in 12 weeks?")
    
    # Step 1: Load data
    train_df, test_df = load_train_test_data()
    
    # Step 2: Prepare features
    logger.info("\n" + "="*80)
    logger.info("PREPARING FEATURES")
    logger.info("="*80)
    
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)
    
    logger.info(f"Training features: {X_train.shape}")
    logger.info(f"Test features: {X_test.shape}")
    logger.info(f"Feature count: {len(feature_names)}")
    
    # Step 3: Train all models
    trained_models = train_all_models(X_train, y_train)
    
    # Step 4: Evaluate on test set
    results = evaluate_all_models(trained_models, X_test, y_test)
    
    # Step 5: Create comparison table
    comparison_df = create_comparison_table(results)
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)
    print("\n" + comparison_df.to_string(index=False))
    
    # Step 6: Select best model
    best_model_name, best_model_info, best_metrics = select_best_model(
        results, trained_models, metric='pr_auc'
    )
    
    # Step 7: Save model artifacts
    save_model_artifacts(
        best_model_name, best_model_info, best_metrics, feature_names
    )
    
    # Step 8: Save comparison results
    save_comparison_results(comparison_df, results)
    
    logger.info("\n" + "="*80)
    logger.info("✓ ORDER-BASED MODEL TRAINING PIPELINE COMPLETE")
    logger.info("="*80)
    
    return {
        'best_model': best_model_info['model'],
        'best_model_name': best_model_name,
        'best_metrics': best_metrics,
        'scaler': best_model_info['scaler'],
        'feature_names': feature_names,
        'comparison_df': comparison_df,
        'all_results': results
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Run the training pipeline
    results = run_order_based_training_pipeline()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Model: {results['best_model_name']}")
    print(f"Precision: {results['best_metrics']['precision']:.4f}")
    print(f"Recall: {results['best_metrics']['recall']:.4f}")
    print(f"F1 Score: {results['best_metrics']['f1_score']:.4f}")
    print(f"ROC-AUC: {results['best_metrics']['roc_auc']:.4f}")
    print(f"PR-AUC: {results['best_metrics']['pr_auc']:.4f}")
