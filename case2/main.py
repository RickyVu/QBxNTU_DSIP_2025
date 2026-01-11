# ---
# Main Pipeline for Case 2: Propensity Prediction
# Orchestrates: Feature Engineering → Model Training → Evaluation → SHAP → Visualization
# ---

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('outputs/pipeline.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Import modules
from feature_engineering import run_complete_feature_engineering
from model_training import (
    load_processed_features,
    prepare_train_test_split,
    scale_features,
    train_all_models,
    evaluate_all_models,
    create_comparison_table,
    select_best_model,
    save_model_artifacts,
    save_comparison_results
)
from visualizations import (
    generate_all_evaluation_plots,
    plot_segment_propensity_boxplot,
    plot_segment_propensity_violin,
    plot_segment_propensity_heatmap,
    plot_strategic_matrix
)
from shap_analysis import run_shap_analysis_pipeline

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_NAME = 'will_purchase'  # 'will_purchase' or 'will_be_high_value'

# ============================================================================
# PIPELINE STEPS
# ============================================================================

def step_1_feature_engineering(target_type='loyalty'):
    """
    Step 1: Run feature engineering pipeline.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 1: FEATURE ENGINEERING")
    logger.info("="*80)
    
    feature_results = run_complete_feature_engineering(target_type=target_type)
    
    return feature_results


def step_2_model_training(target_name=TARGET_NAME, test_size=TEST_SIZE, cv=CV_FOLDS):
    """
    Step 2: Load features and train all models.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("="*80)
    
    # Load processed features
    X, y, feature_names, user_ids = load_processed_features(target_name)
    
    # Train-test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y, test_size)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, feature_names)
    
    # Get class counts
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    
    # Train all models
    trained_models = train_all_models(X_train, X_train_scaled, y_train, n_pos, n_neg, cv)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'user_ids': user_ids,
        'trained_models': trained_models
    }


def step_3_model_evaluation(training_results, target_name=TARGET_NAME):
    """
    Step 3: Evaluate all models and select best.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 3: MODEL EVALUATION")
    logger.info("="*80)
    
    X_test = training_results['X_test']
    X_test_scaled = training_results['X_test_scaled']
    y_test = training_results['y_test']
    trained_models = training_results['trained_models']
    scaler = training_results['scaler']
    feature_names = training_results['feature_names']
    
    # Evaluate all models
    results = evaluate_all_models(trained_models, X_test, X_test_scaled, y_test)
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)
    print(comparison_df.to_string(index=False))
    
    # Select best model (use PR-AUC if imbalanced)
    target_rate = y_test.mean()
    selection_metric = 'pr_auc' if target_rate < 0.3 else 'roc_auc'
    
    best_model_name, best_model, best_metrics, requires_scaling = select_best_model(
        results, trained_models, metric=selection_metric
    )
    
    # Save artifacts
    save_model_artifacts(
        best_model, best_model_name, best_metrics, scaler,
        feature_names, requires_scaling, target_name
    )
    save_comparison_results(comparison_df, results, target_name)
    
    return {
        'results': results,
        'comparison_df': comparison_df,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_metrics': best_metrics,
        'requires_scaling': requires_scaling
    }


def step_4_shap_analysis(training_results, evaluation_results, target_name=TARGET_NAME):
    """
    Step 4: Run SHAP analysis on best model.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 4: SHAP ANALYSIS")
    logger.info("="*80)
    
    best_model = evaluation_results['best_model']
    best_model_name = evaluation_results['best_model_name']
    requires_scaling = evaluation_results['requires_scaling']
    
    # Use scaled or unscaled data based on model
    X_test = training_results['X_test_scaled'] if requires_scaling else training_results['X_test']
    feature_names = training_results['feature_names']
    user_ids = training_results['user_ids'].iloc[training_results['X_test'].index]
    
    # Run SHAP analysis
    shap_results = run_shap_analysis_pipeline(
        best_model, X_test, feature_names, user_ids, best_model_name, target_name
    )
    
    return shap_results


def step_5_visualizations(training_results, evaluation_results, target_name=TARGET_NAME):
    """
    Step 5: Generate all visualizations.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 5: VISUALIZATIONS")
    logger.info("="*80)
    
    results = evaluation_results['results']
    comparison_df = evaluation_results['comparison_df']
    best_metrics = evaluation_results['best_metrics']
    best_model_name = evaluation_results['best_model_name']
    best_model = evaluation_results['best_model']
    requires_scaling = evaluation_results['requires_scaling']
    
    # Use scaled or unscaled data based on model
    X_test = training_results['X_test_scaled'] if requires_scaling else training_results['X_test']
    y_test = training_results['y_test']
    user_ids = training_results['user_ids'].iloc[training_results['X_test'].index]
    
    # Generate evaluation plots
    plot_paths = generate_all_evaluation_plots(
        results, comparison_df, best_metrics, best_model_name, target_name
    )
    
    # Generate segment × propensity visualizations
    logger.info("\nGenerating segment × propensity plots...")
    
    # Get propensity scores
    propensity_scores = best_model.predict_proba(X_test)[:, 1]
    
    # Load segmentation
    try:
        seg_df = pd.read_csv("../cluster01/sg_user.csv")
        if 'user_id' not in seg_df.columns and 'id' in seg_df.columns:
            seg_df = seg_df.rename(columns={'id': 'user_id'})
        
        propensity_df = pd.DataFrame({
            'user_id': user_ids.values,
            'propensity_score': propensity_scores
        })
        propensity_df = propensity_df.merge(seg_df[['user_id', 'value_cluster']], on='user_id', how='left')
        propensity_df = propensity_df.dropna(subset=['value_cluster'])
        
        if len(propensity_df) > 0:
            plot_paths['segment_boxplot'] = plot_segment_propensity_boxplot(propensity_df, target_name)
            plot_paths['segment_violin'] = plot_segment_propensity_violin(propensity_df, target_name)
            plot_paths['segment_heatmap'] = plot_segment_propensity_heatmap(propensity_df, target_name)
            plot_paths['strategic_matrix'] = plot_strategic_matrix(propensity_df, target_name)
            
            # Save propensity data
            propensity_path = f"outputs/predictions/{target_name}_propensity_by_segment.csv"
            os.makedirs("outputs/predictions", exist_ok=True)
            propensity_df.to_csv(propensity_path, index=False)
            logger.info(f"  ✓ Saved propensity data to: {propensity_path}")
    except FileNotFoundError:
        logger.warning("  ⚠️ Segmentation file not found. Skipping segment plots.")
    
    return plot_paths


def step_6_business_insights(evaluation_results, shap_results, target_name=TARGET_NAME):
    """
    Step 6: Generate business insights summary.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 6: BUSINESS INSIGHTS")
    logger.info("="*80)
    
    best_model_name = evaluation_results['best_model_name']
    best_metrics = evaluation_results['best_metrics']
    top_drivers = shap_results['top_drivers']
    segment_results = shap_results.get('segment_results', {})
    
    insights = []
    
    # Model Performance Summary
    insights.append("\n📊 MODEL PERFORMANCE SUMMARY")
    insights.append("-" * 40)
    insights.append(f"Best Model: {best_model_name}")
    insights.append(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
    insights.append(f"PR-AUC: {best_metrics['pr_auc']:.4f}")
    insights.append(f"Precision: {best_metrics['precision']:.4f}")
    insights.append(f"Recall: {best_metrics['recall']:.4f}")
    
    # Top Drivers
    insights.append("\n🎯 TOP DRIVERS OF CUSTOMER LOYALTY")
    insights.append("-" * 40)
    for i, row in top_drivers.head(10).iterrows():
        insights.append(f"  {row['rank']:2d}. {row['feature']}")
    
    # Segment-specific insights
    if segment_results:
        insights.append("\n👥 SEGMENT-SPECIFIC INSIGHTS")
        insights.append("-" * 40)
        
        for segment_name, results in segment_results.items():
            insights.append(f"\n  {segment_name} (n={results['n_samples']})")
            top_5 = results['top_10'].head(5)
            for _, row in top_5.iterrows():
                insights.append(f"    - {row['feature']}")
    
    # Strategic Recommendations
    insights.append("\n💡 STRATEGIC RECOMMENDATIONS")
    insights.append("-" * 40)
    insights.append("  1. Focus retention efforts on top loyalty drivers identified by SHAP")
    insights.append("  2. Prioritize 'Golden Whales' with low propensity for win-back campaigns")
    insights.append("  3. Accelerate 'High Potential' customers with high propensity to VIP status")
    insights.append("  4. Implement targeted re-engagement for 'Drifting Risk' segment")
    insights.append("  5. Use propensity scores to personalize marketing communications")
    
    # Print insights
    for line in insights:
        logger.info(line)
    
    # Save insights to file
    insights_path = f"outputs/models/{target_name}_business_insights.txt"
    with open(insights_path, 'w') as f:
        f.write('\n'.join(insights))
    logger.info(f"\n  ✓ Saved insights to: {insights_path}")
    
    return insights


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_pipeline(target_type='loyalty', skip_feature_engineering=False):
    """
    Run the complete Case 2 pipeline.
    
    Args:
        target_type: 'loyalty' (will_purchase) or 'high_value' (will_be_high_value)
        skip_feature_engineering: If True, skip step 1 (use existing features)
        
    Returns:
        dict: Complete pipeline results
    """
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info("CASE 2: PROPENSITY PREDICTION - COMPLETE PIPELINE")
    logger.info("="*80)
    logger.info(f"Target type: {target_type}")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine target name
    target_name = 'will_purchase' if target_type == 'loyalty' else 'will_be_high_value'
    
    # Create output directories
    os.makedirs("outputs/features", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)
    
    # Step 1: Feature Engineering
    if not skip_feature_engineering:
        feature_results = step_1_feature_engineering(target_type)
    else:
        logger.info("\n⏭️ Skipping feature engineering (using existing features)")
        feature_results = None
    
    # Step 2: Model Training
    training_results = step_2_model_training(target_name)
    
    # Step 3: Model Evaluation
    evaluation_results = step_3_model_evaluation(training_results, target_name)
    
    # Step 4: SHAP Analysis
    shap_results = step_4_shap_analysis(training_results, evaluation_results, target_name)
    
    # Step 5: Visualizations
    plot_paths = step_5_visualizations(training_results, evaluation_results, target_name)
    
    # Step 6: Business Insights
    insights = step_6_business_insights(evaluation_results, shap_results, target_name)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "="*80)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration}")
    logger.info(f"Best Model: {evaluation_results['best_model_name']}")
    logger.info(f"ROC-AUC: {evaluation_results['best_metrics']['roc_auc']:.4f}")
    logger.info(f"PR-AUC: {evaluation_results['best_metrics']['pr_auc']:.4f}")
    logger.info("\nOutputs saved to:")
    logger.info("  - outputs/features/  (processed features)")
    logger.info("  - outputs/models/    (trained models, metrics, SHAP)")
    logger.info("  - outputs/plots/     (visualizations)")
    logger.info("  - outputs/predictions/ (propensity scores)")
    
    return {
        'feature_results': feature_results,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'shap_results': shap_results,
        'plot_paths': plot_paths,
        'insights': insights
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the complete pipeline for loyalty prediction
    results = run_complete_pipeline(
        target_type='loyalty',
        skip_feature_engineering=False  # Set to True to skip if features already exist
    )
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"Best Model: {results['evaluation_results']['best_model_name']}")
    print(f"ROC-AUC: {results['evaluation_results']['best_metrics']['roc_auc']:.4f}")
