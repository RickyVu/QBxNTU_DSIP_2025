# ---
# Main Pipeline for Repurchase Prediction Model
# Orchestrates: Feature Engineering → Model Training → Evaluation → SHAP → User Aggregation → Visualization
# ---

import pandas as pd
import numpy as np
import os
import sys
import logging
import json
from datetime import datetime

# Setup logging
os.makedirs('outputs/repurchase', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('outputs/repurchase/pipeline.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Import modules
from repurchase_features import run_repurchase_feature_engineering
from repurchase_revenue_impact import run_revenue_impact_analysis
from repurchase_model import (
    load_train_test_data,
    prepare_features,
    train_all_models,
    evaluate_all_models,
    create_comparison_table,
    select_best_model,
    save_model_artifacts,
    save_comparison_results
)
from repurchase_visualizations import (
    generate_all_evaluation_plots,
    generate_segment_plots
)
from repurchase_shap_analysis import run_repurchase_shap_analysis
from repurchase_user_aggregation import run_user_shap_analysis

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TARGET_NAME = 'will_repurchase'
CLUSTER_PATH = "../cluster01/sg_user.csv"
OUTPUT_DIR = "outputs/repurchase"
MODELS_PATH = "outputs/repurchase/models"
PLOTS_PATH = "outputs/repurchase/visualizations"

# ============================================================================
# PIPELINE STEPS
# ============================================================================

def step_1_feature_engineering(prediction_weeks=12):
    """
    Step 1: Run feature engineering pipeline.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 1: FEATURE ENGINEERING")
    logger.info("="*80)
    
    features_df, train_df, test_df = run_repurchase_feature_engineering(
        prediction_weeks=prediction_weeks,
        output_dir=OUTPUT_DIR
    )
    
    return {
        'features_df': features_df,
        'train_df': train_df,
        'test_df': test_df
    }


def step_2_model_training():
    """
    Step 2: Load features and train all models.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("="*80)
    
    # Load train/test data
    train_df, test_df = load_train_test_data()
    
    # Prepare features
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)
    
    logger.info(f"Training features: {X_train.shape}")
    logger.info(f"Test features: {X_test.shape}")
    logger.info(f"Feature count: {len(feature_names)}")
    
    # Get user IDs for later segment analysis
    user_ids_test = test_df['user_id']
    
    # Train all models
    trained_models = train_all_models(X_train, y_train)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'user_ids_test': user_ids_test,
        'trained_models': trained_models,
        'train_df': train_df,
        'test_df': test_df
    }


def step_3_model_evaluation(training_results, force_model=None):
    """
    Step 3: Evaluate all models and select best (or use specified model).
    
    Args:
        training_results: Results from step 2
        force_model: If specified, use this model instead of auto-selecting best.
                     Options: 'LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM'
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 3: MODEL EVALUATION")
    logger.info("="*80)
    
    X_test = training_results['X_test']
    y_test = training_results['y_test']
    trained_models = training_results['trained_models']
    feature_names = training_results['feature_names']
    
    # Evaluate all models and store predictions for curves
    results = {}
    for model_name, model_info in trained_models.items():
        from repurchase_model import evaluate_model
        metrics = evaluate_model(
            model_info['model'], 
            X_test, 
            y_test,
            model_info['scaler']
        )
        
        # Get predictions for ROC/PR curves
        if model_info['scaler'] is not None:
            X_test_scaled = model_info['scaler'].transform(X_test)
            y_proba = model_info['model'].predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = model_info['model'].predict_proba(X_test)[:, 1]
        
        metrics['y_test'] = y_test
        metrics['y_proba'] = y_proba
        
        results[model_name] = metrics
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)
    print(comparison_df.to_string(index=False))
    
    # Select model (forced or best)
    if force_model is not None:
        # Validate model name
        if force_model not in trained_models:
            available = list(trained_models.keys())
            raise ValueError(f"Model '{force_model}' not found. Available models: {available}")
        
        logger.info(f"\n⚠️ USING SPECIFIED MODEL: {force_model}")
        selected_model_name = force_model
        selected_model_info = trained_models[force_model]
        selected_metrics = results[force_model]
    else:
        # Auto-select best model
        selected_model_name, selected_model_info, selected_metrics = select_best_model(
            results, trained_models, metric='pr_auc'
        )
    
    # Get selected model predictions
    if selected_model_info['scaler'] is not None:
        X_test_scaled = selected_model_info['scaler'].transform(X_test)
        selected_y_proba = selected_model_info['model'].predict_proba(X_test_scaled)[:, 1]
    else:
        selected_y_proba = selected_model_info['model'].predict_proba(X_test)[:, 1]
    
    # Save artifacts
    save_model_artifacts(
        selected_model_name, selected_model_info, selected_metrics, feature_names
    )
    save_comparison_results(comparison_df, results)
    
    return {
        'results': results,
        'comparison_df': comparison_df,
        'best_model': selected_model_info['model'],
        'best_model_name': selected_model_name,
        'best_metrics': selected_metrics,
        'best_scaler': selected_model_info['scaler'],
        'best_y_proba': selected_y_proba
    }


def step_4a_shap_analysis(training_results, evaluation_results):
    """
    Step 4a: Run order-level SHAP analysis on best model.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 4a: SHAP ANALYSIS (ORDER-LEVEL)")
    logger.info("="*80)

    best_model = evaluation_results['best_model']
    best_model_name = evaluation_results['best_model_name']
    best_scaler = evaluation_results['best_scaler']

    X_test = training_results['X_test']
    feature_names = training_results['feature_names']
    user_ids = training_results['user_ids_test']

    # Scale if needed
    if best_scaler is not None:
        X_test_for_shap = best_scaler.transform(X_test)
    else:
        X_test_for_shap = X_test

    # Run order-level SHAP analysis
    shap_results = run_repurchase_shap_analysis(
        best_model,
        X_test_for_shap,
        feature_names,
        user_ids,
        best_model_name,
        target_name=TARGET_NAME,
        models_path=MODELS_PATH,
        plots_path=PLOTS_PATH
    )

    return shap_results


def step_4b_user_shap_analysis(training_results, shap_results):
    """
    Step 4b: Aggregate order-level SHAP to user-level.
    Uses most recent order per user.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 4b: USER-LEVEL SHAP AGGREGATION")
    logger.info("="*80)

    X_test = training_results['X_test']
    user_ids = training_results['user_ids_test']
    test_df = training_results['test_df']

    # Run user-level SHAP analysis
    user_shap_results = run_user_shap_analysis(
        order_shap_results=shap_results,
        X_test=X_test,
        user_ids=user_ids,
        test_df=test_df,
        segmentation_df=None,  # Will load from CLUSTER_PATH
        target_name=TARGET_NAME,
        plots_path=PLOTS_PATH
    )

    return user_shap_results


def step_5_visualizations(training_results, evaluation_results):
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
    best_y_proba = evaluation_results['best_y_proba']
    
    y_test = training_results['y_test']
    user_ids = training_results['user_ids_test']
    
    # Generate evaluation plots
    plot_paths = generate_all_evaluation_plots(
        results, comparison_df, best_metrics, best_model_name,
        y_test, best_y_proba,
        target_name=TARGET_NAME, save_path=PLOTS_PATH
    )
    
    # Generate segment × propensity visualizations
    logger.info("\nGenerating segment × propensity plots...")
    
    try:
        seg_df = pd.read_csv(CLUSTER_PATH)
        if 'user_id' not in seg_df.columns and 'id' in seg_df.columns:
            seg_df = seg_df.rename(columns={'id': 'user_id'})
        
        propensity_df = pd.DataFrame({
            'user_id': user_ids.values,
            'propensity_score': best_y_proba
        })
        propensity_df = propensity_df.merge(seg_df[['user_id', 'value_cluster']], on='user_id', how='left')
        propensity_df = propensity_df.dropna(subset=['value_cluster'])
        
        if len(propensity_df) > 0:
            segment_plot_paths = generate_segment_plots(propensity_df, TARGET_NAME, PLOTS_PATH)
            plot_paths.update(segment_plot_paths)
            
            # Save propensity data
            propensity_path = f"{OUTPUT_DIR}/predictions"
            os.makedirs(propensity_path, exist_ok=True)
            propensity_df.to_csv(f"{propensity_path}/{TARGET_NAME}_propensity_by_segment.csv", index=False)
            logger.info(f"  ✓ Saved propensity data to: {propensity_path}/{TARGET_NAME}_propensity_by_segment.csv")
    except FileNotFoundError:
        logger.warning("  ⚠️ Segmentation file not found. Skipping segment plots.")
    
    return plot_paths


def step_6_revenue_impact_analysis(training_results, evaluation_results):
    """
    Step 6: Calculate revenue impact by segment (USER-LEVEL).
    
    Uses the dedicated revenue impact module which:
    - Aggregates to user level (no double counting)
    - Uses most recent order propensity per user
    - Uses user's average order value
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 6: REVENUE IMPACT ANALYSIS (USER-LEVEL)")
    logger.info("="*80)
    
    best_y_proba = evaluation_results['best_y_proba']
    test_df = training_results['test_df']
    
    # Use the dedicated revenue impact module
    results_df, summary, user_df = run_revenue_impact_analysis(
        test_df=test_df,
        propensity_scores=best_y_proba,
        cluster_path=CLUSTER_PATH,
        output_dir=OUTPUT_DIR
    )
    
    if results_df is None:
        logger.warning("  ⚠️ Revenue impact analysis failed.")
        return None
    
    return results_df, summary


def step_7_business_insights(evaluation_results, shap_results):
    """
    Step 7: Generate business insights summary.
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 7: BUSINESS INSIGHTS")
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
    insights.append(f"F1-Score: {best_metrics['f1_score']:.4f}")
    
    # Top Drivers
    insights.append("\n🎯 TOP DRIVERS OF REPURCHASE")
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
                direction = "+" if row['mean_shap'] > 0 else "-"
                insights.append(f"    {direction} {row['feature']}")
    
    # Strategic Recommendations
    insights.append("\n💡 STRATEGIC RECOMMENDATIONS")
    insights.append("-" * 40)
    insights.append("  1. Focus on 'Golden Whales' with low propensity - highest revenue impact")
    insights.append("  2. Use top SHAP drivers to personalize re-engagement campaigns")
    insights.append("  3. Implement segment-specific intervention strategies")
    insights.append("  4. Monitor propensity scores post-purchase for timely outreach")
    insights.append("  5. A/B test intervention strategies by segment")
    
    # Print insights
    for line in insights:
        logger.info(line)
    
    # Save insights to file
    insights_path = f"{MODELS_PATH}/{TARGET_NAME}_business_insights.txt"
    with open(insights_path, 'w') as f:
        f.write('\n'.join(insights))
    logger.info(f"\n  ✓ Saved insights to: {insights_path}")
    
    return insights


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_pipeline(skip_feature_engineering=False, prediction_weeks=12, force_model=None):
    """
    Run the complete repurchase prediction model pipeline.
    
    Args:
        skip_feature_engineering: If True, skip step 1 (use existing features)
        prediction_weeks: Number of weeks for repurchase prediction window
        force_model: If specified, use this model instead of auto-selecting best.
                     Options: 'LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM'
                     Default: None (auto-select best by PR-AUC)
        
    Returns:
        dict: Complete pipeline results
    """
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info("REPURCHASE PREDICTION MODEL - COMPLETE PIPELINE")
    logger.info("="*80)
    logger.info(f"Target: Will customer repurchase within {prediction_weeks} weeks?")
    if force_model:
        logger.info(f"Model: {force_model} (user specified)")
    else:
        logger.info("Model: Auto-select best by PR-AUC")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/predictions", exist_ok=True)
    
    # Step 1: Feature Engineering
    if not skip_feature_engineering:
        feature_results = step_1_feature_engineering(prediction_weeks)
    else:
        logger.info("\n⏭️ Skipping feature engineering (using existing features)")
        feature_results = None
    
    # Step 2: Model Training
    training_results = step_2_model_training()
    
    # Step 3: Model Evaluation (with optional forced model selection)
    evaluation_results = step_3_model_evaluation(training_results, force_model=force_model)
    
    # Step 4a: Order-Level SHAP Analysis
    shap_results = step_4a_shap_analysis(training_results, evaluation_results)

    # Step 4b: User-Level SHAP Aggregation
    user_shap_results = step_4b_user_shap_analysis(training_results, shap_results)

    # Step 5: Visualizations
    plot_paths = step_5_visualizations(training_results, evaluation_results)

    # Step 6: Revenue Impact Analysis
    revenue_results = step_6_revenue_impact_analysis(training_results, evaluation_results)

    # Step 7: Business Insights
    insights = step_7_business_insights(evaluation_results, shap_results)
    
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
    logger.info(f"F1-Score: {evaluation_results['best_metrics']['f1_score']:.4f}")
    logger.info("\nOutputs saved to:")
    logger.info(f"  - {OUTPUT_DIR}/  (features)")
    logger.info(f"  - {MODELS_PATH}/  (models, SHAP)")
    logger.info(f"  - {PLOTS_PATH}/  (visualizations)")
    logger.info(f"  - {OUTPUT_DIR}/predictions/  (propensity scores)")
    logger.info(f"  - {OUTPUT_DIR}/revenue_impact/  (revenue analysis)")
    
    return {
        'feature_results': feature_results,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'shap_results': shap_results,
        'user_shap_results': user_shap_results,
        'plot_paths': plot_paths,
        'revenue_results': revenue_results,
        'insights': insights
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    # 
    # Options:
    #   skip_feature_engineering: Set to True to skip if features already exist
    #   prediction_weeks: Number of weeks for repurchase prediction window
    #   force_model: Specify a model to use instead of auto-selecting best
    #                Options: 'LogisticRegression', 'RandomForest', 'XGBoost', 'LightGBM'
    #                Default: None (auto-select best by PR-AUC)
    #
    # Example with forced model:
    #   results = run_complete_pipeline(skip_feature_engineering=True, force_model='XGBoost')
    
    results = run_complete_pipeline(
        skip_feature_engineering=False,  # Set to True to skip if features already exist
        prediction_weeks=12,
        force_model=None  # Set to 'LogisticRegression', 'RandomForest', 'XGBoost', or 'LightGBM' to force a specific model
    )
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"Selected Model: {results['evaluation_results']['best_model_name']}")
    print(f"ROC-AUC: {results['evaluation_results']['best_metrics']['roc_auc']:.4f}")
    print(f"PR-AUC: {results['evaluation_results']['best_metrics']['pr_auc']:.4f}")
    print(f"F1-Score: {results['evaluation_results']['best_metrics']['f1_score']:.4f}")
