# ---
# SHAP Analysis for Case 2: Propensity Prediction
# ---

import pandas as pd
import numpy as np
import os
import json
import logging
import joblib
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
MODELS_PATH = "outputs/models"
PLOTS_PATH = "outputs/plots"
CLUSTER_PATH = "../cluster01/sg_user.csv"

# ============================================================================
# SHAP VALUE CALCULATION
# ============================================================================

def calculate_shap_values(model, X_test, model_name):
    """
    Calculate SHAP values for the model.
    
    Args:
        model: Trained model
        X_test: Test features DataFrame
        model_name: Name of the model (for selecting appropriate explainer)
        
    Returns:
        tuple: (shap_values, explainer)
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed. Run: pip install shap")
        raise ImportError("SHAP package required. Install with: pip install shap")
    
    logger.info("="*80)
    logger.info("CALCULATING SHAP VALUES")
    logger.info("="*80)
    
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Test samples: {len(X_test)}")
    
    # Select appropriate explainer based on model type
    if 'Logistic' in model_name:
        logger.info("  Using LinearExplainer...")
        explainer = shap.LinearExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)
    elif 'Random Forest' in model_name or 'XGBoost' in model_name or 'LightGBM' in model_name:
        logger.info("  Using TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # For binary classification, tree models may return list [neg_class, pos_class]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class
    else:
        logger.info("  Using KernelExplainer (slower)...")
        # Sample for efficiency
        background = shap.sample(X_test, min(100, len(X_test)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    
    logger.info(f"  ✓ SHAP values calculated: shape {shap_values.shape}")
    
    return shap_values, explainer


def extract_top_drivers(shap_values, feature_names, top_n=20):
    """
    Extract top N drivers based on mean absolute SHAP values.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        pd.DataFrame: Top drivers with their importance scores
    """
    logger.info(f"\nExtracting top {top_n} drivers...")
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Add rank
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    top_drivers = importance_df.head(top_n).reset_index(drop=True)
    
    logger.info("\n  Top Drivers:")
    for _, row in top_drivers.head(10).iterrows():
        logger.info(f"    {row['rank']:2d}. {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    return importance_df, top_drivers


# ============================================================================
# SEGMENT-SPECIFIC SHAP ANALYSIS
# ============================================================================

def load_segmentation():
    """
    Load Case 1 segmentation data.
    
    Returns:
        pd.DataFrame: Segmentation data with user_id and value_cluster
    """
    logger.info("\nLoading segmentation data...")
    
    if not os.path.exists(CLUSTER_PATH):
        raise FileNotFoundError(f"Segmentation file not found: {CLUSTER_PATH}")
    
    seg_df = pd.read_csv(CLUSTER_PATH)
    
    # Ensure user_id column exists
    if 'user_id' not in seg_df.columns:
        if 'id' in seg_df.columns:
            seg_df = seg_df.rename(columns={'id': 'user_id'})
    
    logger.info(f"  ✓ Loaded {len(seg_df)} users with segments")
    
    return seg_df[['user_id', 'value_cluster']]


def analyze_by_segment(shap_values, X_test, feature_names, user_ids, segmentation_df):
    """
    Calculate SHAP values per Case 1 segment.
    
    Args:
        shap_values: SHAP values array
        X_test: Test features DataFrame
        feature_names: List of feature names
        user_ids: User IDs for test set
        segmentation_df: DataFrame with user_id and value_cluster
        
    Returns:
        dict: SHAP analysis results per segment
    """
    logger.info("="*80)
    logger.info("SEGMENT-SPECIFIC SHAP ANALYSIS")
    logger.info("="*80)
    
    # Define segment names
    segment_names = {
        0: 'Casual Walk-in',
        1: 'Golden Whales',
        2: 'High Potential',
        3: 'Drifting Risk'
    }
    
    # Create DataFrame with SHAP values and user info
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df['user_id'] = user_ids.values
    
    # Merge with segmentation
    shap_df = shap_df.merge(segmentation_df, on='user_id', how='left')
    
    segment_results = {}
    
    for segment_id, segment_name in segment_names.items():
        logger.info(f"\n  Segment: {segment_name} (Cluster {segment_id})")
        
        # Filter to segment
        segment_mask = shap_df['value_cluster'] == segment_id
        segment_shap = shap_df[segment_mask][feature_names].values
        
        if len(segment_shap) == 0:
            logger.info(f"    No samples in this segment")
            continue
        
        logger.info(f"    Samples: {len(segment_shap)}")
        
        # Calculate mean absolute SHAP for segment
        mean_abs_shap = np.abs(segment_shap).mean(axis=0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        segment_results[segment_name] = {
            'segment_id': segment_id,
            'n_samples': len(segment_shap),
            'importance': importance_df,
            'top_10': importance_df.head(10),
            'shap_values': segment_shap
        }
        
        # Log top 5 drivers for this segment
        logger.info("    Top 5 drivers:")
        for _, row in importance_df.head(5).iterrows():
            logger.info(f"      {row['rank']:2d}. {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    return segment_results


def compare_segment_drivers(segment_results, top_n=10):
    """
    Compare top drivers across segments.
    
    Args:
        segment_results: Dictionary of segment SHAP results
        top_n: Number of top features to compare
        
    Returns:
        pd.DataFrame: Comparison table
    """
    logger.info("\n" + "="*80)
    logger.info("COMPARING DRIVERS ACROSS SEGMENTS")
    logger.info("="*80)
    
    comparison_data = []
    
    for segment_name, results in segment_results.items():
        top_features = results['top_10'].head(top_n)
        for _, row in top_features.iterrows():
            comparison_data.append({
                'segment': segment_name,
                'feature': row['feature'],
                'importance': row['mean_abs_shap'],
                'rank': row['rank']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Pivot to show features as rows, segments as columns
    pivot_df = comparison_df.pivot_table(
        index='feature',
        columns='segment',
        values='importance',
        aggfunc='first'
    ).fillna(0)
    
    # Sort by average importance across segments
    pivot_df['avg_importance'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('avg_importance', ascending=False)
    
    logger.info("\n  Top features across all segments:")
    print(pivot_df.head(10).to_string())
    
    return comparison_df, pivot_df


# ============================================================================
# SHAP VISUALIZATIONS
# ============================================================================

def plot_shap_summary(shap_values, X_test, feature_names, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot SHAP summary plot (beeswarm).
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Skipping plot.")
        return None
    
    logger.info("\nCreating SHAP summary plot...")
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                     show=False, max_display=20)
    
    plt.title('SHAP Feature Importance (Overall)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_shap_summary.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved to: {file_path}")
    
    return file_path


def plot_shap_bar(shap_values, feature_names, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot SHAP bar chart (mean absolute values).
    """
    logger.info("\nCreating SHAP bar plot...")
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Sort and get top 20
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:20]
    top_features = [feature_names[i] for i in sorted_idx]
    top_values = mean_abs_shap[sorted_idx]
    
    plt.figure(figsize=(10, 10))
    plt.barh(range(len(top_features)), top_values[::-1], color='steelblue')
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.title('Top 20 Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_shap_bar.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved to: {file_path}")
    
    return file_path


def plot_shap_by_segment(segment_results, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot SHAP importance comparison across segments.
    """
    logger.info("\nCreating segment comparison plot...")
    
    # Collect top 10 features from each segment
    all_top_features = set()
    for segment_name, results in segment_results.items():
        top_10 = results['top_10']['feature'].tolist()
        all_top_features.update(top_10)
    
    # Get importance for each feature in each segment
    plot_data = []
    for feature in all_top_features:
        for segment_name, results in segment_results.items():
            importance_df = results['importance']
            importance = importance_df[importance_df['feature'] == feature]['mean_abs_shap'].values
            if len(importance) > 0:
                plot_data.append({
                    'feature': feature,
                    'segment': segment_name,
                    'importance': importance[0]
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Pivot for heatmap
    pivot_df = plot_df.pivot(index='feature', columns='segment', values='importance').fillna(0)
    
    # Sort by average importance
    pivot_df['avg'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('avg', ascending=False).drop('avg', axis=1).head(15)
    
    plt.figure(figsize=(12, 10))
    import seaborn as sns
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Mean |SHAP Value|'})
    plt.title('Feature Importance by Segment', fontsize=14, fontweight='bold')
    plt.xlabel('Customer Segment', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_shap_by_segment.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved to: {file_path}")
    
    return file_path


# ============================================================================
# SAVE SHAP RESULTS
# ============================================================================

def save_shap_results(importance_df, top_drivers, segment_results, target_name='will_purchase', models_path=MODELS_PATH):
    """
    Save SHAP analysis results.
    """
    logger.info("\n" + "="*80)
    logger.info("SAVING SHAP RESULTS")
    logger.info("="*80)
    
    os.makedirs(models_path, exist_ok=True)
    
    # Save overall importance
    importance_path = f"{models_path}/{target_name}_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"  ✓ Saved feature importance to: {importance_path}")
    
    # Save top drivers
    top_drivers_path = f"{models_path}/{target_name}_top_drivers.csv"
    top_drivers.to_csv(top_drivers_path, index=False)
    logger.info(f"  ✓ Saved top drivers to: {top_drivers_path}")
    
    # Save segment-specific results
    for segment_name, results in segment_results.items():
        segment_path = f"{models_path}/{target_name}_importance_{segment_name.replace(' ', '_').lower()}.csv"
        results['importance'].to_csv(segment_path, index=False)
        logger.info(f"  ✓ Saved {segment_name} importance to: {segment_path}")
    
    logger.info("\n✓ All SHAP results saved")


# ============================================================================
# COMPLETE SHAP ANALYSIS PIPELINE
# ============================================================================

def run_shap_analysis_pipeline(model, X_test, feature_names, user_ids, model_name,
                               target_name='will_purchase'):
    """
    Run the complete SHAP analysis pipeline.
    
    Args:
        model: Trained model
        X_test: Test features DataFrame
        feature_names: List of feature names
        user_ids: User IDs for test set
        model_name: Name of the model
        target_name: Name of target variable
        
    Returns:
        dict: SHAP analysis results
    """
    logger.info("="*80)
    logger.info(f"SHAP ANALYSIS PIPELINE - {target_name.upper()}")
    logger.info("="*80)
    
    # Step 1: Calculate SHAP values
    shap_values, explainer = calculate_shap_values(model, X_test, model_name)
    
    # Step 2: Extract top drivers
    importance_df, top_drivers = extract_top_drivers(shap_values, feature_names, top_n=20)
    
    # Step 3: Load segmentation and analyze by segment
    try:
        segmentation_df = load_segmentation()
        segment_results = analyze_by_segment(
            shap_values, X_test, feature_names, user_ids, segmentation_df
        )
        comparison_df, pivot_df = compare_segment_drivers(segment_results)
    except FileNotFoundError as e:
        logger.warning(f"  ⚠️ Segmentation not available: {e}")
        segment_results = {}
        comparison_df = None
        pivot_df = None
    
    # Step 4: Generate plots
    plot_paths = {}
    plot_paths['summary'] = plot_shap_summary(shap_values, X_test, feature_names, target_name)
    plot_paths['bar'] = plot_shap_bar(shap_values, feature_names, target_name)
    
    if segment_results:
        plot_paths['by_segment'] = plot_shap_by_segment(segment_results, target_name)
    
    # Step 5: Save results
    save_shap_results(importance_df, top_drivers, segment_results, target_name)
    
    logger.info("\n" + "="*80)
    logger.info("✓ SHAP ANALYSIS COMPLETE")
    logger.info("="*80)
    
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'importance_df': importance_df,
        'top_drivers': top_drivers,
        'segment_results': segment_results,
        'comparison_df': comparison_df,
        'plot_paths': plot_paths
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("SHAP Analysis module loaded successfully.")
    print("Use run_shap_analysis_pipeline() after training models.")
    print("Requires: model, X_test, feature_names, user_ids, model_name")
