# ---
# Repurchase User Aggregation Module
# Aggregates order-level SHAP values to user-level for business insights
# Uses most recent order per user
# ---

import pandas as pd
import numpy as np
import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
PLOTS_PATH = "outputs/repurchase/visualizations"
CLUSTER_PATH = "../cluster01/sg_user.csv"

# Segment names from Case 1
SEGMENT_NAMES = {
    0: 'Casual Walk-in',
    1: 'Golden Whales',
    2: 'High Potential',
    3: 'Drifting Risk'
}

# ============================================================================
# USER-LEVEL AGGREGATION
# ============================================================================

def aggregate_order_shap_to_users(shap_values, X_test, user_ids, feature_names, test_df):
    """
    Aggregate order-level SHAP values to user-level using most recent order.

    This follows the same logic as revenue impact analysis:
    - For each user, use their MOST RECENT order's SHAP values
    - This represents the user's current state

    Args:
        shap_values: SHAP values array (order-level)
        X_test: Test features array (order-level)
        user_ids: User IDs for test set (Series or array)
        feature_names: List of feature names
        test_df: Test DataFrame with order_date

    Returns:
        pd.DataFrame: User-level DataFrame with aggregated SHAP values
    """
    logger.info("Aggregating order-level SHAP to user-level (using most recent order)...")

    # Create DataFrame with SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # Handle user_ids as Series or array
    if hasattr(user_ids, 'values'):
        shap_df['user_id'] = user_ids.values
    else:
        shap_df['user_id'] = user_ids

    # Add order_date from test_df
    shap_df['order_date'] = pd.to_datetime(test_df['order_date'].values)

    # Sort by order_date and get most recent order per user
    shap_df_sorted = shap_df.sort_values('order_date')
    user_shap_df = shap_df_sorted.groupby('user_id').last().reset_index()

    # Remove order_date column
    user_shap_df = user_shap_df.drop('order_date', axis=1)

    logger.info(f"  Orders in test set: {len(shap_df):,}")
    logger.info(f"  Unique users: {len(user_shap_df):,}")
    logger.info(f"  Avg orders per user: {len(shap_df) / len(user_shap_df):.2f}")

    return user_shap_df


def load_segmentation(cluster_path=CLUSTER_PATH):
    """
    Load Case 1 segmentation data.

    Returns:
        pd.DataFrame: Segmentation data with user_id and value_cluster
    """
    logger.info("Loading segmentation data...")

    if not os.path.exists(cluster_path):
        raise FileNotFoundError(f"Segmentation file not found: {cluster_path}")

    seg_df = pd.read_csv(cluster_path)

    # Ensure user_id column exists
    if 'user_id' not in seg_df.columns:
        if 'id' in seg_df.columns:
            seg_df = seg_df.rename(columns={'id': 'user_id'})

    logger.info(f"  ✓ Loaded {len(seg_df)} users with segments")

    # Show segment distribution
    for cluster_id, name in SEGMENT_NAMES.items():
        count = (seg_df['value_cluster'] == cluster_id).sum()
        logger.info(f"    {name}: {count:,}")

    return seg_df[['user_id', 'value_cluster']]


# ============================================================================
# USER-LEVEL SHAP VISUALIZATIONS
# ============================================================================

def plot_user_shap_summary(user_shap_values, user_features, feature_names,
                            target_name='will_repurchase', save_path=PLOTS_PATH):
    """
    Create SHAP summary plot at user level.

    Args:
        user_shap_values: User-level SHAP values array
        user_features: User-level features array
        feature_names: List of feature names
        target_name: Name of target variable
        save_path: Path to save plot

    Returns:
        str: Path to saved plot
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Skipping plot.")
        return None

    logger.info("\nCreating user-level SHAP summary plot...")

    n_users = len(user_shap_values)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(user_shap_values, user_features, feature_names=feature_names,
                     show=False, max_display=20)

    plt.title(f'SHAP Feature Importance - User Level\n(n={n_users:,} users)',
             fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_shap_summary_user_level.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  ✓ Saved to: {file_path}")

    return file_path


def plot_user_shap_by_segment(user_shap_df, segmentation_df, feature_names,
                                target_name='will_repurchase', save_path=PLOTS_PATH):
    """
    Create per-segment SHAP plots at user level.

    Args:
        user_shap_df: DataFrame with user_id and SHAP values
        segmentation_df: DataFrame with user_id and value_cluster
        feature_names: List of feature names
        target_name: Name of target variable
        save_path: Path to save plots

    Returns:
        list: Paths to saved plots
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Skipping per-segment plots.")
        return []

    logger.info("\nCreating user-level SHAP plots by segment...")

    # Merge with segmentation
    user_data = user_shap_df.merge(segmentation_df, on='user_id', how='left')

    # Count matched users
    matched = user_data['value_cluster'].notna().sum()
    logger.info(f"  Matched {matched:,} / {len(user_data):,} users with segments")

    plot_paths = []

    for segment_id, segment_name in SEGMENT_NAMES.items():
        # Filter to segment
        segment_mask = user_data['value_cluster'] == segment_id
        segment_data = user_data[segment_mask]

        if len(segment_data) == 0:
            logger.info(f"  Skipping {segment_name} (no users)")
            continue

        n_users = len(segment_data)
        logger.info(f"  Creating SHAP summary for {segment_name} (n={n_users:,} users)")

        # Extract SHAP values and features
        segment_shap_values = segment_data[feature_names].values

        # Create SHAP summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            segment_shap_values,
            segment_shap_values,  # Use same for features (shows SHAP value distributions)
            feature_names=feature_names,
            show=False,
            max_display=20
        )

        plt.title(f'SHAP Feature Importance - {segment_name}\n(n={n_users:,} users)',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        os.makedirs(save_path, exist_ok=True)
        safe_name = segment_name.replace(' ', '_').replace('-', '_').lower()
        file_path = f"{save_path}/{target_name}_shap_summary_user_level_{safe_name}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        plot_paths.append(file_path)
        logger.info(f"    ✓ Saved to: {file_path}")

    logger.info(f"  ✓ Created {len(plot_paths)} user-level segment SHAP plots")

    return plot_paths


def plot_user_shap_bar(user_shap_df, feature_names, target_name='will_repurchase', save_path=PLOTS_PATH):
    """
    Plot user-level SHAP bar chart (mean absolute values).

    Args:
        user_shap_df: DataFrame with user SHAP values
        feature_names: List of feature names
        target_name: Name of target variable
        save_path: Path to save plot

    Returns:
        str: Path to saved plot
    """
    logger.info("Creating user-level SHAP bar plot...")

    # Extract SHAP values
    user_shap_values = user_shap_df[feature_names].values
    n_users = len(user_shap_values)

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(user_shap_values).mean(axis=0)

    # Sort and get top 20
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:20]
    top_features = [feature_names[i] for i in sorted_idx]
    top_values = mean_abs_shap[sorted_idx]

    plt.figure(figsize=(10, 10))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    plt.barh(range(len(top_features)), top_values[::-1], color=colors[::-1])
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.title(f'Top 20 Feature Importance (SHAP) - User Level\n(n={n_users:,} users)',
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_shap_bar_user_level.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  ✓ Saved to: {file_path}")

    return file_path


# ============================================================================
# MAIN USER SHAP ANALYSIS PIPELINE
# ============================================================================

def run_user_shap_analysis(order_shap_results, X_test, user_ids, test_df, segmentation_df=None,
                            target_name='will_repurchase', plots_path=PLOTS_PATH):
    """
    Main entry point - orchestrates user-level SHAP analysis.

    Args:
        order_shap_results: Results from order-level SHAP analysis (from repurchase_shap_analysis)
        X_test: Test features array (order-level)
        user_ids: User IDs for test set
        test_df: Test DataFrame with order_date
        segmentation_df: Optional segmentation DataFrame (if None, will load from CLUSTER_PATH)
        target_name: Name of target variable
        plots_path: Path to save visualizations

    Returns:
        dict: User-level SHAP analysis results
    """
    logger.info("="*80)
    logger.info("USER-LEVEL SHAP ANALYSIS")
    logger.info("="*80)
    logger.info("Aggregating order-level SHAP to user-level using most recent order per user")

    # Extract from order-level results
    order_shap_values = order_shap_results['shap_values']
    # Get ALL feature names from importance_df (not just top drivers)
    feature_names = order_shap_results['importance_df']['feature'].tolist()

    # Step 1: Aggregate to user level
    user_shap_df = aggregate_order_shap_to_users(
        order_shap_values, X_test, user_ids, feature_names, test_df
    )

    # Step 2: Load segmentation if not provided
    if segmentation_df is None:
        try:
            segmentation_df = load_segmentation()
        except FileNotFoundError as e:
            logger.warning(f"  ⚠️ Segmentation not available: {e}")
            segmentation_df = None

    # Step 3: Generate user-level visualizations
    plot_paths = {}

    # Overall user-level SHAP summary
    user_shap_values = user_shap_df[feature_names].values
    plot_paths['summary'] = plot_user_shap_summary(
        user_shap_values, user_shap_values, feature_names, target_name, plots_path
    )

    # User-level SHAP bar chart
    plot_paths['bar'] = plot_user_shap_bar(
        user_shap_df, feature_names, target_name, plots_path
    )

    # Per-segment user-level plots
    if segmentation_df is not None:
        segment_plot_paths = plot_user_shap_by_segment(
            user_shap_df, segmentation_df, feature_names, target_name, plots_path
        )
        plot_paths['per_segment'] = segment_plot_paths

    logger.info("\n" + "="*80)
    logger.info("✓ USER-LEVEL SHAP ANALYSIS COMPLETE")
    logger.info("="*80)

    return {
        'user_shap_df': user_shap_df,
        'user_shap_values': user_shap_values,
        'feature_names': feature_names,
        'plot_paths': plot_paths,
        'n_users': len(user_shap_df)
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Repurchase User Aggregation module loaded successfully.")
    print("Use run_user_shap_analysis() after order-level SHAP analysis.")
    print("Requires: order_shap_results, X_test, user_ids, test_df")
