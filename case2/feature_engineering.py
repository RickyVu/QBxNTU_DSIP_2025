# ---
# Feature Engineering Pipeline for Case 2: Propensity Prediction
# ---

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "../original_data"
CLUSTER_PATH = "../cluster01/sg_user.csv"
OBSERVATION_WINDOW_WEEKS = 12  # 12-week prediction window

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_all_datasets(data_path=DATA_PATH):
    """
    Load all required datasets from original_data folder.

    Returns:
        dict: Dictionary containing all datasets
    """
    logger.info("Loading datasets...")

    datasets = {}

    # Load main datasets
    datasets['users'] = pd.read_csv(f"{data_path}/users.csv")
    datasets['orders'] = pd.read_csv(f"{data_path}/orders.csv")
    datasets['subscriptions'] = pd.read_csv(f"{data_path}/subscriptions.csv")
    datasets['products'] = pd.read_csv(f"{data_path}/products.csv")
    datasets['preferences'] = pd.read_csv(f"{data_path}/preferences.csv")
    datasets['events'] = pd.read_csv(f"{data_path}/events.csv")
    datasets['voucher_applications'] = pd.read_csv(f"{data_path}/voucher_applications.csv")
    datasets['user_references'] = pd.read_csv(f"{data_path}/user_references.csv")

    logger.info(f"✓ Users: {len(datasets['users']):,} rows")
    logger.info(f"✓ Orders: {len(datasets['orders']):,} rows")
    logger.info(f"✓ Subscriptions: {len(datasets['subscriptions']):,} rows")
    logger.info(f"✓ Products: {len(datasets['products']):,} rows")
    logger.info(f"✓ Preferences: {len(datasets['preferences']):,} rows")
    logger.info(f"✓ Events: {len(datasets['events']):,} rows")
    logger.info(f"✓ Voucher Applications: {len(datasets['voucher_applications']):,} rows")
    logger.info(f"✓ User References: {len(datasets['user_references']):,} rows")

    return datasets


def load_case1_segmentation(cluster_path=CLUSTER_PATH):
    """
    Load Case 1 RFM segmentation results.
    
    WARNING: This function loads segmentation with DATA LEAKAGE.
    Use recalculate_rfm_segmentation() instead for proper temporal validation.

    Returns:
        pd.DataFrame: DataFrame with user_id and value_cluster columns
    """
    logger.info("Loading Case 1 segmentation...")

    cluster_df = pd.read_csv(cluster_path)

    # Keep only user_id and value_cluster
    segmentation = cluster_df[['user_id', 'value_cluster']].copy()

    logger.info(f"✓ Segmentation loaded: {len(segmentation):,} users")
    logger.info(f"  Cluster distribution:")
    for cluster in sorted(segmentation['value_cluster'].unique()):
        count = (segmentation['value_cluster'] == cluster).sum()
        pct = count / len(segmentation) * 100
        logger.info(f"    Cluster {cluster}: {count:,} ({pct:.1f}%)")

    return segmentation


def recalculate_rfm_segmentation(orders_df, temporal_params, n_clusters=4):
    """
    Recalculate RFM segmentation using only historical data to prevent data leakage.
    
    This function replicates the exact RFM calculation from main.py but uses only
    data before the feature cutoff date to ensure no future information leaks into
    the features.
    
    RFM Definition (from main.py):
    - Recency: (reference_date - last_order_date).days
    - Frequency: order_count / max((last_order_date - first_order_date).days, 1)
    - Monetary: sum of weighted_total_incl_tax_sgd with time-based weights
      * Orders ≤2020: weight 0.4
      * Orders ≤2021: weight 0.6
      * Orders ≤2022: weight 0.8
      * Orders ≥2023: weight 1.0
    
    Clustering:
    - Uses ['recency_scores', 'frequency_orders', 'monetary_total']
    - recency_scores = -recency_days (inverted so higher is better)
    - StandardScaler normalization
    - KMeans with n_clusters=4, random_state=42
    
    Args:
        orders_df: Cleaned orders DataFrame
        temporal_params: Dictionary from setup_temporal_validation()
        n_clusters: Number of clusters for segmentation (default: 4)
        
    Returns:
        pd.DataFrame: DataFrame with user_id and value_cluster columns
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    logger.info("="*80)
    logger.info("RECALCULATING RFM SEGMENTATION (NO DATA LEAKAGE)")
    logger.info("="*80)
    
    cutoff_date = temporal_params['feature_cutoff_date']
    
    # Use only historical orders (before cutoff)
    historical = orders_df[orders_df['date_placed'] < cutoff_date].copy()
    
    logger.info(f"Reference date (cutoff): {cutoff_date.date()}")
    logger.info(f"Historical orders: {len(historical):,}")
    
    # Apply time-based monetary weights (same as main.py)
    def arbitrary_date_weights(dates):
        year = dates.dt.year
        return np.select(
            [year <= 2020, year <= 2021, year <= 2022],
            [0.4, 0.6, 0.8],
            default=1.0,
        )
    
    historical['monetary_weight'] = arbitrary_date_weights(historical['date_placed'])
    historical['weighted_total_incl_tax'] = (
        historical['total_incl_tax'] * historical['monetary_weight']
    )
    
    # Calculate RFM features (exactly as in main.py)
    rfm_features = historical.groupby("user_id").agg(
        {
            "date_placed": [
                lambda x: (cutoff_date - x.max()).days,                      # Recency
                lambda x: len(x) / max((x.max() - x.min()).days, 1)          # Frequency
            ],
            "weighted_total_incl_tax": "sum"  # Monetary
        }
    ).reset_index()
    
    rfm_features.columns = ["user_id", "recency_days", "frequency_orders", "monetary_total"]
    
    logger.info(f"✓ RFM features calculated for {len(rfm_features):,} users")
    logger.info(f"\nRFM Statistics:")
    logger.info(f"  Recency (days):")
    logger.info(f"    Mean: {rfm_features['recency_days'].mean():.1f}")
    logger.info(f"    Median: {rfm_features['recency_days'].median():.1f}")
    logger.info(f"  Frequency (orders/day):")
    logger.info(f"    Mean: {rfm_features['frequency_orders'].mean():.4f}")
    logger.info(f"    Median: {rfm_features['frequency_orders'].median():.4f}")
    logger.info(f"  Monetary (weighted):")
    logger.info(f"    Mean: ${rfm_features['monetary_total'].mean():.2f}")
    logger.info(f"    Median: ${rfm_features['monetary_total'].median():.2f}")
    
    # Prepare data for clustering (exactly as in main.py)
    value_cols = ['recency_days', 'frequency_orders', 'monetary_total']
    value_data = rfm_features[value_cols].copy()
    
    # Invert recency so higher is better
    value_data['recency_scores'] = -value_data['recency_days']
    
    value_data_for_cluster = value_data[['recency_scores', 'frequency_orders', 'monetary_total']].copy()
    value_data_for_cluster = value_data_for_cluster.fillna(0)
    
    # Standardize the data
    scaler_value = StandardScaler()
    value_data_scaled = scaler_value.fit_transform(value_data_for_cluster)
    
    # Perform KMeans clustering (same parameters as main.py)
    kmeans_value = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_features['value_cluster'] = kmeans_value.fit_predict(value_data_scaled)
    
    logger.info(f"\n✓ KMeans clustering complete (k={n_clusters})")
    logger.info(f"\nCluster distribution:")
    for cluster in sorted(rfm_features['value_cluster'].unique()):
        count = (rfm_features['value_cluster'] == cluster).sum()
        pct = count / len(rfm_features) * 100
        logger.info(f"  Cluster {cluster}: {count:,} ({pct:.1f}%)")
    
    # Show cluster characteristics
    logger.info(f"\nCluster characteristics (mean values):")
    cluster_means = rfm_features.groupby('value_cluster')[value_cols].mean()
    for cluster in sorted(rfm_features['value_cluster'].unique()):
        logger.info(f"  Cluster {cluster}:")
        logger.info(f"    Recency: {cluster_means.loc[cluster, 'recency_days']:.1f} days")
        logger.info(f"    Frequency: {cluster_means.loc[cluster, 'frequency_orders']:.4f} orders/day")
        logger.info(f"    Monetary: ${cluster_means.loc[cluster, 'monetary_total']:.2f}")
    
    logger.info("="*80)
    
    return rfm_features[['user_id', 'value_cluster']]


# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================

def clean_orders(orders_df):
    """
    Clean orders dataset following the pattern from case2/main.py.

    - Keep only completed orders (Shipped/Complete status)
    - Remove orders with invalid totals
    - Parse and fix datetime columns
    - Handle timezone issues

    Args:
        orders_df: Raw orders DataFrame

    Returns:
        pd.DataFrame: Cleaned orders DataFrame
    """
    logger.info("Cleaning orders data...")

    orders = orders_df.copy()

    # Filter to completed orders only
    orders = orders[orders["status"].isin(["Shipped", "Complete"])]
    orders = orders[orders["total_incl_tax"] > 0]

    # Fix datetime columns - handle inconsistent formats and timezones
    orders["date_placed"] = pd.to_datetime(orders["date_placed"], errors="coerce", utc=True).dt.tz_localize(None)
    orders["shipping_date"] = pd.to_datetime(orders["shipping_date"], errors="coerce", utc=True).dt.tz_localize(None)

    # Drop rows with missing critical data
    orders = orders.dropna(subset=["date_placed", "user_id"])

    logger.info(f"✓ Cleaned orders: {len(orders):,} rows")
    logger.info(f"  Date range: {orders['date_placed'].min()} to {orders['date_placed'].max()}")

    return orders


def clean_subscriptions(subscriptions_df):
    """
    Clean subscriptions dataset.

    Args:
        subscriptions_df: Raw subscriptions DataFrame

    Returns:
        pd.DataFrame: Cleaned subscriptions DataFrame
    """
    logger.info("Cleaning subscriptions data...")

    subs = subscriptions_df.copy()

    # Drop rows without user_id
    subs = subs.dropna(subset=["user_id"])

    # Parse datetime columns
    date_cols = ["last_order", "next_order", "created", "updated"]
    for col in date_cols:
        if col in subs.columns:
            subs[col] = pd.to_datetime(subs[col], errors="coerce", utc=True).dt.tz_localize(None)

    logger.info(f"✓ Cleaned subscriptions: {len(subs):,} rows")

    return subs


def clean_preferences(preferences_df):
    """
    Clean preferences dataset.

    Args:
        preferences_df: Raw preferences DataFrame

    Returns:
        pd.DataFrame: Cleaned preferences DataFrame
    """
    logger.info("Cleaning preferences data...")

    prefs = preferences_df.copy()

    # Drop duplicates and missing user_ids
    prefs = prefs.dropna(subset=["user_id"]).drop_duplicates(subset='user_id')

    # Parse datetime
    prefs["created"] = pd.to_datetime(prefs["created"], errors="coerce", utc=True).dt.tz_localize(None)

    logger.info(f"✓ Cleaned preferences: {len(prefs):,} rows")

    return prefs


def clean_events(events_df):
    """
    Clean events dataset.

    Args:
        events_df: Raw events DataFrame

    Returns:
        pd.DataFrame: Cleaned events DataFrame
    """
    logger.info("Cleaning events data...")

    events = events_df.copy()

    # Drop rows without user_id or timestamp
    events = events.dropna(subset=["user_id", "timestamp"])

    # Parse datetime
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)

    logger.info(f"✓ Cleaned events: {len(events):,} rows")

    return events


def clean_vouchers(voucher_applications_df):
    """
    Clean voucher applications dataset.

    Args:
        voucher_applications_df: Raw voucher applications DataFrame

    Returns:
        pd.DataFrame: Cleaned voucher applications DataFrame
    """
    logger.info("Cleaning voucher applications data...")

    vouchers = voucher_applications_df.copy()

    # Drop rows without user_id
    vouchers = vouchers.dropna(subset=["user_id"])

    # Parse datetime
    vouchers["created"] = pd.to_datetime(vouchers["created"], errors="coerce", utc=True).dt.tz_localize(None)

    logger.info(f"✓ Cleaned voucher applications: {len(vouchers):,} rows")

    return vouchers


def clean_users(users_df):
    """
    Clean users dataset.

    Args:
        users_df: Raw users DataFrame

    Returns:
        pd.DataFrame: Cleaned users DataFrame
    """
    logger.info("Cleaning users data...")

    users = users_df.copy()

    # Parse datetime
    if 'date_signed_up' in users.columns:
        users['date_signed_up'] = pd.to_datetime(users['date_signed_up'], errors="coerce")

    logger.info(f"✓ Cleaned users: {len(users):,} rows")

    return users


def clean_all_datasets(datasets):
    """
    Clean all datasets.

    Args:
        datasets: Dictionary of raw datasets

    Returns:
        dict: Dictionary of cleaned datasets
    """
    logger.info("="*80)
    logger.info("DATA CLEANING")
    logger.info("="*80)

    cleaned = {
        'orders': clean_orders(datasets['orders']),
        'subscriptions': clean_subscriptions(datasets['subscriptions']),
        'preferences': clean_preferences(datasets['preferences']),
        'events': clean_events(datasets['events']),
        'voucher_applications': clean_vouchers(datasets['voucher_applications']),
        'users': clean_users(datasets['users']),
        'products': datasets['products'].copy(),  # Products don't need special cleaning
        'user_references': datasets['user_references'].copy()  # User references don't need special cleaning
    }

    logger.info("✓ All datasets cleaned successfully")

    return cleaned


# ============================================================================
# TEMPORAL VALIDATION SETUP
# ============================================================================

def setup_temporal_validation(orders_df, observation_weeks=OBSERVATION_WINDOW_WEEKS):
    """
    Set up temporal validation windows to prevent data leakage.

    Args:
        orders_df: Cleaned orders DataFrame
        observation_weeks: Number of weeks for prediction window (default: 12)

    Returns:
        dict: Dictionary containing temporal validation parameters
    """
    logger.info("="*80)
    logger.info("TEMPORAL VALIDATION SETUP")
    logger.info("="*80)

    # Get max date from orders
    max_date = orders_df['date_placed'].max()

    # Define temporal windows
    observation_days = observation_weeks * 7
    feature_cutoff_date = max_date - pd.Timedelta(days=observation_days)
    prediction_start = feature_cutoff_date
    prediction_end = max_date

    temporal_params = {
        'max_date': max_date,
        'observation_weeks': observation_weeks,
        'observation_days': observation_days,
        'feature_cutoff_date': feature_cutoff_date,
        'prediction_start': prediction_start,
        'prediction_end': prediction_end
    }

    logger.info(f"Dataset max date: {max_date.date()}")
    logger.info(f"Observation window: {observation_weeks} weeks ({observation_days} days)")
    logger.info(f"")
    logger.info(f"Feature calculation period: All data up to {feature_cutoff_date.date()}")
    logger.info(f"Prediction window: {prediction_start.date()} to {prediction_end.date()}")
    logger.info(f"")
    logger.info(f"Timeline:")
    logger.info(f"|--- Feature Period (historical) ---|--- Prediction Window ({observation_weeks} weeks) ---|")
    logger.info(f"                              {feature_cutoff_date.date()}                            {prediction_end.date()}")
    logger.info(f"                       (features calculated here)                  (target measured here)")

    return temporal_params


# ============================================================================
# TARGET VARIABLE CREATION
# ============================================================================

def create_loyalty_target(orders_df, temporal_params):
    """
    Create target variable for Loyalty/Purchase Propensity Model.

    Target: will_purchase_next_12_weeks (binary)
    - 1 = Has ≥1 completed order in prediction window
    - 0 = No orders in prediction window

    Args:
        orders_df: Cleaned orders DataFrame
        temporal_params: Dictionary from setup_temporal_validation()

    Returns:
        pd.DataFrame: DataFrame with user_id and will_purchase columns
    """
    logger.info("="*80)
    logger.info("LOYALTY MODEL - TARGET VARIABLE CREATION")
    logger.info("="*80)

    feature_cutoff_date = temporal_params['feature_cutoff_date']
    prediction_start = temporal_params['prediction_start']
    prediction_end = temporal_params['prediction_end']

    # Get eligible users (had at least one order BEFORE cutoff)
    eligible_users = orders_df[orders_df['date_placed'] < feature_cutoff_date]['user_id'].unique()

    # Get users who made orders DURING prediction window
    orders_in_window = orders_df[
        (orders_df['date_placed'] >= prediction_start) &
        (orders_df['date_placed'] < prediction_end)
    ]
    users_with_orders_in_window = orders_in_window['user_id'].unique()

    # Create target DataFrame
    target_df = pd.DataFrame({'user_id': eligible_users})
    target_df['will_purchase'] = target_df['user_id'].isin(users_with_orders_in_window).astype(int)

    # Calculate statistics
    n_eligible = len(eligible_users)
    n_will_purchase = target_df['will_purchase'].sum()
    n_wont_purchase = len(target_df) - n_will_purchase
    purchase_rate = target_df['will_purchase'].mean()

    logger.info(f"Eligible users (had orders before {feature_cutoff_date.date()}): {n_eligible:,}")
    logger.info(f"")
    logger.info(f"Will Purchase (target=1): {n_will_purchase:,} ({purchase_rate:.2%})")
    logger.info(f"Won't Purchase (target=0): {n_wont_purchase:,} ({1-purchase_rate:.2%})")
    logger.info(f"")
    logger.info(f"Class balance: {'Balanced' if 0.3 <= purchase_rate <= 0.7 else 'Imbalanced'}")
    if purchase_rate < 0.3:
        logger.warning(f"⚠️  Class imbalance detected: Only {purchase_rate:.1%} positive class")
        logger.warning(f"   Recommendation: Use class_weight='balanced' and focus on PR-AUC")

    return target_df


def create_high_value_target(orders_df, temporal_params):
    """
    Create target variable for High-Value Propensity Model.

    Target: will_be_high_value_next_12_weeks (binary)
    - Subset: Only customers who purchased in prediction window
    - 1 = Total spend > median spend in window
    - 0 = Total spend ≤ median spend in window

    Args:
        orders_df: Cleaned orders DataFrame
        temporal_params: Dictionary from setup_temporal_validation()

    Returns:
        pd.DataFrame: DataFrame with user_id and will_be_high_value columns
    """
    logger.info("="*80)
    logger.info("HIGH-VALUE MODEL - TARGET VARIABLE CREATION")
    logger.info("="*80)

    feature_cutoff_date = temporal_params['feature_cutoff_date']
    prediction_start = temporal_params['prediction_start']
    prediction_end = temporal_params['prediction_end']

    # Get users who purchased in prediction window
    orders_in_window = orders_df[
        (orders_df['date_placed'] >= prediction_start) &
        (orders_df['date_placed'] < prediction_end)
    ]

    # Calculate total spend per user in window
    user_spend = orders_in_window.groupby('user_id')['total_incl_tax'].sum().reset_index()
    user_spend.columns = ['user_id', 'total_spend_in_window']

    # Calculate median spend
    median_spend = user_spend['total_spend_in_window'].median()

    # Create binary target
    user_spend['will_be_high_value'] = (user_spend['total_spend_in_window'] > median_spend).astype(int)

    # Calculate statistics
    n_purchased = len(user_spend)
    n_high_value = user_spend['will_be_high_value'].sum()
    n_low_value = len(user_spend) - n_high_value
    high_value_rate = user_spend['will_be_high_value'].mean()

    logger.info(f"Eligible users (purchased in prediction window): {n_purchased:,}")
    logger.info(f"Median spend in window: ${median_spend:.2f}")
    logger.info(f"")
    logger.info(f"High-Value (above median, target=1): {n_high_value:,} ({high_value_rate:.2%})")
    logger.info(f"Low-Value (below median, target=0): {n_low_value:,} ({1-high_value_rate:.2%})")
    logger.info(f"")
    logger.info(f"Spend distribution:")
    logger.info(f"  Min: ${user_spend['total_spend_in_window'].min():.2f}")
    logger.info(f"  25th percentile: ${user_spend['total_spend_in_window'].quantile(0.25):.2f}")
    logger.info(f"  Median: ${median_spend:.2f}")
    logger.info(f"  75th percentile: ${user_spend['total_spend_in_window'].quantile(0.75):.2f}")
    logger.info(f"  Max: ${user_spend['total_spend_in_window'].max():.2f}")

    return user_spend[['user_id', 'will_be_high_value']]


# ============================================================================
# VALIDATION SUMMARY
# ============================================================================

def print_validation_summary(temporal_params, loyalty_target, high_value_target):
    """
    Print consolidated validation summary with date ranges and target rates.
    
    Args:
        temporal_params: Dictionary from setup_temporal_validation()
        loyalty_target: DataFrame with user_id and will_purchase columns
        high_value_target: DataFrame with user_id and will_be_high_value columns
    """
    logger.info("="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    
    feature_cutoff_date = temporal_params['feature_cutoff_date']
    prediction_start = temporal_params['prediction_start']
    prediction_end = temporal_params['prediction_end']
    observation_weeks = temporal_params['observation_weeks']
    
    # Date ranges
    logger.info("\n📅 DATE RANGES:")
    logger.info(f"  Feature calculation period: All data up to {feature_cutoff_date.date()}")
    logger.info(f"  Prediction window: {prediction_start.date()} to {prediction_end.date()} ({observation_weeks} weeks)")
    
    # Loyalty Model Summary
    n_eligible = len(loyalty_target)
    n_will_purchase = loyalty_target['will_purchase'].sum()
    n_wont_purchase = len(loyalty_target) - n_will_purchase
    purchase_rate = loyalty_target['will_purchase'].mean()
    
    logger.info("\n🎯 LOYALTY MODEL (Purchase Propensity):")
    logger.info(f"  Eligible users: {n_eligible:,}")
    logger.info(f"  Will Purchase (target=1): {n_will_purchase:,} ({purchase_rate:.2%})")
    logger.info(f"  Won't Purchase (target=0): {n_wont_purchase:,} ({1-purchase_rate:.2%})")
    logger.info(f"  Class balance: {'✅ Balanced' if 0.3 <= purchase_rate <= 0.7 else '⚠️  Imbalanced'}")
    
    # High-Value Model Summary
    n_purchased = len(high_value_target)
    n_high_value = high_value_target['will_be_high_value'].sum()
    n_low_value = len(high_value_target) - n_high_value
    high_value_rate = high_value_target['will_be_high_value'].mean()
    
    logger.info("\n💰 HIGH-VALUE MODEL (Upsell Propensity):")
    logger.info(f"  Eligible users (purchased in window): {n_purchased:,}")
    logger.info(f"  High-Value (above median, target=1): {n_high_value:,} ({high_value_rate:.2%})")
    logger.info(f"  Low-Value (below median, target=0): {n_low_value:,} ({1-high_value_rate:.2%})")
    
    logger.info("\n" + "="*80)


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def run_data_loading_and_validation():
    """
    Run the complete data loading, cleaning, and temporal validation pipeline.

    Returns:
        tuple: (cleaned_datasets, segmentation, temporal_params, loyalty_target, high_value_target)
    """
    logger.info("="*80)
    logger.info("CASE 2: PROPENSITY PREDICTION - DATA LOADING & VALIDATION")
    logger.info("="*80)

    # Load raw datasets
    datasets = load_all_datasets()

    # Clean all datasets
    cleaned = clean_all_datasets(datasets)

    # Setup temporal validation
    temporal_params = setup_temporal_validation(cleaned['orders'])

    # Recalculate RFM segmentation with proper temporal validation (NO DATA LEAKAGE)
    segmentation = recalculate_rfm_segmentation(cleaned['orders'], temporal_params)

    # Create target variables
    loyalty_target = create_loyalty_target(cleaned['orders'], temporal_params)
    high_value_target = create_high_value_target(cleaned['orders'], temporal_params)
    
    # Print consolidated validation summary
    print_validation_summary(temporal_params, loyalty_target, high_value_target)

    logger.info("="*80)
    logger.info("✓ DATA LOADING & VALIDATION COMPLETE")
    logger.info("="*80)

    return cleaned, segmentation, temporal_params, loyalty_target, high_value_target


# ============================================================================
# FEATURE ENGINEERING - CATEGORY 1: PURCHASE HISTORY & RECENCY
# ============================================================================

def create_purchase_history_features(orders_df, temporal_params):
    """
    Create purchase history and recency features (Category 1).

    Features include:
    - Rolling windows (30/60/90 days)
    - Recency metrics
    - Frequency patterns
    - Trend indicators

    Args:
        orders_df: Cleaned orders DataFrame
        temporal_params: Dictionary from setup_temporal_validation()

    Returns:
        pd.DataFrame: Purchase history features by user_id
    """
    logger.info("Creating Category 1: Purchase History & Recency Features...")

    cutoff_date = temporal_params['feature_cutoff_date']

    # Filter to historical orders only (NO DATA LEAKAGE)
    historical = orders_df[orders_df['date_placed'] < cutoff_date].copy()

    features = []

    # === ROLLING WINDOW FEATURES ===
    for window_days in [30, 60, 90]:
        window_start = cutoff_date - pd.Timedelta(days=window_days)
        window_orders = historical[historical['date_placed'] >= window_start]

        window_features = window_orders.groupby('user_id').agg({
            'id': 'count',
            'total_incl_tax': ['sum', 'mean']
        }).reset_index()

        window_features.columns = [
            'user_id',
            f'total_orders_last_{window_days}d',
            f'total_spend_last_{window_days}d',
            f'avg_order_value_last_{window_days}d'
        ]

        features.append(window_features)

    # Merge rolling window features
    purchase_features = features[0]
    for feat_df in features[1:]:
        purchase_features = purchase_features.merge(feat_df, on='user_id', how='outer')

    # === RECENCY FEATURES ===
    recency_df = historical.groupby('user_id').agg({
        'date_placed': ['min', 'max']
    }).reset_index()
    recency_df.columns = ['user_id', 'first_order_date', 'last_order_date']

    recency_df['days_since_last_order'] = (cutoff_date - recency_df['last_order_date']).dt.days
    recency_df['days_since_first_order'] = (cutoff_date - recency_df['first_order_date']).dt.days
    recency_df['customer_tenure_days'] = (recency_df['last_order_date'] - recency_df['first_order_date']).dt.days

    # Recency score (binned)
    recency_df['recency_score'] = pd.cut(
        recency_df['days_since_last_order'],
        bins=[-1, 30, 60, 90, 180, np.inf],
        labels=[5, 4, 3, 2, 1]
    ).astype(float)

    purchase_features = purchase_features.merge(recency_df[['user_id', 'days_since_last_order',
                                                             'days_since_first_order', 'customer_tenure_days',
                                                             'recency_score']], on='user_id', how='left')

    # === FREQUENCY PATTERNS ===
    frequency_df = historical.groupby('user_id').agg({
        'id': 'count'
    }).reset_index()
    frequency_df.columns = ['user_id', 'total_lifetime_orders']

    purchase_features = purchase_features.merge(frequency_df, on='user_id', how='left')

    # Order frequency (orders per day)
    purchase_features['order_frequency'] = (
        purchase_features['total_lifetime_orders'] /
        (purchase_features['customer_tenure_days'] + 1)
    )

    # Average days between orders
    def calc_avg_days_between_orders(group):
        if len(group) < 2:
            return np.nan
        intervals = group.sort_values('date_placed')['date_placed'].diff().dt.days
        return intervals.mean()

    avg_intervals = historical.groupby('user_id').apply(calc_avg_days_between_orders).reset_index()
    avg_intervals.columns = ['user_id', 'avg_days_between_orders']
    purchase_features = purchase_features.merge(avg_intervals, on='user_id', how='left')

    # Order consistency (std of inter-order intervals)
    def calc_order_consistency(group):
        if len(group) < 2:
            return 0
        intervals = group.sort_values('date_placed')['date_placed'].diff().dt.days
        return intervals.std()

    order_std = historical.groupby('user_id').apply(calc_order_consistency).reset_index()
    order_std.columns = ['user_id', 'order_consistency']
    purchase_features = purchase_features.merge(order_std, on='user_id', how='left')

    # === TREND FEATURES ===
    # Order frequency trend (first half vs second half)
    def calc_order_trend(group):
        if len(group) < 4:
            return 0
        orders_sorted = group.sort_values('date_placed')
        midpoint = len(orders_sorted) // 2

        first_half = orders_sorted.iloc[:midpoint]
        second_half = orders_sorted.iloc[midpoint:]

        first_days = (first_half['date_placed'].max() - first_half['date_placed'].min()).days + 1
        second_days = (second_half['date_placed'].max() - second_half['date_placed'].min()).days + 1

        first_freq = len(first_half) / first_days if first_days > 0 else 0
        second_freq = len(second_half) / second_days if second_days > 0 else 0

        return second_freq - first_freq

    trend = historical.groupby('user_id').apply(calc_order_trend).reset_index()
    trend.columns = ['user_id', 'order_frequency_trend']
    purchase_features = purchase_features.merge(trend, on='user_id', how='left')

    # Spend trend (recent 90d vs previous 90d)
    recent_90d_start = cutoff_date - pd.Timedelta(days=90)
    prev_90d_start = cutoff_date - pd.Timedelta(days=180)
    prev_90d_end = recent_90d_start

    recent_spend = historical[historical['date_placed'] >= recent_90d_start].groupby('user_id')['total_incl_tax'].sum()
    prev_spend = historical[(historical['date_placed'] >= prev_90d_start) &
                           (historical['date_placed'] < prev_90d_end)].groupby('user_id')['total_incl_tax'].sum()

    spend_trend_df = pd.DataFrame({
        'user_id': recent_spend.index,
        'spend_trend': recent_spend.values - prev_spend.reindex(recent_spend.index, fill_value=0).values
    })
    purchase_features = purchase_features.merge(spend_trend_df, on='user_id', how='left')

    # Order acceleration (last 30d vs last 60d)
    orders_last_30d = purchase_features['total_orders_last_30d'].fillna(0)
    orders_last_60d = purchase_features['total_orders_last_60d'].fillna(0)
    purchase_features['order_acceleration'] = orders_last_30d - (orders_last_60d - orders_last_30d)

    logger.info(f"  ✓ Created {len(purchase_features.columns)-1} purchase history features for {len(purchase_features):,} users")

    return purchase_features


# ============================================================================
# FEATURE ENGINEERING - CATEGORY 2: MONETARY VALUE
# ============================================================================

def create_monetary_features(orders_df, temporal_params):
    """
    Create monetary value features (Category 2).

    Args:
        orders_df: Cleaned orders DataFrame
        temporal_params: Dictionary from setup_temporal_validation()

    Returns:
        pd.DataFrame: Monetary features by user_id
    """
    logger.info("Creating Category 2: Monetary Value Features...")

    cutoff_date = temporal_params['feature_cutoff_date']
    historical = orders_df[orders_df['date_placed'] < cutoff_date].copy()

    # Lifetime monetary features
    monetary_features = historical.groupby('user_id').agg({
        'total_incl_tax': ['sum', 'mean', 'max', 'min', 'std', 'count']
    }).reset_index()

    monetary_features.columns = [
        'user_id', 'total_lifetime_revenue', 'avg_order_value_lifetime',
        'max_order_value', 'min_order_value', 'std_order_value', 'order_count_for_monetary'
    ]

    # Spend volatility (coefficient of variation)
    monetary_features['spend_volatility'] = (
        monetary_features['std_order_value'] /
        (monetary_features['avg_order_value_lifetime'] + 0.01)
    )

    # Revenue percentile
    monetary_features['revenue_percentile'] = monetary_features['total_lifetime_revenue'].rank(pct=True)

    # Is high value customer (>75th percentile)
    p75 = monetary_features['total_lifetime_revenue'].quantile(0.75)
    monetary_features['is_high_value_customer'] = (monetary_features['total_lifetime_revenue'] > p75).astype(int)

    # Revenue growth rate - will be calculated later when we have spend_trend from Category 1
    # Placeholder column that will be filled during merge
    monetary_features['revenue_growth_rate'] = np.nan

    monetary_features = monetary_features.drop(columns=['order_count_for_monetary'])

    logger.info(f"  ✓ Created {len(monetary_features.columns)-1} monetary features for {len(monetary_features):,} users")

    return monetary_features


# ============================================================================
# FEATURE ENGINEERING - CATEGORY 3: PRODUCT ENGAGEMENT
# ============================================================================

def create_product_engagement_features(orders_df, products_df, temporal_params):
    """
    Create product engagement features (Category 3).

    Parse order_items JSON to extract product information.

    Args:
        orders_df: Cleaned orders DataFrame
        products_df: Products DataFrame
        temporal_params: Dictionary from setup_temporal_validation()

    Returns:
        pd.DataFrame: Product engagement features by user_id
    """
    logger.info("Creating Category 3: Product Engagement Features...")

    cutoff_date = temporal_params['feature_cutoff_date']
    historical = orders_df[orders_df['date_placed'] < cutoff_date].copy()

    # Parse order_items JSON
    all_user_products = []

    for idx, row in historical.iterrows():
        try:
            items = json.loads(row['order_items'].replace("'", '"'))
            for item in items:
                all_user_products.append({
                    'user_id': row['user_id'],
                    'product_id': item.get('product_id'),
                    'order_id': row['id'],
                    'date_placed': row['date_placed']
                })
        except:
            continue

    if len(all_user_products) == 0:
        logger.warning("  ⚠️  No order_items could be parsed")
        return pd.DataFrame({'user_id': historical['user_id'].unique()})

    product_df = pd.DataFrame(all_user_products)

    # Convert product_id to int (handle type mismatch)
    product_df['product_id'] = pd.to_numeric(product_df['product_id'], errors='coerce')
    product_df = product_df.dropna(subset=['product_id'])
    product_df['product_id'] = product_df['product_id'].astype(int)

    # Merge with products to get product_type and product_attributes
    product_df = product_df.merge(
        products_df[['product_id', 'product_type', 'product_attributes']],
        on='product_id',
        how='left'
    )

    # === PRODUCT DIVERSITY ===
    product_features = product_df.groupby('user_id').agg({
        'product_id': ['nunique', 'count']
    }).reset_index()
    product_features.columns = ['user_id', 'unique_products_ordered_lifetime', 'total_items']

    # Unique products in last 90 days
    recent_90d = cutoff_date - pd.Timedelta(days=90)
    recent_products = product_df[product_df['date_placed'] >= recent_90d].groupby('user_id')['product_id'].nunique().reset_index()
    recent_products.columns = ['user_id', 'unique_products_last_90d']
    product_features = product_features.merge(recent_products, on='user_id', how='left')

    # Product diversity score (entropy - simplified as normalized unique count)
    product_features['product_diversity_score'] = (
        product_features['unique_products_ordered_lifetime'] /
        (product_features['total_items'] + 1)
    )

    # === PRODUCT CATEGORY PREFERENCES ===
    # Count orders by product type
    category_counts = product_df.groupby(['user_id', 'product_type'])['order_id'].nunique().unstack(fill_value=0)

    # Calculate percentages
    total_orders_by_user = category_counts.sum(axis=1)
    category_pcts = category_counts.div(total_orders_by_user, axis=0) * 100

    # Rename columns to match plan
    category_mapping = {
        'Tea Pod': 'tea_pod_pct',
        'Tea Bag': 'tea_bag_pct',
        'Merchandise': 'merchandise_pct',
        'Bundle': 'bundle_pct'
    }

    for old_name, new_name in category_mapping.items():
        if old_name in category_pcts.columns:
            category_pcts = category_pcts.rename(columns={old_name: new_name})
        else:
            category_pcts[new_name] = 0

    # Keep only the mapped columns
    category_pcts = category_pcts[[col for col in category_mapping.values() if col in category_pcts.columns]]
    category_pcts = category_pcts.reset_index()

    product_features = product_features.merge(category_pcts, on='user_id', how='left')
    
    # === PREMIUM PRODUCT PERCENTAGE ===
    # Identify premium products based on product_attributes
    # Premium indicators: keywords like "premium", "luxury", "exclusive", "limited", "special"
    def is_premium_product(attributes_str):
        """Check if product is premium based on product_attributes"""
        if pd.isna(attributes_str) or attributes_str == '{}' or attributes_str == '':
            return False
        attributes_lower = str(attributes_str).lower()
        premium_keywords = ['premium', 'luxury', 'exclusive', 'limited', 'special', 'artisan', 'handcrafted']
        return any(keyword in attributes_lower for keyword in premium_keywords)
    
    product_df['is_premium'] = product_df['product_attributes'].apply(is_premium_product)
    
    # Calculate premium product percentage per user
    premium_counts = product_df.groupby('user_id').agg({
        'is_premium': 'sum',
        'order_id': 'nunique'  # Count unique orders
    }).reset_index()
    premium_counts.columns = ['user_id', 'premium_product_orders', 'total_orders']
    
    # Calculate percentage
    premium_counts['premium_product_pct'] = (
        premium_counts['premium_product_orders'] / premium_counts['total_orders'].clip(lower=1)
    ) * 100
    
    product_features = product_features.merge(
        premium_counts[['user_id', 'premium_product_pct']],
        on='user_id',
        how='left'
    )

    # Avg products per order
    product_features['avg_products_per_order'] = (
        product_features['total_items'] /
        historical.groupby('user_id')['id'].count().reindex(product_features['user_id'], fill_value=1).values
    )

    # Product exploration rate (simplified - new products / orders)
    # This would require tracking first-time purchases, approximating with diversity score for now
    product_features['product_exploration_rate'] = product_features['product_diversity_score']

    logger.info(f"  ✓ Created {len(product_features.columns)-1} product engagement features for {len(product_features):,} users")

    return product_features


# ============================================================================
# FEATURE ENGINEERING - CATEGORY 4: TEMPORAL & SEASONALITY
# ============================================================================

def create_temporal_features(orders_df, temporal_params):
    """
    Create temporal and seasonality features (Category 4).

    Args:
        orders_df: Cleaned orders DataFrame
        temporal_params: Dictionary from setup_temporal_validation()

    Returns:
        pd.DataFrame: Temporal features by user_id
    """
    logger.info("Creating Category 4: Temporal & Seasonality Features...")

    cutoff_date = temporal_params['feature_cutoff_date']
    historical = orders_df[orders_df['date_placed'] < cutoff_date].copy()

    # Get last order per user
    last_orders = historical.sort_values('date_placed').groupby('user_id').last().reset_index()

    temporal_features = pd.DataFrame({'user_id': last_orders['user_id']})

    # === CALENDAR PATTERNS (from last order) ===
    temporal_features['last_order_day_of_week'] = last_orders['date_placed'].dt.dayofweek
    temporal_features['last_order_month'] = last_orders['date_placed'].dt.month
    temporal_features['last_order_is_weekend'] = (last_orders['date_placed'].dt.dayofweek >= 5).astype(int)
    temporal_features['last_order_quarter'] = last_orders['date_placed'].dt.quarter

    # === CYCLICAL ENCODING ===
    temporal_features['day_of_week_sin'] = np.sin(2 * np.pi * temporal_features['last_order_day_of_week'] / 7)
    temporal_features['day_of_week_cos'] = np.cos(2 * np.pi * temporal_features['last_order_day_of_week'] / 7)
    temporal_features['month_sin'] = np.sin(2 * np.pi * temporal_features['last_order_month'] / 12)
    temporal_features['month_cos'] = np.cos(2 * np.pi * temporal_features['last_order_month'] / 12)

    # === ORDER TIME PATTERNS ===
    # Count weekend vs weekday orders
    historical['is_weekend'] = historical['date_placed'].dt.dayofweek >= 5
    weekend_counts = historical.groupby('user_id')['is_weekend'].agg(['sum', 'count']).reset_index()
    weekend_counts.columns = ['user_id', 'orders_on_weekends', 'total_orders']
    weekend_counts['orders_on_weekdays'] = weekend_counts['total_orders'] - weekend_counts['orders_on_weekends']
    weekend_counts['weekend_order_pct'] = (weekend_counts['orders_on_weekends'] / weekend_counts['total_orders']) * 100

    temporal_features = temporal_features.merge(
        weekend_counts[['user_id', 'orders_on_weekends', 'orders_on_weekdays', 'weekend_order_pct']],
        on='user_id',
        how='left'
    )

    # Most active month
    month_counts = historical.groupby(['user_id', historical['date_placed'].dt.month])['id'].count().reset_index()
    most_active = month_counts.loc[month_counts.groupby('user_id')['id'].idxmax()]
    most_active.columns = ['user_id', 'most_active_month', 'orders_in_most_active_month']
    temporal_features = temporal_features.merge(most_active[['user_id', 'most_active_month']], on='user_id', how='left')

    # Month diversity
    month_diversity = historical.groupby('user_id')['date_placed'].apply(lambda x: x.dt.month.nunique()).reset_index()
    month_diversity.columns = ['user_id', 'month_diversity']
    temporal_features = temporal_features.merge(month_diversity, on='user_id', how='left')

    # Holiday season orders (Nov-Dec)
    historical['is_holiday'] = historical['date_placed'].dt.month.isin([11, 12])
    holiday_orders = historical.groupby('user_id')['is_holiday'].sum().reset_index()
    holiday_orders.columns = ['user_id', 'orders_in_holiday_season']
    temporal_features = temporal_features.merge(holiday_orders, on='user_id', how='left')

    logger.info(f"  ✓ Created {len(temporal_features.columns)-1} temporal features for {len(temporal_features):,} users")

    return temporal_features


# ============================================================================
# FEATURE ENGINEERING - CATEGORY 5: SUBSCRIPTION
# ============================================================================

def create_subscription_features(subscriptions_df, orders_df, temporal_params):
    """
    Create subscription features (Category 5).

    Args:
        subscriptions_df: Cleaned subscriptions DataFrame
        orders_df: Cleaned orders DataFrame
        temporal_params: Dictionary from setup_temporal_validation()

    Returns:
        pd.DataFrame: Subscription features by user_id
    """
    logger.info("Creating Category 5: Subscription Features...")

    cutoff_date = temporal_params['feature_cutoff_date']
    historical_subs = subscriptions_df[subscriptions_df['created'] < cutoff_date].copy()

    # === SUBSCRIPTION STATUS ===
    # Active subscriptions (status = 'Active' or similar)
    if 'status' in historical_subs.columns:
        active_subs = historical_subs[historical_subs['status'] == 'Active'].groupby('user_id').size().reset_index()
        active_subs.columns = ['user_id', 'active_subscriptions_count']
    else:
        active_subs = pd.DataFrame(columns=['user_id', 'active_subscriptions_count'])

    # Total subscriptions
    total_subs = historical_subs.groupby('user_id').size().reset_index()
    total_subs.columns = ['user_id', 'total_subscriptions_lifetime']

    subscription_features = total_subs.merge(active_subs, on='user_id', how='left')
    subscription_features['active_subscriptions_count'] = subscription_features['active_subscriptions_count'].fillna(0)

    # Has active subscription
    subscription_features['has_active_subscription'] = (subscription_features['active_subscriptions_count'] > 0).astype(int)

    # Subscription active ratio
    subscription_features['subscription_active_ratio'] = (
        subscription_features['active_subscriptions_count'] /
        subscription_features['total_subscriptions_lifetime']
    )

    # === SUBSCRIPTION BEHAVIOR ===
    # Average interval
    interval_features = historical_subs.groupby('user_id')['interval'].agg(['mean', 'std']).reset_index()
    interval_features.columns = ['user_id', 'avg_subscription_interval', 'subscription_interval_std']
    subscription_features = subscription_features.merge(interval_features, on='user_id', how='left')

    # Prefers surprise box
    if 'surprise' in historical_subs.columns:
        surprise_count = historical_subs[historical_subs['surprise'] == 'Yes'].groupby('user_id').size().reset_index()
        surprise_count.columns = ['user_id', 'surprise_subscriptions']
        subscription_features = subscription_features.merge(surprise_count, on='user_id', how='left')
        subscription_features['surprise_subscriptions'] = subscription_features['surprise_subscriptions'].fillna(0)
        subscription_features['prefers_surprise_box'] = (
            subscription_features['surprise_subscriptions'] /
            subscription_features['total_subscriptions_lifetime']
        )

    # === SUBSCRIPTION ENGAGEMENT ===
    # Subscription tenure
    if 'created' in historical_subs.columns:
        tenure = historical_subs.groupby('user_id')['created'].min().reset_index()
        tenure['subscription_tenure_days'] = (cutoff_date - tenure['created']).dt.days
        subscription_features = subscription_features.merge(tenure[['user_id', 'subscription_tenure_days']], on='user_id', how='left')

    # Days since last subscription order
    if 'last_order' in historical_subs.columns:
        last_sub_order = historical_subs.groupby('user_id')['last_order'].max().reset_index()
        last_sub_order['days_since_last_subscription_order'] = (cutoff_date - last_sub_order['last_order']).dt.days
        subscription_features = subscription_features.merge(last_sub_order[['user_id', 'days_since_last_subscription_order']], on='user_id', how='left')

    # Subscription order rate and revenue (requires joining with orders - will calculate if subscription_id exists)
    historical_orders = orders_df[orders_df['date_placed'] < cutoff_date].copy()
    if 'subscription_id' in historical_orders.columns:
        sub_orders = historical_orders[historical_orders['subscription_id'].notna()]
        sub_order_count = sub_orders.groupby('user_id').size().reset_index()
        sub_order_count.columns = ['user_id', 'subscription_orders']

        total_orders = historical_orders.groupby('user_id').size().reset_index()
        total_orders.columns = ['user_id', 'total_orders']

        sub_rate = sub_order_count.merge(total_orders, on='user_id', how='right')
        sub_rate['subscription_orders'] = sub_rate['subscription_orders'].fillna(0)
        sub_rate['subscription_order_rate'] = sub_rate['subscription_orders'] / sub_rate['total_orders']

        subscription_features = subscription_features.merge(sub_rate[['user_id', 'subscription_order_rate']], on='user_id', how='left')

    logger.info(f"  ✓ Created {len(subscription_features.columns)-1} subscription features for {len(subscription_features):,} users")

    return subscription_features


# ============================================================================
# FEATURE ENGINEERING - CATEGORY 6: DISCOUNT & PROMOTION
# ============================================================================

def create_discount_promotion_features(orders_df, voucher_applications_df, temporal_params):
    """
    Create discount and promotion features (Category 6).

    Args:
        orders_df: Cleaned orders DataFrame
        voucher_applications_df: Cleaned voucher applications DataFrame
        temporal_params: Dictionary from setup_temporal_validation()

    Returns:
        pd.DataFrame: Discount & promotion features by user_id
    """
    logger.info("Creating Category 6: Discount & Promotion Features...")

    cutoff_date = temporal_params['feature_cutoff_date']
    historical_vouchers = voucher_applications_df[voucher_applications_df['created'] < cutoff_date].copy()

    # === VOUCHER USAGE ===
    voucher_features = historical_vouchers.groupby('user_id').agg({
        'voucher_id': ['count', 'nunique'],
        'created': 'max'
    }).reset_index()
    voucher_features.columns = ['user_id', 'total_vouchers_used_lifetime', 'unique_vouchers_used', 'last_voucher_date']

    # Vouchers used in last 90 days
    recent_90d = cutoff_date - pd.Timedelta(days=90)
    recent_vouchers = historical_vouchers[historical_vouchers['created'] >= recent_90d].groupby('user_id').size().reset_index()
    recent_vouchers.columns = ['user_id', 'vouchers_used_last_90d']
    voucher_features = voucher_features.merge(recent_vouchers, on='user_id', how='left')
    voucher_features['vouchers_used_last_90d'] = voucher_features['vouchers_used_last_90d'].fillna(0)

    # Days since last voucher use
    voucher_features['days_since_last_voucher_use'] = (cutoff_date - voucher_features['last_voucher_date']).dt.days
    voucher_features = voucher_features.drop(columns=['last_voucher_date'])

    # === DISCOUNT BEHAVIOR (from orders) ===
    historical_orders = orders_df[orders_df['date_placed'] < cutoff_date].copy()

    # Parse discounts JSON if available
    if 'discounts' in historical_orders.columns:
        discount_amounts = []
        for idx, row in historical_orders.iterrows():
            try:
                if pd.notna(row['discounts']) and row['discounts'] != '[]':
                    discounts = json.loads(row['discounts'].replace("'", '"'))
                    total_discount = sum([float(d.get('amount', 0)) for d in discounts])
                    discount_amounts.append({
                        'user_id': row['user_id'],
                        'order_id': row['id'],
                        'discount_amount': total_discount
                    })
            except:
                continue

        if len(discount_amounts) > 0:
            discount_df = pd.DataFrame(discount_amounts)

            # Aggregate discount features
            discount_agg = discount_df.groupby('user_id').agg({
                'discount_amount': ['sum', 'mean', 'max'],
                'order_id': 'count'
            }).reset_index()
            discount_agg.columns = ['user_id', 'total_discount_amount_lifetime', 'avg_discount_per_order',
                                   'max_discount_received', 'discount_orders_count']

            voucher_features = voucher_features.merge(discount_agg, on='user_id', how='left')

    # Voucher usage rate (voucher orders / total orders)
    total_orders = historical_orders.groupby('user_id').size().reset_index()
    total_orders.columns = ['user_id', 'total_orders']
    voucher_features = voucher_features.merge(total_orders, on='user_id', how='left')

    voucher_features['voucher_usage_rate'] = (
        voucher_features['total_vouchers_used_lifetime'] /
        voucher_features['total_orders']
    )

    # Discount dependency score (orders with discount / total orders)
    if 'discount_orders_count' in voucher_features.columns:
        voucher_features['discount_dependency_score'] = (
            voucher_features['discount_orders_count'] /
            voucher_features['total_orders']
        )

    # Responds to promotions (used voucher in last 90d)
    voucher_features['responds_to_promotions'] = (voucher_features['vouchers_used_last_90d'] > 0).astype(int)

    voucher_features = voucher_features.drop(columns=['total_orders'], errors='ignore')

    logger.info(f"  ✓ Created {len(voucher_features.columns)-1} discount & promotion features for {len(voucher_features):,} users")

    return voucher_features


# ============================================================================
# FEATURE ENGINEERING - CATEGORY 7: SHIPPING & LOGISTICS
# ============================================================================

def create_shipping_logistics_features(orders_df, temporal_params):
    """
    Create shipping and logistics features (Category 7).

    Args:
        orders_df: Cleaned orders DataFrame
        temporal_params: Dictionary from setup_temporal_validation()

    Returns:
        pd.DataFrame: Shipping & logistics features by user_id
    """
    logger.info("Creating Category 7: Shipping & Logistics Features...")

    cutoff_date = temporal_params['feature_cutoff_date']
    historical = orders_df[orders_df['date_placed'] < cutoff_date].copy()

    # === SHIPPING PREFERENCES ===
    # Primary shipping method (most used)
    if 'shipping_method' in historical.columns:
        shipping_counts = historical.groupby(['user_id', 'shipping_method']).size().reset_index(name='count')
        primary_method = shipping_counts.loc[shipping_counts.groupby('user_id')['count'].idxmax()]
        primary_method = primary_method[['user_id', 'shipping_method']]
        primary_method.columns = ['user_id', 'primary_shipping_method']

        # Shipping method diversity
        method_diversity = historical.groupby('user_id')['shipping_method'].nunique().reset_index()
        method_diversity.columns = ['user_id', 'shipping_method_diversity']

        shipping_features = primary_method.merge(method_diversity, on='user_id', how='left')

        # Prefers specific methods
        shipping_features['prefers_ninjavan'] = (shipping_features['primary_shipping_method'] == 'ninjavan').astype(int)
        shipping_features['prefers_singpost'] = (shipping_features['primary_shipping_method'] == 'singpost').astype(int)

        # Has changed shipping method
        shipping_features['has_changed_shipping_method'] = (shipping_features['shipping_method_diversity'] > 1).astype(int)
    else:
        shipping_features = pd.DataFrame({'user_id': historical['user_id'].unique()})

    # === SHIPPING PATTERNS ===
    if 'shipping_incl_tax' in historical.columns:
        shipping_cost = historical.groupby('user_id')['shipping_incl_tax'].agg(['mean', 'sum']).reset_index()
        shipping_cost.columns = ['user_id', 'avg_shipping_cost', 'total_shipping_paid']
        shipping_features = shipping_features.merge(shipping_cost, on='user_id', how='left')

        # Free shipping orders percentage
        free_shipping = (historical['shipping_incl_tax'] == 0).groupby(historical['user_id']).agg(['sum', 'count']).reset_index()
        free_shipping.columns = ['user_id', 'free_shipping_orders', 'total_orders']
        free_shipping['free_shipping_orders_pct'] = (free_shipping['free_shipping_orders'] / free_shipping['total_orders']) * 100
        shipping_features = shipping_features.merge(free_shipping[['user_id', 'free_shipping_orders_pct']], on='user_id', how='left')

    # Average days to ship
    if 'shipping_date' in historical.columns and 'date_placed' in historical.columns:
        historical['days_to_ship'] = (historical['shipping_date'] - historical['date_placed']).dt.days
        days_to_ship = historical.groupby('user_id')['days_to_ship'].mean().reset_index()
        days_to_ship.columns = ['user_id', 'avg_days_to_ship']
        shipping_features = shipping_features.merge(days_to_ship, on='user_id', how='left')

        # Has shipping delays (>7 days)
        has_delays = (historical['days_to_ship'] > 7).groupby(historical['user_id']).any().reset_index()
        has_delays.columns = ['user_id', 'has_shipping_delays']
        has_delays['has_shipping_delays'] = has_delays['has_shipping_delays'].astype(int)
        shipping_features = shipping_features.merge(has_delays, on='user_id', how='left')

    # === ADDRESS PATTERNS ===
    if 'shipping_postal_code' in historical.columns:
        address_counts = historical.groupby('user_id')['shipping_postal_code'].agg(['nunique', 'count']).reset_index()
        address_counts.columns = ['user_id', 'num_unique_shipping_addresses', 'total_orders']
        address_counts['address_consistency_score'] = 1 - (address_counts['num_unique_shipping_addresses'] / address_counts['total_orders'])
        address_counts['uses_multiple_addresses'] = (address_counts['num_unique_shipping_addresses'] > 1).astype(int)

        shipping_features = shipping_features.merge(
            address_counts[['user_id', 'num_unique_shipping_addresses', 'address_consistency_score', 'uses_multiple_addresses']],
            on='user_id',
            how='left'
        )

    logger.info(f"  ✓ Created {len(shipping_features.columns)-1} shipping & logistics features for {len(shipping_features):,} users")

    return shipping_features


# ============================================================================
# FEATURE ENGINEERING - CATEGORY 8: ENGAGEMENT & BEHAVIORAL
# ============================================================================

def create_engagement_behavioral_features(events_df, orders_df, temporal_params):
    """
    Create engagement and behavioral features (Category 8).

    Args:
        events_df: Cleaned events DataFrame
        orders_df: Cleaned orders DataFrame
        temporal_params: Dictionary from setup_temporal_validation()

    Returns:
        pd.DataFrame: Engagement & behavioral features by user_id
    """
    logger.info("Creating Category 8: Engagement & Behavioral Features...")

    cutoff_date = temporal_params['feature_cutoff_date']
    historical_events = events_df[events_df['timestamp'] < cutoff_date].copy()

    # === EVENT VOLUME ===
    event_features = historical_events.groupby('user_id').agg({
        'event': 'count',
        'timestamp': ['min', 'max']
    }).reset_index()
    event_features.columns = ['user_id', 'total_events_lifetime', 'first_event', 'last_event']

    # Unique event types
    event_types = historical_events.groupby('user_id')['event'].nunique().reset_index()
    event_types.columns = ['user_id', 'unique_event_types']
    event_features = event_features.merge(event_types, on='user_id', how='left')

    # Events in last 30/60/90 days
    for window_days in [30, 60, 90]:
        window_start = cutoff_date - pd.Timedelta(days=window_days)
        window_events = historical_events[historical_events['timestamp'] >= window_start].groupby('user_id').size().reset_index()
        window_events.columns = ['user_id', f'events_last_{window_days}d']
        event_features = event_features.merge(window_events, on='user_id', how='left')

    # Active days (unique dates with events)
    historical_events['event_date'] = historical_events['timestamp'].dt.date
    active_days = historical_events.groupby('user_id')['event_date'].nunique().reset_index()
    active_days.columns = ['user_id', 'active_days']
    event_features = event_features.merge(active_days, on='user_id', how='left')

    # Events per active day
    event_features['avg_events_per_active_day'] = (
        event_features['total_events_lifetime'] /
        event_features['active_days']
    )

    # === HIGH-INTENT EVENTS ===
    # Identify high-intent event types
    high_intent_events = ['ACTION_ADD_ITEM_TO_CART', 'ACTION_VIEW_PRODUCT', 'ACTION_VOUCHER_APPLY']

    for event_type in high_intent_events:
        event_counts = historical_events[historical_events['event'] == event_type].groupby('user_id').size().reset_index()
        event_counts.columns = ['user_id', event_type.lower() + '_events']
        event_features = event_features.merge(event_counts, on='user_id', how='left')

    # Total high-intent events
    high_intent_cols = [col for col in event_features.columns if any(x in col for x in ['add_item', 'view_product', 'voucher_apply'])]
    if high_intent_cols:
        event_features['total_high_intent_events'] = event_features[high_intent_cols].fillna(0).sum(axis=1)
        event_features['high_intent_ratio'] = (
            event_features['total_high_intent_events'] /
            event_features['total_events_lifetime']
        )

    # === BEHAVIORAL PATTERNS ===
    # Days since last event
    event_features['days_since_last_event'] = (cutoff_date - event_features['last_event']).dt.days

    # Days since last cart add
    if 'action_add_item_to_cart_events' in event_features.columns:
        cart_events = historical_events[historical_events['event'] == 'ACTION_ADD_ITEM_TO_CART']
        last_cart = cart_events.groupby('user_id')['timestamp'].max().reset_index()
        last_cart['days_since_last_cart_add'] = (cutoff_date - last_cart['timestamp']).dt.days
        event_features = event_features.merge(last_cart[['user_id', 'days_since_last_cart_add']], on='user_id', how='left')

    # Is engaged user (events in last 30 days)
    event_features['is_engaged_user'] = (event_features.get('events_last_30d', 0) > 0).astype(int)

    # Events per order ratio (requires orders data)
    historical_orders = orders_df[orders_df['date_placed'] < cutoff_date].copy()
    order_counts = historical_orders.groupby('user_id').size().reset_index()
    order_counts.columns = ['user_id', 'total_orders']
    event_features = event_features.merge(order_counts, on='user_id', how='left')
    event_features['events_per_order_ratio'] = (
        event_features['total_events_lifetime'] /
        event_features['total_orders'].fillna(1)
    )

    # Clean up
    event_features = event_features.drop(columns=['first_event', 'last_event', 'total_orders'], errors='ignore')

    logger.info(f"  ✓ Created {len(event_features.columns)-1} engagement & behavioral features for {len(event_features):,} users")

    return event_features


# ============================================================================
# FEATURE ENGINEERING - CATEGORY 9: CUSTOMER LIFECYCLE
# ============================================================================

def create_customer_lifecycle_features(users_df, user_references_df, orders_df, temporal_params):
    """
    Create customer lifecycle features (Category 9).

    Args:
        users_df: Cleaned users DataFrame
        user_references_df: User references DataFrame
        orders_df: Cleaned orders DataFrame
        temporal_params: Dictionary from setup_temporal_validation()

    Returns:
        pd.DataFrame: Customer lifecycle features by user_id
    """
    logger.info("Creating Category 9: Customer Lifecycle Features...")

    cutoff_date = temporal_params['feature_cutoff_date']

    # Start with user demographics
    lifecycle_features = users_df[['id', 'country', 'date_signed_up', 'active']].copy()
    lifecycle_features.columns = ['user_id', 'acquisition_country', 'signup_date', 'is_active_account']

    # === CUSTOMER AGE & TENURE ===
    lifecycle_features['customer_age_days'] = (cutoff_date - lifecycle_features['signup_date']).dt.days

    # Customer lifecycle stage
    def get_lifecycle_stage(age_days):
        if age_days < 30:
            return 'new'
        elif age_days < 90:
            return 'growing'
        elif age_days < 180:
            return 'mature'
        else:
            return 'loyal'

    lifecycle_features['customer_lifecycle_stage'] = lifecycle_features['customer_age_days'].apply(get_lifecycle_stage)

    # Is new customer
    lifecycle_features['is_new_customer'] = (lifecycle_features['customer_age_days'] < 30).astype(int)

    # === ACQUISITION & CHANNEL ===
    # Is Singapore customer
    lifecycle_features['is_singapore_customer'] = (lifecycle_features['acquisition_country'] == 'Singapore').astype(int)

    # Active status
    lifecycle_features['is_active_account'] = (lifecycle_features['is_active_account'] == 'Yes').astype(int)

    # === REFERRAL BEHAVIOR ===
    # Is referred customer
    if 'user_id' in user_references_df.columns:
        referred_users = user_references_df['user_id'].unique()
        lifecycle_features['is_referred_customer'] = lifecycle_features['user_id'].isin(referred_users).astype(int)

    # Has referred others
    if 'referred_by' in user_references_df.columns:
        referrers = user_references_df.groupby('referred_by').size().reset_index()
        referrers.columns = ['user_id', 'total_referrals_made']
        lifecycle_features = lifecycle_features.merge(referrers, on='user_id', how='left')
        lifecycle_features['total_referrals_made'] = lifecycle_features['total_referrals_made'].fillna(0)
        lifecycle_features['has_referred_others'] = (lifecycle_features['total_referrals_made'] > 0).astype(int)

    # === COHORT FEATURES ===
    lifecycle_features['signup_year'] = lifecycle_features['signup_date'].dt.year
    lifecycle_features['signup_month'] = lifecycle_features['signup_date'].dt.month
    lifecycle_features['signup_quarter'] = lifecycle_features['signup_date'].dt.quarter

    # === LIFECYCLE METRICS (requires orders) ===
    historical_orders = orders_df[orders_df['date_placed'] < cutoff_date].copy()

    # Orders per month of tenure
    order_stats = historical_orders.groupby('user_id').agg({
        'id': 'count',
        'total_incl_tax': 'sum'
    }).reset_index()
    order_stats.columns = ['user_id', 'total_orders', 'total_revenue']

    lifecycle_features = lifecycle_features.merge(order_stats, on='user_id', how='left')

    lifecycle_features['orders_per_month_of_tenure'] = (
        lifecycle_features['total_orders'] /
        (lifecycle_features['customer_age_days'] / 30.0 + 0.1)
    )

    lifecycle_features['revenue_per_month_of_tenure'] = (
        lifecycle_features['total_revenue'] /
        (lifecycle_features['customer_age_days'] / 30.0 + 0.1)
    )

    # Clean up
    lifecycle_features = lifecycle_features.drop(columns=['signup_date', 'total_orders', 'total_revenue'], errors='ignore')

    logger.info(f"  ✓ Created {len(lifecycle_features.columns)-1} customer lifecycle features for {len(lifecycle_features):,} users")

    return lifecycle_features


# ============================================================================
# FEATURE ENGINEERING - CATEGORY 10: INTERACTION FEATURES
# ============================================================================

def create_interaction_features(all_features_dict):
    """
    Create interaction features (Category 10).

    This combines features from multiple categories to capture complex relationships.

    Args:
        all_features_dict: Dictionary containing all feature DataFrames from categories 1-9

    Returns:
        pd.DataFrame: Interaction features by user_id
    """
    logger.info("Creating Category 10: Interaction Features...")

    # Get base features for interactions
    cat1 = all_features_dict.get('category_1')  # Purchase history
    cat2 = all_features_dict.get('category_2')  # Monetary
    cat8 = all_features_dict.get('category_8')  # Engagement

    # Start with user_id from category 1
    if cat1 is not None:
        interaction_features = cat1[['user_id']].copy()
    else:
        logger.warning("  ⚠️  No base features available for interactions")
        return pd.DataFrame()

    # === RFM INTERACTIONS ===
    if cat1 is not None:
        # Merge recency and frequency from cat1
        if 'days_since_last_order' in cat1.columns and 'order_frequency' in cat1.columns:
            rfm_df = cat1[['user_id', 'days_since_last_order', 'order_frequency', 'recency_score']].copy()
            interaction_features = interaction_features.merge(rfm_df, on='user_id', how='left')

            # Recency × Frequency interaction
            interaction_features['recency_frequency_interaction'] = (
                interaction_features['days_since_last_order'] *
                interaction_features['order_frequency']
            )

    if cat2 is not None:
        # Merge monetary from cat2
        if 'total_lifetime_revenue' in cat2.columns:
            monetary_df = cat2[['user_id', 'total_lifetime_revenue']].copy()
            interaction_features = interaction_features.merge(monetary_df, on='user_id', how='left')

            # Recency × Monetary interaction
            if 'days_since_last_order' in interaction_features.columns:
                interaction_features['recency_monetary_interaction'] = (
                    interaction_features['days_since_last_order'] *
                    interaction_features['total_lifetime_revenue']
                )

            # Frequency × Monetary interaction
            if 'order_frequency' in interaction_features.columns:
                interaction_features['frequency_monetary_interaction'] = (
                    interaction_features['order_frequency'] *
                    interaction_features['total_lifetime_revenue']
                )

            # RFM score (simple sum - recency is inverse)
            if 'recency_score' in interaction_features.columns:
                # Create frequency and monetary scores (simple quantile-based)
                # Handle case where there aren't enough unique values for 5 bins
                try:
                    interaction_features['frequency_score'] = pd.qcut(
                        interaction_features['order_frequency'].fillna(0),
                        q=5,
                        labels=[1,2,3,4,5],
                        duplicates='drop'
                    ).astype(float)
                except (ValueError, TypeError):
                    # Fallback: use simple ranking
                    interaction_features['frequency_score'] = pd.cut(
                        interaction_features['order_frequency'].fillna(0),
                        bins=[0, 0.001, 0.01, 0.05, 0.1, np.inf],
                        labels=[1,2,3,4,5]
                    ).astype(float)

                interaction_features['frequency_score'] = interaction_features['frequency_score'].fillna(1)

                try:
                    interaction_features['monetary_score'] = pd.qcut(
                        interaction_features['total_lifetime_revenue'].fillna(0),
                        q=5,
                        labels=[1,2,3,4,5],
                        duplicates='drop'
                    ).astype(float)
                except (ValueError, TypeError):
                    # Fallback: use percentile-based cut
                    p20 = interaction_features['total_lifetime_revenue'].quantile(0.2)
                    p40 = interaction_features['total_lifetime_revenue'].quantile(0.4)
                    p60 = interaction_features['total_lifetime_revenue'].quantile(0.6)
                    p80 = interaction_features['total_lifetime_revenue'].quantile(0.8)
                    interaction_features['monetary_score'] = pd.cut(
                        interaction_features['total_lifetime_revenue'].fillna(0),
                        bins=[-np.inf, p20, p40, p60, p80, np.inf],
                        labels=[1,2,3,4,5]
                    ).astype(float)

                interaction_features['monetary_score'] = interaction_features['monetary_score'].fillna(1)

                interaction_features['rfm_score'] = (
                    interaction_features['recency_score'].fillna(1) +
                    interaction_features['frequency_score'] +
                    interaction_features['monetary_score']
                )

    # === ENGAGEMENT × VALUE ===
    if cat8 is not None and cat2 is not None:
        if 'total_events_lifetime' in cat8.columns and 'total_lifetime_revenue' in interaction_features.columns:
            events_df = cat8[['user_id', 'total_events_lifetime']].copy()
            interaction_features = interaction_features.merge(events_df, on='user_id', how='left')

            # Events per dollar spent
            interaction_features['events_per_dollar_spent'] = (
                interaction_features['total_events_lifetime'] /
                (interaction_features['total_lifetime_revenue'] + 0.01)
            )

            # Revenue per event
            interaction_features['revenue_per_event'] = (
                interaction_features['total_lifetime_revenue'] /
                (interaction_features['total_events_lifetime'] + 1)
            )

    # === TIME × BEHAVIOR ===
    if cat1 is not None and cat8 is not None:
        if 'customer_tenure_days' in cat1.columns:
            tenure_df = cat1[['user_id', 'customer_tenure_days', 'total_lifetime_orders']].copy()
            interaction_features = interaction_features.merge(tenure_df, on='user_id', how='left')

        if 'active_days' in cat8.columns:
            active_df = cat8[['user_id', 'active_days']].copy()
            interaction_features = interaction_features.merge(active_df, on='user_id', how='left')

            # Orders per active day
            if 'total_lifetime_orders' in interaction_features.columns:
                interaction_features['orders_per_active_day'] = (
                    interaction_features['total_lifetime_orders'] /
                    (interaction_features['active_days'] + 1)
                )

    # Drop intermediate columns used only for calculation
    cols_to_drop = ['days_since_last_order', 'order_frequency', 'recency_score',
                    'total_lifetime_revenue', 'frequency_score', 'monetary_score',
                    'total_events_lifetime', 'customer_tenure_days', 'total_lifetime_orders', 'active_days']
    interaction_features = interaction_features.drop(columns=[c for c in cols_to_drop if c in interaction_features.columns], errors='ignore')

    logger.info(f"  ✓ Created {len(interaction_features.columns)-1} interaction features for {len(interaction_features):,} users")

    return interaction_features


# ============================================================================
# FEATURE MERGING & INTEGRATION
# ============================================================================

def merge_all_features(feature_dict, segmentation_df, target_df):
    """
    Merge all feature categories and integrate Case 1 segmentation.

    Args:
        feature_dict: Dictionary with keys 'cat1' through 'cat10' containing feature DataFrames
        segmentation_df: Case 1 segmentation with user_id and value_cluster
        target_df: Target variable DataFrame with user_id and target column

    Returns:
        pd.DataFrame: Merged feature dataset with all features and target
    """
    logger.info("="*80)
    logger.info("MERGING ALL FEATURES & INTEGRATING CASE 1 SEGMENTATION")
    logger.info("="*80)

    # Start with target DataFrame (contains eligible users)
    master_df = target_df.copy()
    logger.info(f"Starting with {len(master_df):,} users from target variable")

    # Merge each category
    for cat_name, cat_df in feature_dict.items():
        if cat_df is not None and len(cat_df) > 0:
            before_cols = len(master_df.columns)
            master_df = master_df.merge(cat_df, on='user_id', how='left')
            after_cols = len(master_df.columns)
            logger.info(f"  ✓ Merged {cat_name}: Added {after_cols - before_cols} features")
    
    # Calculate revenue_growth_rate from spend_trend (Category 1) and total_lifetime_revenue (Category 2)
    if 'spend_trend' in master_df.columns and 'total_lifetime_revenue' in master_df.columns:
        # Revenue growth rate = (spend_trend / previous_period_revenue) * 100
        # Approximate previous period revenue as total_lifetime_revenue - spend_trend
        # This gives us a growth rate indicator
        previous_period_revenue = master_df['total_lifetime_revenue'] - master_df['spend_trend'].fillna(0)
        master_df['revenue_growth_rate'] = (
            master_df['spend_trend'].fillna(0) / previous_period_revenue.clip(lower=1)
        ) * 100
        # Replace infinite values with 0
        master_df['revenue_growth_rate'] = master_df['revenue_growth_rate'].replace([np.inf, -np.inf], 0)
        logger.info("  ✓ Calculated revenue_growth_rate from spend_trend and total_lifetime_revenue")

    # Merge Case 1 segmentation (value_cluster)
    logger.info("\nIntegrating Case 1 RFM Segmentation...")
    before_cols = len(master_df.columns)
    master_df = master_df.merge(segmentation_df[['user_id', 'value_cluster']], on='user_id', how='left')

    # Check for missing clusters
    missing_clusters = master_df['value_cluster'].isna().sum()
    if missing_clusters > 0:
        logger.warning(f"  ⚠️  {missing_clusters} users missing value_cluster - will be imputed")
        # Fill missing clusters with mode (most common cluster)
        master_df['value_cluster'] = master_df['value_cluster'].fillna(master_df['value_cluster'].mode()[0])

    logger.info(f"  ✓ Added value_cluster from Case 1 segmentation")
    logger.info(f"  Cluster distribution:")
    for cluster in sorted(master_df['value_cluster'].unique()):
        count = (master_df['value_cluster'] == cluster).sum()
        pct = count / len(master_df) * 100
        logger.info(f"    Cluster {int(cluster)}: {count:,} ({pct:.1f}%)")

    logger.info(f"\n✓ Total features (before preprocessing): {len(master_df.columns) - 2} (excluding user_id and target)")
    logger.info(f"✓ Total users: {len(master_df):,}")

    return master_df


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_features(features_df, target_col='will_purchase'):
    """
    Preprocess features: handle missing values, encode categoricals, prepare for modeling.

    Args:
        features_df: Merged feature DataFrame with all features and target
        target_col: Name of target column (default: 'will_purchase')

    Returns:
        tuple: (X_features, y_target, feature_names, categorical_cols, numeric_cols)
    """
    logger.info("="*80)
    logger.info("DATA PREPROCESSING")
    logger.info("="*80)

    df = features_df.copy()

    # Separate target and features
    y = df[target_col]
    df = df.drop(columns=[target_col])

    # Store user_id for later reference
    user_ids = df['user_id']
    df = df.drop(columns=['user_id'])

    logger.info(f"Starting features: {len(df.columns)}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    # === IDENTIFY COLUMN TYPES ===
    categorical_cols = []
    numeric_cols = []

    for col in df.columns:
        if df[col].dtype == 'object' or col in ['value_cluster', 'customer_lifecycle_stage',
                                                  'acquisition_country', 'primary_shipping_method']:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    logger.info(f"\nColumn types:")
    logger.info(f"  Categorical: {len(categorical_cols)}")
    logger.info(f"  Numeric: {len(numeric_cols)}")

    # === HANDLE MISSING VALUES ===
    logger.info("\nHandling missing values...")

    # Subscription/Event/Voucher features: Fill with 0 (indicates absence)
    subscription_features = [col for col in df.columns if 'subscription' in col.lower() or 'surprise' in col.lower()]
    event_features = [col for col in df.columns if 'event' in col.lower()]
    voucher_features = [col for col in df.columns if 'voucher' in col.lower() or 'discount' in col.lower()]

    for col in subscription_features + event_features + voucher_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Recency features: Fill with large value (very old)
    recency_features = [col for col in df.columns if 'days_since' in col.lower() or 'recency' in col.lower()]
    max_recency = df[recency_features].max().max() if len(recency_features) > 0 else 9999
    for col in recency_features:
        if col in df.columns:
            df[col] = df[col].fillna(max_recency + 1)

    # Ratio features: Fill with -1 (undefined/not applicable)
    ratio_features = [col for col in df.columns if any(x in col.lower() for x in ['_pct', '_ratio', '_rate', 'per_'])]
    for col in ratio_features:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    # Categorical features: Fill with 'unknown'
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')

    # Remaining numeric features: Fill with 0
    for col in numeric_cols:
        if col in df.columns and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(0)

    # Check for any remaining NaN
    remaining_na = df.isna().sum().sum()
    if remaining_na > 0:
        logger.warning(f"  ⚠️  {remaining_na} NaN values remaining - filling with 0")
        df = df.fillna(0)

    logger.info(f"  ✓ All missing values handled")

    # === ENCODE CATEGORICAL VARIABLES ===
    logger.info("\nEncoding categorical variables...")

    encoded_dfs = [df[numeric_cols]]  # Start with numeric columns

    for col in categorical_cols:
        if col in df.columns:
            # One-hot encode with drop_first=True to avoid multicollinearity
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            encoded_dfs.append(dummies)
            logger.info(f"  ✓ Encoded {col}: {len(dummies.columns)} dummy variables")

    # Combine all encoded features
    X = pd.concat(encoded_dfs, axis=1)

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0)

    feature_names = X.columns.tolist()

    logger.info(f"\n✓ Preprocessing complete:")
    logger.info(f"  Total features: {len(feature_names)}")
    logger.info(f"  Samples: {len(X)}")
    logger.info(f"  No NaN values: {X.isna().sum().sum() == 0}")
    logger.info(f"  No infinite values: {np.isinf(X.values).sum() == 0}")

    return X, y, feature_names, user_ids


def save_processed_features(X, y, feature_names, user_ids, target_name='will_purchase', output_dir='outputs/features'):
    """
    Save processed features to CSV for future use.

    Args:
        X: Feature matrix (DataFrame or array)
        y: Target variable
        feature_names: List of feature names
        user_ids: User IDs
        target_name: Name of target column
        output_dir: Directory to save features
    """
    logger.info(f"\nSaving processed features to {output_dir}...")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)

    # Add user_id and target
    output_df = pd.DataFrame({'user_id': user_ids})
    output_df = pd.concat([output_df, X], axis=1)
    output_df[target_name] = y.values

    # Save to CSV
    output_path = f"{output_dir}/{target_name}_features.csv"
    output_df.to_csv(output_path, index=False)

    logger.info(f"  ✓ Saved {len(output_df):,} rows × {len(output_df.columns)} columns to {output_path}")
    logger.info(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return output_path


# ============================================================================
# COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================================

def run_complete_feature_engineering(target_type='loyalty'):
    """
    Run the complete feature engineering pipeline for Case 2.

    Args:
        target_type: 'loyalty' for purchase propensity or 'high_value' for upsell model

    Returns:
        tuple: (X, y, feature_names, user_ids, temporal_params)
    """
    logger.info("="*80)
    logger.info(f"COMPLETE FEATURE ENGINEERING PIPELINE - {target_type.upper()} MODEL")
    logger.info("="*80)

    # Step 1: Load and clean data
    datasets = load_all_datasets()
    cleaned = clean_all_datasets(datasets)

    # Step 2: Setup temporal validation
    temporal_params = setup_temporal_validation(cleaned['orders'])

    # Step 3: Recalculate RFM segmentation with proper temporal validation (NO DATA LEAKAGE)
    segmentation = recalculate_rfm_segmentation(cleaned['orders'], temporal_params)

    # Step 4: Create target variables (both for validation summary)
    loyalty_target = create_loyalty_target(cleaned['orders'], temporal_params)
    high_value_target = create_high_value_target(cleaned['orders'], temporal_params)
    
    # Print validation summary
    print_validation_summary(temporal_params, loyalty_target, high_value_target)
    
    # Select target based on target_type
    if target_type == 'loyalty':
        target_df = loyalty_target
        target_col = 'will_purchase'
    else:
        target_df = high_value_target
        target_col = 'will_be_high_value'

    # Step 5: Create all features
    logger.info("\n" + "="*80)
    logger.info("CREATING ALL FEATURES")
    logger.info("="*80)

    cat1 = create_purchase_history_features(cleaned['orders'], temporal_params)
    cat2 = create_monetary_features(cleaned['orders'], temporal_params)
    cat3 = create_product_engagement_features(cleaned['orders'], cleaned['products'], temporal_params)
    cat4 = create_temporal_features(cleaned['orders'], temporal_params)
    cat5 = create_subscription_features(cleaned['subscriptions'], cleaned['orders'], temporal_params)
    cat6 = create_discount_promotion_features(cleaned['orders'], cleaned['voucher_applications'], temporal_params)
    cat7 = create_shipping_logistics_features(cleaned['orders'], temporal_params)
    cat8 = create_engagement_behavioral_features(cleaned['events'], cleaned['orders'], temporal_params)
    cat9 = create_customer_lifecycle_features(cleaned['users'], cleaned['user_references'], cleaned['orders'], temporal_params)

    # Category 10 requires categories 1, 2, 8
    all_features_for_interactions = {
        'category_1': cat1,
        'category_2': cat2,
        'category_8': cat8
    }
    cat10 = create_interaction_features(all_features_for_interactions)

    # Step 6: Merge all features
    feature_dict = {
        'Category 1 (Purchase History)': cat1,
        'Category 2 (Monetary)': cat2,
        'Category 3 (Product Engagement)': cat3,
        'Category 4 (Temporal)': cat4,
        'Category 5 (Subscription)': cat5,
        'Category 6 (Discount)': cat6,
        'Category 7 (Shipping)': cat7,
        'Category 8 (Engagement)': cat8,
        'Category 9 (Lifecycle)': cat9,
        'Category 10 (Interactions)': cat10
    }

    merged_features = merge_all_features(feature_dict, segmentation, target_df)

    # Step 7: Preprocess features
    X, y, feature_names, user_ids = preprocess_features(merged_features, target_col=target_col)

    # Step 8: Save processed features
    save_processed_features(X, y, feature_names, user_ids, target_name=target_col)

    logger.info("\n" + "="*80)
    logger.info("✓ COMPLETE FEATURE ENGINEERING PIPELINE FINISHED")
    logger.info("="*80)

    return X, y, feature_names, user_ids, temporal_params


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Run the pipeline
    cleaned, segmentation, temporal_params, loyalty_target, high_value_target = run_data_loading_and_validation()

    print("\n" + "="*80)
    print("TESTING FEATURE ENGINEERING - ALL 10 CATEGORIES")
    print("="*80)

    # Test Category 1: Purchase History & Recency
    cat1_features = create_purchase_history_features(cleaned['orders'], temporal_params)
    print(f"\nCategory 1 - Purchase History & Recency:")
    print(f"  Features: {len(cat1_features.columns)-1}")
    print(f"  Users: {len(cat1_features):,}")

    # Test Category 2: Monetary Value
    cat2_features = create_monetary_features(cleaned['orders'], temporal_params)
    print(f"\nCategory 2 - Monetary Value:")
    print(f"  Features: {len(cat2_features.columns)-1}")
    print(f"  Users: {len(cat2_features):,}")

    # Test Category 3: Product Engagement
    cat3_features = create_product_engagement_features(cleaned['orders'], cleaned['products'], temporal_params)
    print(f"\nCategory 3 - Product Engagement:")
    print(f"  Features: {len(cat3_features.columns)-1}")
    print(f"  Users: {len(cat3_features):,}")

    # Test Category 4: Temporal & Seasonality
    cat4_features = create_temporal_features(cleaned['orders'], temporal_params)
    print(f"\nCategory 4 - Temporal & Seasonality:")
    print(f"  Features: {len(cat4_features.columns)-1}")
    print(f"  Users: {len(cat4_features):,}")

    # Test Category 5: Subscription
    cat5_features = create_subscription_features(cleaned['subscriptions'], cleaned['orders'], temporal_params)
    print(f"\nCategory 5 - Subscription:")
    print(f"  Features: {len(cat5_features.columns)-1}")
    print(f"  Users: {len(cat5_features):,}")

    # Test Category 6: Discount & Promotion
    cat6_features = create_discount_promotion_features(cleaned['orders'], cleaned['voucher_applications'], temporal_params)
    print(f"\nCategory 6 - Discount & Promotion:")
    print(f"  Features: {len(cat6_features.columns)-1}")
    print(f"  Users: {len(cat6_features):,}")

    # Test Category 7: Shipping & Logistics
    cat7_features = create_shipping_logistics_features(cleaned['orders'], temporal_params)
    print(f"\nCategory 7 - Shipping & Logistics:")
    print(f"  Features: {len(cat7_features.columns)-1}")
    print(f"  Users: {len(cat7_features):,}")

    # Test Category 8: Engagement & Behavioral
    cat8_features = create_engagement_behavioral_features(cleaned['events'], cleaned['orders'], temporal_params)
    print(f"\nCategory 8 - Engagement & Behavioral:")
    print(f"  Features: {len(cat8_features.columns)-1}")
    print(f"  Users: {len(cat8_features):,}")

    # Test Category 9: Customer Lifecycle
    cat9_features = create_customer_lifecycle_features(cleaned['users'], cleaned['user_references'], cleaned['orders'], temporal_params)
    print(f"\nCategory 9 - Customer Lifecycle:")
    print(f"  Features: {len(cat9_features.columns)-1}")
    print(f"  Users: {len(cat9_features):,}")

    # Test Category 10: Interaction Features
    all_features = {
        'category_1': cat1_features,
        'category_2': cat2_features,
        'category_8': cat8_features
    }
    cat10_features = create_interaction_features(all_features)
    print(f"\nCategory 10 - Interaction Features:")
    print(f"  Features: {len(cat10_features.columns)-1}")
    print(f"  Users: {len(cat10_features):,}")

    print("\n" + "="*80)
    print("SUMMARY - ALL 10 CATEGORIES")
    print("="*80)
    all_feature_dfs = [cat1_features, cat2_features, cat3_features, cat4_features, cat5_features,
                       cat6_features, cat7_features, cat8_features, cat9_features, cat10_features]
    total_features = sum([len(df.columns)-1 for df in all_feature_dfs])
    print(f"Total features created: {total_features}")
    print(f"Datasets loaded and cleaned: {len(cleaned)}")
    print(f"Case 1 segmentation loaded: {len(segmentation):,} users")
    print(f"Temporal validation configured: {temporal_params['observation_weeks']}-week window")
    print(f"Loyalty target created: {len(loyalty_target):,} users")
    print(f"High-value target created: {len(high_value_target):,} users")
    print("="*80)
