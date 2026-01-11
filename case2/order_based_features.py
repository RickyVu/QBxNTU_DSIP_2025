# ---
# Order-Based Feature Engineering
# Creates one entry per order, predicting if user will purchase again within 12 weeks
# This answers: "Given a user just made a purchase, will they purchase again?"
# ---

import pandas as pd
import numpy as np
import os
import logging
from datetime import timedelta
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "../original_data"
PREDICTION_WINDOW_WEEKS = 12  # 12-week lookforward
OUTPUT_DIR = "outputs/order_based"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_clean_orders(data_path=DATA_PATH):
    """Load and clean orders dataset."""
    logger.info("Loading and cleaning orders data...")
    
    orders = pd.read_csv(f"{data_path}/orders.csv")
    
    # Filter to completed orders only
    orders = orders[orders["status"].isin(["Shipped", "Complete"])]
    orders = orders[orders["total_incl_tax"] > 0]
    
    # Fix datetime columns
    orders["date_placed"] = pd.to_datetime(orders["date_placed"], errors="coerce", utc=True).dt.tz_localize(None)
    orders["shipping_date"] = pd.to_datetime(orders["shipping_date"], errors="coerce", utc=True).dt.tz_localize(None)
    
    # Drop rows with missing critical data
    orders = orders.dropna(subset=["date_placed", "user_id"])
    
    logger.info(f"✓ Cleaned orders: {len(orders):,} rows")
    logger.info(f"  Date range: {orders['date_placed'].min().date()} to {orders['date_placed'].max().date()}")
    logger.info(f"  Unique users: {orders['user_id'].nunique():,}")
    
    return orders


def load_all_datasets(data_path=DATA_PATH):
    """Load all required datasets."""
    logger.info("Loading all datasets...")
    
    datasets = {}
    
    # Load main datasets
    datasets['orders'] = load_and_clean_orders(data_path)
    datasets['users'] = pd.read_csv(f"{data_path}/users.csv")
    datasets['subscriptions'] = pd.read_csv(f"{data_path}/subscriptions.csv")
    datasets['products'] = pd.read_csv(f"{data_path}/products.csv")
    datasets['events'] = pd.read_csv(f"{data_path}/events.csv")
    datasets['voucher_applications'] = pd.read_csv(f"{data_path}/voucher_applications.csv")
    
    # Clean datetime columns
    if 'date_signed_up' in datasets['users'].columns:
        datasets['users']['date_signed_up'] = pd.to_datetime(
            datasets['users']['date_signed_up'], errors="coerce"
        )
    
    # Clean events
    datasets['events']['timestamp'] = pd.to_datetime(
        datasets['events']['timestamp'], errors="coerce", utc=True
    ).dt.tz_localize(None)
    datasets['events'] = datasets['events'].dropna(subset=['user_id', 'timestamp'])
    
    # Clean subscriptions
    for col in ['last_order', 'next_order', 'created', 'updated']:
        if col in datasets['subscriptions'].columns:
            datasets['subscriptions'][col] = pd.to_datetime(
                datasets['subscriptions'][col], errors="coerce", utc=True
            ).dt.tz_localize(None)
    
    # Clean voucher applications
    datasets['voucher_applications']['created'] = pd.to_datetime(
        datasets['voucher_applications']['created'], errors="coerce", utc=True
    ).dt.tz_localize(None)
    
    logger.info(f"✓ All datasets loaded")
    
    return datasets


# ============================================================================
# ORDER-BASED ENTRY CREATION
# ============================================================================

def get_week_start(purchase_date):
    """Get the Monday of the week for a given date."""
    days_since_monday = purchase_date.weekday()
    week_start = purchase_date - timedelta(days=days_since_monday)
    return week_start


def create_order_based_entries(orders_df, prediction_weeks=PREDICTION_WINDOW_WEEKS):
    """
    Create one entry per order.
    
    For each order, we predict: Will this user purchase again within next 12 weeks?
    
    Args:
        orders_df: Cleaned orders DataFrame
        prediction_weeks: Number of weeks to look forward
        
    Returns:
        pd.DataFrame: One row per order with target variable
    """
    logger.info("="*80)
    logger.info("CREATING ORDER-BASED ENTRIES")
    logger.info("="*80)
    
    # Get unique orders (one per order_id)
    orders_unique = orders_df.groupby(['user_id', 'id', 'date_placed']).agg({
        'total_incl_tax': 'first',
        'shipping_method': 'first'
    }).reset_index()
    
    # Add week start
    orders_unique['week_start'] = orders_unique['date_placed'].apply(get_week_start)
    
    logger.info(f"Total orders: {len(orders_unique):,}")
    logger.info(f"Unique users: {orders_unique['user_id'].nunique():,}")
    logger.info(f"Date range: {orders_unique['date_placed'].min().date()} to {orders_unique['date_placed'].max().date()}")
    
    # Create target: Will user purchase again within prediction_weeks?
    logger.info(f"\nCreating {prediction_weeks}-week lookforward targets (vectorized)...")
    
    # OPTIMIZED APPROACH - Only check next order!
    # Since orders are sorted, we only need to check if the NEXT order is within the window
    
    # Sort by user and date
    orders_sorted = orders_df[['user_id', 'date_placed']].sort_values(['user_id', 'date_placed']).copy()
    
    # Create a mapping of order to whether there's a repurchase
    targets_dict = {}
    prediction_window = pd.Timedelta(weeks=prediction_weeks)
    
    for user_id, user_orders in tqdm(orders_sorted.groupby('user_id'), desc="Processing users"):
        user_order_dates = user_orders['date_placed'].values
        
        # For each order by this user
        for i in range(len(user_order_dates)):
            order_date = user_order_dates[i]
            
            # Check if there's a next order and if it's within prediction window
            if i + 1 < len(user_order_dates):
                next_order_date = user_order_dates[i + 1]
                prediction_end = order_date + prediction_window
                has_repurchase = next_order_date <= prediction_end
            else:
                # This is the last order for this user
                has_repurchase = False
            
            # Use pandas Timestamp for key
            targets_dict[(user_id, pd.Timestamp(order_date))] = 1 if has_repurchase else 0
    
    # Map targets back to orders_unique
    orders_unique['will_repurchase'] = orders_unique.apply(
        lambda row: targets_dict.get((row['user_id'], row['date_placed']), 0),
        axis=1
    )
    
    # Calculate statistics
    n_positive = orders_unique['will_repurchase'].sum()
    n_total = len(orders_unique)
    positive_rate = n_positive / n_total
    
    logger.info(f"\n✓ Targets created:")
    logger.info(f"  Total orders: {n_total:,}")
    logger.info(f"  Will repurchase: {n_positive:,} ({positive_rate:.1%})")
    logger.info(f"  Won't repurchase: {n_total - n_positive:,} ({1-positive_rate:.1%})")
    
    return orders_unique


# ============================================================================
# FEATURE CALCULATION AT ORDER TIME
# ============================================================================

def calculate_features_at_order_time(order_entries, orders_df, events_df, 
                                      subscriptions_df, users_df, voucher_df):
    """
    Calculate features for each order using only data BEFORE that order.
    
    OPTIMIZED: Calculate features per user, not per date.
    
    Args:
        order_entries: DataFrame with one row per order
        orders_df: All orders
        events_df: Events DataFrame
        subscriptions_df: Subscriptions DataFrame
        users_df: Users DataFrame
        voucher_df: Voucher applications DataFrame
        
    Returns:
        pd.DataFrame: Features for each order
    """
    logger.info("="*80)
    logger.info("CALCULATING FEATURES AT ORDER TIME (OPTIMIZED)")
    logger.info("="*80)
    
    logger.info("This will calculate features per order using historical data...")
    logger.info("Note: For 120k orders, this may take a few minutes.")
    
    # Sort everything by date for efficiency
    orders_sorted = orders_df.sort_values('date_placed')
    events_sorted = events_df.sort_values('timestamp')
    
    all_features = []
    
    # Process by user (much more efficient than by date)
    logger.info(f"Processing {order_entries['user_id'].nunique():,} unique users...")
    
    for user_id, user_order_entries in tqdm(order_entries.groupby('user_id'), desc="Processing users"):
        # Get all data for this user
        user_all_orders = orders_sorted[orders_sorted['user_id'] == user_id]
        user_all_events = events_sorted[events_sorted['user_id'] == user_id]
        user_subs = subscriptions_df[subscriptions_df['user_id'] == user_id]
        user_vouchers = voucher_df[voucher_df['user_id'] == user_id]
        user_info = users_df[users_df['id'] == user_id]
        
        # For each order by this user
        for idx, order_row in user_order_entries.iterrows():
            order_date = order_row['date_placed']
            
            # Get historical data BEFORE this order
            hist_orders = user_all_orders[user_all_orders['date_placed'] < order_date]
            hist_events = user_all_events[user_all_events['timestamp'] < order_date]
            hist_subs = user_subs[user_subs['created'] < order_date]
            hist_vouchers = user_vouchers[user_vouchers['created'] < order_date]
            
            # Calculate features
            features = calculate_single_order_features(
                order_row,
                hist_orders,
                hist_events,
                hist_subs,
                user_info,
                hist_vouchers,
                order_date
            )
            
            all_features.append(features)
    
    # Combine all features
    features_df = pd.DataFrame(all_features)
    
    logger.info(f"\n✓ Features calculated: {len(features_df):,} orders × {len(features_df.columns)} features")
    
    return features_df


def calculate_single_order_features(order_row, hist_orders, hist_events,
                                     hist_subs, user_info, hist_vouchers, order_date):
    """
    Calculate features for a single order.
    
    Args:
        order_row: Single order row
        hist_orders: Historical orders for this user
        hist_events: Historical events for this user
        hist_subs: Historical subscriptions for this user
        user_info: User information
        hist_vouchers: Historical vouchers for this user
        order_date: The order date
        
    Returns:
        dict: Features for this order
    """
    features = {
        'user_id': order_row['user_id'],
        'order_id': order_row['id'],
        'order_date': order_row['date_placed'],
        'week_start': order_row['week_start'],
        'current_order_value': order_row['total_incl_tax'],
        'current_shipping_method': order_row['shipping_method'],
        'will_repurchase': order_row['will_repurchase']
    }
    
    # Purchase history features
    if len(hist_orders) == 0:
        # First order
        features['is_first_order'] = 1
        features['previous_orders_count'] = 0
        features['days_since_last_order'] = 9999
        features['avg_order_value_historical'] = 0
        features['total_revenue_historical'] = 0
        features['customer_tenure_days'] = 0
        features['avg_days_between_orders'] = 0
        features['orders_last_30d'] = 0
        features['spend_last_30d'] = 0
        features['orders_last_60d'] = 0
        features['spend_last_60d'] = 0
        features['orders_last_90d'] = 0
        features['spend_last_90d'] = 0
    else:
        features['is_first_order'] = 0
        features['previous_orders_count'] = len(hist_orders)
        features['days_since_last_order'] = (order_date - hist_orders['date_placed'].max()).days
        features['avg_order_value_historical'] = hist_orders['total_incl_tax'].mean()
        features['total_revenue_historical'] = hist_orders['total_incl_tax'].sum()
        features['customer_tenure_days'] = (order_date - hist_orders['date_placed'].min()).days
        features['avg_days_between_orders'] = features['customer_tenure_days'] / (len(hist_orders) + 1)
        
        # Rolling windows
        for days in [30, 60, 90]:
            window_start = order_date - pd.Timedelta(days=days)
            window_orders = hist_orders[hist_orders['date_placed'] >= window_start]
            features[f'orders_last_{days}d'] = len(window_orders)
            features[f'spend_last_{days}d'] = window_orders['total_incl_tax'].sum() if len(window_orders) > 0 else 0
    
    # Event features
    if len(hist_events) == 0:
        features['total_events_historical'] = 0
        features['events_last_30d'] = 0
        features['events_last_60d'] = 0
        features['events_last_90d'] = 0
        features['high_intent_events'] = 0
    else:
        features['total_events_historical'] = len(hist_events)
        for days in [30, 60, 90]:
            window_start = order_date - pd.Timedelta(days=days)
            features[f'events_last_{days}d'] = len(hist_events[hist_events['timestamp'] >= window_start])
        
        high_intent = ['ACTION_ADD_ITEM_TO_CART', 'ACTION_VIEW_PRODUCT', 'ACTION_VOUCHER_APPLY']
        features['high_intent_events'] = len(hist_events[hist_events['event'].isin(high_intent)])
    
    # Subscription features
    features['has_subscription'] = 1 if len(hist_subs) > 0 else 0
    features['total_subscriptions'] = len(hist_subs)
    
    # Voucher features
    features['total_vouchers_used'] = len(hist_vouchers)
    features['voucher_user'] = 1 if len(hist_vouchers) > 0 else 0
    
    # User lifecycle features
    if len(user_info) > 0:
        user_row = user_info.iloc[0]
        if pd.notna(user_row.get('date_signed_up')):
            features['customer_age_days'] = (order_date - user_row['date_signed_up']).days
        else:
            features['customer_age_days'] = 0
        features['is_singapore'] = 1 if user_row.get('country') == 'Singapore' else 0
        features['is_active_account'] = 1 if user_row.get('active') == 'Yes' else 0
    else:
        features['customer_age_days'] = 0
        features['is_singapore'] = 0
        features['is_active_account'] = 0
    
    return features


def calculate_order_features(date_orders, historical_orders, historical_events,
                              historical_subs, users_df, historical_vouchers, order_date):
    """
    Calculate features for orders on a specific date.
    
    Args:
        date_orders: Orders on this date
        historical_orders: Orders before this date
        historical_events: Events before this date
        historical_subs: Subscriptions before this date
        users_df: Users DataFrame
        historical_vouchers: Voucher applications before this date
        order_date: The order date
        
    Returns:
        pd.DataFrame: Features for all orders on this date
    """
    user_ids = date_orders['user_id'].unique()
    
    # Filter to only relevant users
    user_orders = historical_orders[historical_orders['user_id'].isin(user_ids)]
    user_events = historical_events[historical_events['user_id'].isin(user_ids)]
    user_subs = historical_subs[historical_subs['user_id'].isin(user_ids)]
    user_vouchers = historical_vouchers[historical_vouchers['user_id'].isin(user_ids)]
    
    # === PURCHASE HISTORY FEATURES (BEFORE THIS ORDER) ===
    purchase_features = calculate_historical_purchase_features(user_orders, user_ids, order_date)
    
    # === CURRENT ORDER FEATURES ===
    current_order_features = date_orders[['user_id', 'id', 'date_placed', 'week_start', 
                                          'total_incl_tax', 'shipping_method', 
                                          'will_repurchase']].copy()
    current_order_features.columns = ['user_id', 'order_id', 'order_date', 'week_start',
                                       'current_order_value', 'current_shipping_method',
                                       'will_repurchase']
    
    # === ENGAGEMENT FEATURES ===
    engagement_features = calculate_historical_engagement_features(user_events, user_ids, order_date)
    
    # === SUBSCRIPTION FEATURES ===
    subscription_features = calculate_historical_subscription_features(user_subs, user_ids, order_date)
    
    # === VOUCHER FEATURES ===
    voucher_features = calculate_historical_voucher_features(user_vouchers, user_ids, order_date)
    
    # === CUSTOMER LIFECYCLE FEATURES ===
    lifecycle_features = calculate_lifecycle_features(users_df, user_ids, order_date)
    
    # Merge all features
    features = current_order_features.copy()
    
    for feat_df in [purchase_features, engagement_features, subscription_features, 
                    voucher_features, lifecycle_features]:
        if feat_df is not None and len(feat_df) > 0:
            features = features.merge(feat_df, on='user_id', how='left')
    
    return features


def calculate_historical_purchase_features(orders_df, user_ids, order_date):
    """Calculate purchase history features BEFORE current order."""
    features = pd.DataFrame({'user_id': user_ids})
    
    if len(orders_df) == 0:
        # No historical orders - this is first order
        features['is_first_order'] = 1
        features['previous_orders_count'] = 0
        features['days_since_last_order'] = 9999
        features['avg_order_value_historical'] = 0
        features['total_revenue_historical'] = 0
        features['customer_tenure_days'] = 0
        features['avg_days_between_orders'] = 0
        return features
    
    # Basic aggregations
    order_agg = orders_df.groupby('user_id').agg({
        'id': 'count',
        'total_incl_tax': ['sum', 'mean', 'max', 'min', 'std'],
        'date_placed': ['min', 'max']
    }).reset_index()
    order_agg.columns = ['user_id', 'previous_orders_count', 'total_revenue_historical', 
                         'avg_order_value_historical', 'max_order_value_historical', 
                         'min_order_value_historical', 'std_order_value_historical',
                         'first_order_date', 'last_order_date']
    
    features = features.merge(order_agg, on='user_id', how='left')
    
    # Is this first order?
    features['is_first_order'] = features['previous_orders_count'].isna().astype(int)
    
    # Recency features
    features['days_since_last_order'] = (order_date - features['last_order_date']).dt.days
    features['customer_tenure_days'] = (order_date - features['first_order_date']).dt.days
    
    # Frequency features
    features['avg_days_between_orders'] = features['customer_tenure_days'] / (features['previous_orders_count'] + 1)
    
    # Rolling window features (last 30, 60, 90 days BEFORE this order)
    for window_days in [30, 60, 90]:
        window_start = order_date - pd.Timedelta(days=window_days)
        window_orders = orders_df[orders_df['date_placed'] >= window_start]
        
        window_agg = window_orders.groupby('user_id').agg({
            'id': 'count',
            'total_incl_tax': 'sum'
        }).reset_index()
        window_agg.columns = ['user_id', f'orders_last_{window_days}d', f'spend_last_{window_days}d']
        
        features = features.merge(window_agg, on='user_id', how='left')
    
    # Drop intermediate date columns
    features = features.drop(columns=['first_order_date', 'last_order_date'], errors='ignore')
    
    return features


def calculate_historical_engagement_features(events_df, user_ids, order_date):
    """Calculate engagement features BEFORE current order."""
    features = pd.DataFrame({'user_id': user_ids})
    
    if len(events_df) == 0:
        features['total_events_historical'] = 0
        features['events_last_30d'] = 0
        return features
    
    # Basic event counts
    event_agg = events_df.groupby('user_id').agg({
        'event': ['count', 'nunique'],
        'timestamp': 'max'
    }).reset_index()
    event_agg.columns = ['user_id', 'total_events_historical', 'unique_event_types', 'last_event_date']
    
    features = features.merge(event_agg, on='user_id', how='left')
    
    # Days since last event
    features['days_since_last_event'] = (order_date - features['last_event_date']).dt.days
    
    # Rolling window events
    for window_days in [30, 60, 90]:
        window_start = order_date - pd.Timedelta(days=window_days)
        window_events = events_df[events_df['timestamp'] >= window_start]
        
        window_agg = window_events.groupby('user_id')['event'].count().reset_index()
        window_agg.columns = ['user_id', f'events_last_{window_days}d']
        
        features = features.merge(window_agg, on='user_id', how='left')
    
    # High intent events
    high_intent = ['ACTION_ADD_ITEM_TO_CART', 'ACTION_VIEW_PRODUCT', 'ACTION_VOUCHER_APPLY']
    high_intent_events = events_df[events_df['event'].isin(high_intent)]
    
    hi_agg = high_intent_events.groupby('user_id')['event'].count().reset_index()
    hi_agg.columns = ['user_id', 'high_intent_events']
    features = features.merge(hi_agg, on='user_id', how='left')
    
    # Drop intermediate columns
    features = features.drop(columns=['last_event_date'], errors='ignore')
    
    return features


def calculate_historical_subscription_features(subs_df, user_ids, order_date):
    """Calculate subscription features BEFORE current order."""
    features = pd.DataFrame({'user_id': user_ids})
    
    if len(subs_df) == 0:
        features['has_subscription'] = 0
        features['total_subscriptions'] = 0
        return features
    
    # Total subscriptions
    sub_counts = subs_df.groupby('user_id').size().reset_index(name='total_subscriptions')
    features = features.merge(sub_counts, on='user_id', how='left')
    
    # Active subscriptions
    if 'status' in subs_df.columns:
        active_subs = subs_df[subs_df['status'] == 'Active'].groupby('user_id').size().reset_index(name='active_subscriptions')
        features = features.merge(active_subs, on='user_id', how='left')
    
    features['has_subscription'] = (features['total_subscriptions'].fillna(0) > 0).astype(int)
    
    return features


def calculate_historical_voucher_features(voucher_df, user_ids, order_date):
    """Calculate voucher features BEFORE current order."""
    features = pd.DataFrame({'user_id': user_ids})
    
    if len(voucher_df) == 0:
        features['total_vouchers_used'] = 0
        features['voucher_user'] = 0
        return features
    
    # Total vouchers used
    voucher_counts = voucher_df.groupby('user_id').agg({
        'voucher_id': ['count', 'nunique'],
        'created': 'max'
    }).reset_index()
    voucher_counts.columns = ['user_id', 'total_vouchers_used', 'unique_vouchers', 'last_voucher_date']
    
    features = features.merge(voucher_counts, on='user_id', how='left')
    
    # Days since last voucher
    features['days_since_last_voucher'] = (order_date - features['last_voucher_date']).dt.days
    
    # Is voucher user
    features['voucher_user'] = (features['total_vouchers_used'].fillna(0) > 0).astype(int)
    
    # Drop intermediate columns
    features = features.drop(columns=['last_voucher_date'], errors='ignore')
    
    return features


def calculate_lifecycle_features(users_df, user_ids, order_date):
    """Calculate customer lifecycle features."""
    features = pd.DataFrame({'user_id': user_ids})
    
    # Merge with users
    user_info = users_df[users_df['id'].isin(user_ids)][['id', 'date_signed_up', 'country', 'active']].copy()
    user_info.columns = ['user_id', 'signup_date', 'country', 'is_active']
    
    features = features.merge(user_info, on='user_id', how='left')
    
    # Customer age
    features['customer_age_days'] = (order_date - features['signup_date']).dt.days
    
    # Is Singapore customer
    features['is_singapore'] = (features['country'] == 'Singapore').astype(int)
    
    # Is active account
    features['is_active_account'] = (features['is_active'] == 'Yes').astype(int)
    
    # Drop intermediate columns
    features = features.drop(columns=['signup_date', 'country', 'is_active'], errors='ignore')
    
    return features


# ============================================================================
# TEMPORAL SPLIT
# ============================================================================

def quantile_train_test_split(df, test_quantile=0.8):
    """
    Split by week_start quantile (no data leakage since features are calculated per-order).
    
    Since each order's features use only historical data BEFORE that order,
    we can safely use a quantile-based split without temporal leakage.
    
    Args:
        df: Features DataFrame
        test_quantile: Quantile cutoff (0.8 = use last 20% for test)
        
    Returns:
        tuple: (train_df, test_df)
    """
    logger.info("="*80)
    logger.info("QUANTILE-BASED TRAIN/TEST SPLIT")
    logger.info("="*80)
    logger.info(f"Using {test_quantile:.0%} quantile for train/test split")
    logger.info("No data leakage: features calculated from historical data per order")
    
    # Find the quantile cutoff date
    cutoff_date = df['week_start'].quantile(test_quantile)
    
    train_df = df[df['week_start'] <= cutoff_date].copy()
    test_df = df[df['week_start'] > cutoff_date].copy()
    
    logger.info(f"\nCutoff date (week_start): {cutoff_date.date()}")
    
    logger.info(f"\nTraining set: {len(train_df):,} orders ({len(train_df)/len(df):.1%})")
    logger.info(f"  Date range: {train_df['order_date'].min().date()} to {train_df['order_date'].max().date()}")
    logger.info(f"  Positive rate: {train_df['will_repurchase'].mean():.1%}")
    logger.info(f"  Unique users: {train_df['user_id'].nunique():,}")
    
    logger.info(f"\nTest set: {len(test_df):,} orders ({len(test_df)/len(df):.1%})")
    logger.info(f"  Date range: {test_df['order_date'].min().date()} to {test_df['order_date'].max().date()}")
    logger.info(f"  Positive rate: {test_df['will_repurchase'].mean():.1%}")
    logger.info(f"  Unique users: {test_df['user_id'].nunique():,}")
    
    return train_df, test_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_order_based_feature_engineering(prediction_weeks=12, output_dir=OUTPUT_DIR):
    """
    Run the complete order-based feature engineering pipeline.
    
    Args:
        prediction_weeks: Number of weeks to look forward for target
        output_dir: Directory to save outputs
        
    Returns:
        pd.DataFrame: Complete order-based feature dataset
    """
    logger.info("="*80)
    logger.info("ORDER-BASED FEATURE ENGINEERING")
    logger.info("="*80)
    logger.info(f"Prediction window: {prediction_weeks} weeks")
    logger.info("Approach: One entry per order, predicting repurchase")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*80)
    datasets = load_all_datasets()
    
    # Step 2: Create order-based entries
    logger.info("\n" + "="*80)
    logger.info("STEP 2: CREATING ORDER-BASED ENTRIES")
    logger.info("="*80)
    order_entries = create_order_based_entries(
        datasets['orders'],
        prediction_weeks=prediction_weeks
    )
    
    # Step 3: Calculate features
    logger.info("\n" + "="*80)
    logger.info("STEP 3: CALCULATING FEATURES AT ORDER TIME")
    logger.info("="*80)
    features_df = calculate_features_at_order_time(
        order_entries,
        datasets['orders'],
        datasets['events'],
        datasets['subscriptions'],
        datasets['users'],
        datasets['voucher_applications']
    )
    
    # Step 4: Handle missing values
    logger.info("\n" + "="*80)
    logger.info("STEP 4: HANDLING MISSING VALUES")
    logger.info("="*80)
    
    # Fill numeric columns with 0
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
    
    # Replace infinite values
    features_df = features_df.replace([np.inf, -np.inf], 0)
    
    logger.info(f"✓ Missing values handled")
    logger.info(f"  Final shape: {features_df.shape}")
    
    # Step 5: Quantile-based split
    logger.info("\n" + "="*80)
    logger.info("STEP 5: QUANTILE-BASED TRAIN/TEST SPLIT")
    logger.info("="*80)
    train_df, test_df = quantile_train_test_split(features_df, test_quantile=0.8)
    
    # Step 6: Save features
    logger.info("\n" + "="*80)
    logger.info("STEP 6: SAVING FEATURES")
    logger.info("="*80)
    
    output_path = f"{output_dir}/order_based_features.csv"
    features_df.to_csv(output_path, index=False)
    
    logger.info(f"✓ Saved to: {output_path}")
    logger.info(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Save train/test splits
    train_df.to_csv(f"{output_dir}/order_based_train.csv", index=False)
    test_df.to_csv(f"{output_dir}/order_based_test.csv", index=False)
    logger.info(f"✓ Saved train/test splits")
    
    # Print summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total orders: {len(features_df):,}")
    logger.info(f"Unique users: {features_df['user_id'].nunique():,}")
    logger.info(f"Date range: {features_df['order_date'].min().date()} to {features_df['order_date'].max().date()}")
    logger.info(f"Features: {len(features_df.columns) - 7}")  # Exclude metadata columns
    
    overall_positive_rate = features_df['will_repurchase'].mean()
    logger.info(f"\nOverall class distribution:")
    logger.info(f"  Will repurchase: {features_df['will_repurchase'].sum():,} ({overall_positive_rate:.1%})")
    logger.info(f"  Won't repurchase: {(1-features_df['will_repurchase']).sum():,.0f} ({1-overall_positive_rate:.1%})")
    
    train_positive_rate = train_df['will_repurchase'].mean()
    logger.info(f"\nTraining set class distribution:")
    logger.info(f"  Will repurchase: {train_df['will_repurchase'].sum():,} ({train_positive_rate:.1%})")
    logger.info(f"  Won't repurchase: {(1-train_df['will_repurchase']).sum():,.0f} ({1-train_positive_rate:.1%})")
    
    test_positive_rate = test_df['will_repurchase'].mean()
    logger.info(f"\nTest set class distribution:")
    logger.info(f"  Will repurchase: {test_df['will_repurchase'].sum():,} ({test_positive_rate:.1%})")
    logger.info(f"  Won't repurchase: {(1-test_df['will_repurchase']).sum():,.0f} ({1-test_positive_rate:.1%})")
    
    # Analysis by order number
    logger.info(f"\n" + "="*80)
    logger.info("REPURCHASE RATE BY ORDER NUMBER")
    logger.info("="*80)
    
    order_number_analysis = features_df.groupby('is_first_order').agg({
        'will_repurchase': ['count', 'sum', 'mean']
    }).reset_index()
    order_number_analysis.columns = ['is_first_order', 'total_orders', 'repurchases', 'repurchase_rate']
    
    logger.info(f"\nFirst-time customers:")
    first_time = order_number_analysis[order_number_analysis['is_first_order'] == 1].iloc[0]
    logger.info(f"  Orders: {first_time['total_orders']:,.0f}")
    logger.info(f"  Repurchase rate: {first_time['repurchase_rate']:.1%}")
    
    logger.info(f"\nRepeat customers:")
    repeat = order_number_analysis[order_number_analysis['is_first_order'] == 0]
    if len(repeat) > 0:
        repeat = repeat.iloc[0]
        logger.info(f"  Orders: {repeat['total_orders']:,.0f}")
        logger.info(f"  Repurchase rate: {repeat['repurchase_rate']:.1%}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ ORDER-BASED FEATURE ENGINEERING COMPLETE")
    logger.info("="*80)
    
    return features_df, train_df, test_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Run the pipeline
    features_df, train_df, test_df = run_order_based_feature_engineering(
        prediction_weeks=12,
        output_dir='outputs/order_based'
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Total orders: {len(features_df):,}")
    print(f"Overall repurchase rate: {features_df['will_repurchase'].mean():.1%}")
    print(f"Training repurchase rate: {train_df['will_repurchase'].mean():.1%}")
    print(f"Test repurchase rate: {test_df['will_repurchase'].mean():.1%}")
