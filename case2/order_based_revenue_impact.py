# ---
# Revenue Impact Analysis (User-Level)
# Analyzes revenue impact of marketing interventions by unique user
# Uses most recent order propensity and user's historical average order value
# ---

import pandas as pd
import numpy as np
import os
import json
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
OUTPUT_DIR = "outputs/order_based"
CLUSTER_PATH = "../cluster01/sg_user.csv"

# Segment names mapping
SEGMENT_NAMES = {
    0: 'Casual Walk-in',
    1: 'Golden Whales',
    2: 'High Potential',
    3: 'Drifting Risk'
}

# Intervention assumptions (realistic for ~$10-20 avg order value)
# Costs: Email = ~$0.50, SMS = ~$1, Personalized offer = ~$2-3
# Lift: Based on typical email marketing benchmarks (5-15% conversion lift)
DEFAULT_INTERVENTION_CONFIG = {
    'High (>60%)': {'cost': 0.50, 'lift': 0.10},      # Light touch: reminder email
    'Medium (30-60%)': {'cost': 1.50, 'lift': 0.15},  # Medium: email + small incentive
    'Low (<30%)': {'cost': 3.00, 'lift': 0.08}        # Heavy: personalized offer/discount
}


# ============================================================================
# USER-LEVEL AGGREGATION
# ============================================================================

def aggregate_to_user_level(test_df, propensity_scores, cluster_df):
    """
    Aggregate order-level data to user-level.
    
    For each user:
    - Use most recent order's propensity score (current state)
    - Use user's historical average order value from cluster data
    - Use actual repurchase rate from test period
    
    Args:
        test_df: Test DataFrame with orders
        propensity_scores: Array of propensity scores (aligned with test_df)
        cluster_df: Cluster data with user segments and historical values
        
    Returns:
        pd.DataFrame: User-level data with propensity and order values
    """
    logger.info("Aggregating to user level...")
    
    # Add propensity scores to test_df
    order_df = test_df.copy()
    order_df['propensity_score'] = propensity_scores
    
    # Sort by date and get most recent order per user
    order_df['order_date'] = pd.to_datetime(order_df['order_date'])
    user_latest = order_df.sort_values('order_date').groupby('user_id').last().reset_index()
    
    logger.info(f"  Orders in test set: {len(order_df):,}")
    logger.info(f"  Unique users: {len(user_latest):,}")
    logger.info(f"  Avg orders per user: {len(order_df) / len(user_latest):.2f}")
    
    # Calculate user's average order value from test period
    user_avg_ov_test = order_df.groupby('user_id')['current_order_value'].mean().reset_index()
    user_avg_ov_test.columns = ['user_id', 'user_avg_order_value_test']
    
    # Calculate user's actual repurchase rate in test period
    user_repurchase = order_df.groupby('user_id')['will_repurchase'].mean().reset_index()
    user_repurchase.columns = ['user_id', 'user_repurchase_rate']
    
    # Merge with user latest
    user_df = user_latest[['user_id', 'propensity_score', 'current_order_value', 'will_repurchase']].copy()
    user_df = user_df.merge(user_avg_ov_test, on='user_id', how='left')
    user_df = user_df.merge(user_repurchase, on='user_id', how='left')
    
    # Recalculate correct avg_order_value from cluster data
    # (The original avg_order_value is wrong due to normalized frequency_orders)
    cluster_df = cluster_df.copy()
    cluster_df['correct_avg_order_value'] = cluster_df['monetary_total'] / cluster_df['order_count']
    
    # Merge with cluster data
    user_df = user_df.merge(
        cluster_df[['user_id', 'value_cluster', 'order_count', 'monetary_total', 'correct_avg_order_value']], 
        on='user_id', 
        how='left'
    )
    
    # Use test period avg if available, else use historical
    user_df['avg_order_value'] = user_df['user_avg_order_value_test'].fillna(
        user_df['correct_avg_order_value']
    )
    
    # Map segment names
    user_df['segment_name'] = user_df['value_cluster'].map(SEGMENT_NAMES)
    
    # Create propensity segments
    user_df['propensity_segment'] = pd.cut(
        user_df['propensity_score'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low (<30%)', 'Medium (30-60%)', 'High (>60%)']
    )
    
    logger.info(f"  ✓ User-level aggregation complete: {len(user_df):,} users")
    
    return user_df, order_df


# ============================================================================
# REVENUE IMPACT CALCULATION
# ============================================================================

def calculate_revenue_impact(user_df, intervention_config=None):
    """
    Calculate revenue impact by segment × propensity at user level.
    
    Args:
        user_df: User-level DataFrame
        intervention_config: Dict with cost and lift per propensity segment
        
    Returns:
        pd.DataFrame: Revenue impact results by segment × propensity
    """
    if intervention_config is None:
        intervention_config = DEFAULT_INTERVENTION_CONFIG
    
    logger.info("Calculating revenue impact by segment × propensity...")
    
    results = []
    
    for segment_name in SEGMENT_NAMES.values():
        for prop_segment in ['Low (<30%)', 'Medium (30-60%)', 'High (>60%)']:
            mask = (user_df['segment_name'] == segment_name) & (user_df['propensity_segment'] == prop_segment)
            segment_data = user_df[mask]
            
            if len(segment_data) == 0:
                continue
            
            n_users = len(segment_data)
            current_repurchase_rate = segment_data['user_repurchase_rate'].mean()
            
            # Use MEDIAN of user's average order value (robust to outliers)
            median_order_value = segment_data['avg_order_value'].median()
            mean_order_value = segment_data['avg_order_value'].mean()
            
            # Intervention impact
            config = intervention_config[prop_segment]
            intervention_cost = n_users * config['cost']
            lift = config['lift']
            
            # Additional repurchases = users × lift probability
            additional_repurchases = n_users * lift
            additional_revenue = additional_repurchases * median_order_value
            
            net_profit = additional_revenue - intervention_cost
            roi = (net_profit / intervention_cost * 100) if intervention_cost > 0 else 0
            
            results.append({
                'value_cluster': segment_name,
                'propensity_segment': prop_segment,
                'n_users': n_users,
                'current_repurchase_rate': current_repurchase_rate,
                'median_order_value': median_order_value,
                'mean_order_value': mean_order_value,
                'intervention_cost_per_user': config['cost'],
                'intervention_cost_total': intervention_cost,
                'expected_lift': lift,
                'additional_repurchases': additional_repurchases,
                'additional_revenue': additional_revenue,
                'net_profit': net_profit,
                'roi_percent': roi
            })
    
    results_df = pd.DataFrame(results)
    logger.info(f"  ✓ Calculated impact for {len(results_df)} segment × propensity combinations")
    
    return results_df


# ============================================================================
# SUMMARY AND REPORTING
# ============================================================================

def generate_summary(results_df, user_df, order_df):
    """
    Generate comprehensive summary of revenue impact analysis.
    
    Args:
        results_df: Revenue impact results
        user_df: User-level data
        order_df: Original order-level data
        
    Returns:
        dict: Summary statistics
    """
    logger.info("\n" + "="*80)
    logger.info("REVENUE IMPACT SUMMARY (USER-LEVEL)")
    logger.info("="*80)
    
    # Test period context
    test_start = order_df['order_date'].min()
    test_end = order_df['order_date'].max()
    test_days = (test_end - test_start).days
    test_revenue = order_df['current_order_value'].sum()
    
    logger.info(f"\nTest Period Context:")
    logger.info(f"  Period: {test_start.date()} to {test_end.date()} ({test_days} days, ~{test_days/30:.1f} months)")
    logger.info(f"  Total Orders: {len(order_df):,}")
    logger.info(f"  Unique Users: {len(user_df):,}")
    logger.info(f"  Total Revenue in Period: ${test_revenue:,.2f}")
    logger.info(f"  Median Order Value: ${order_df['current_order_value'].median():.2f}")
    logger.info(f"  Avg Orders per User: {len(order_df) / len(user_df):.2f}")
    
    # Overall impact (all segments)
    total_users = results_df['n_users'].sum()
    total_cost = results_df['intervention_cost_total'].sum()
    total_revenue = results_df['additional_revenue'].sum()
    total_profit = results_df['net_profit'].sum()
    overall_roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
    
    logger.info(f"\nIntervention Impact (if targeting ALL users):")
    logger.info(f"  Total Users Targeted: {total_users:,.0f}")
    logger.info(f"  Total Investment: ${total_cost:,.2f}")
    logger.info(f"  Expected Additional Revenue: ${total_revenue:,.2f}")
    logger.info(f"  Expected Net Profit: ${total_profit:,.2f}")
    logger.info(f"  Overall ROI: {overall_roi:.1f}%")
    logger.info(f"  Profit as % of Period Revenue: {total_profit/test_revenue*100:.2f}%")
    
    # Smart targeting (only profitable segments)
    profitable_segments = results_df[results_df['roi_percent'] > 0]
    smart_summary = None
    
    if len(profitable_segments) > 0:
        smart_users = profitable_segments['n_users'].sum()
        smart_cost = profitable_segments['intervention_cost_total'].sum()
        smart_revenue = profitable_segments['additional_revenue'].sum()
        smart_profit = profitable_segments['net_profit'].sum()
        smart_roi = (smart_profit / smart_cost * 100) if smart_cost > 0 else 0
        
        logger.info(f"\nSmart Targeting (only profitable segments with ROI > 0%):")
        logger.info(f"  Profitable Segments: {len(profitable_segments)}")
        logger.info(f"  Users to Target: {smart_users:,.0f} ({smart_users/total_users*100:.1f}% of all users)")
        logger.info(f"  Total Investment: ${smart_cost:,.2f}")
        logger.info(f"  Expected Additional Revenue: ${smart_revenue:,.2f}")
        logger.info(f"  Expected Net Profit: ${smart_profit:,.2f}")
        logger.info(f"  ROI: {smart_roi:.1f}%")
        
        smart_summary = {
            'profitable_segments': len(profitable_segments),
            'users_to_target': int(smart_users),
            'pct_of_all_users': float(smart_users/total_users*100),
            'total_investment': float(smart_cost),
            'expected_revenue': float(smart_revenue),
            'expected_profit': float(smart_profit),
            'roi_percent': float(smart_roi)
        }
    
    # Top opportunities
    logger.info("\n" + "-"*80)
    logger.info("TOP 5 OPPORTUNITIES (by Net Profit):")
    logger.info("-"*80)
    
    top_5 = results_df.nlargest(5, 'net_profit')
    for idx, row in top_5.iterrows():
        logger.info(f"\n{row['value_cluster']} → {row['propensity_segment']}")
        logger.info(f"  Users: {row['n_users']:,.0f}")
        logger.info(f"  Median Order Value: ${row['median_order_value']:.2f}")
        logger.info(f"  Investment: ${row['intervention_cost_total']:,.2f}")
        logger.info(f"  Expected Return: ${row['additional_revenue']:,.2f}")
        logger.info(f"  Net Profit: ${row['net_profit']:,.2f}")
        logger.info(f"  ROI: {row['roi_percent']:.1f}%")
    
    # Build summary dict
    summary = {
        'analysis_type': 'user_level',
        'test_period': {
            'start': str(test_start.date()),
            'end': str(test_end.date()),
            'days': int(test_days),
            'total_orders': int(len(order_df)),
            'unique_users': int(len(user_df)),
            'total_revenue': float(test_revenue),
            'median_order_value': float(order_df['current_order_value'].median())
        },
        'all_segments': {
            'total_users': int(total_users),
            'total_investment': float(total_cost),
            'expected_revenue': float(total_revenue),
            'expected_profit': float(total_profit),
            'roi_percent': float(overall_roi)
        },
        'smart_targeting': smart_summary,
        'top_5_opportunities': top_5[['value_cluster', 'propensity_segment', 
                                       'n_users', 'net_profit', 'roi_percent']].to_dict('records')
    }
    
    return summary


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_revenue_impact(results_df, output_path, intervention_config=None):
    """Create revenue impact visualizations."""
    logger.info("\nCreating revenue impact visualizations...")
    
    if intervention_config is None:
        intervention_config = DEFAULT_INTERVENTION_CONFIG
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Dynamic suptitle with actual intervention assumptions
    high_cfg = intervention_config['High (>60%)']
    med_cfg = intervention_config['Medium (30-60%)']
    low_cfg = intervention_config['Low (<30%)']
    
    subtitle = (f"Assumptions: High prop: ${high_cfg['cost']}/user, {high_cfg['lift']:.0%} lift | "
                f"Medium: ${med_cfg['cost']}/user, {med_cfg['lift']:.0%} lift | "
                f"Low: ${low_cfg['cost']}/user, {low_cfg['lift']:.0%} lift")
    fig.suptitle('Revenue Impact Analysis (User-Level)\n' + subtitle, fontsize=11, y=1.02)
    
    # 2. Net Profit by Segment × Propensity
    pivot_profit = results_df.pivot_table(
        values='net_profit', 
        index='value_cluster', 
        columns='propensity_segment'
    )
    
    ax1 = axes[0, 0]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']  # Red, Orange, Green for Low, Medium, High
    pivot_profit.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Expected Net Profit by Segment & Propensity\n(User-Level Analysis)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Customer Segment')
    ax1.set_ylabel('Net Profit ($)')
    ax1.legend(title='Propensity', bbox_to_anchor=(1.05, 1))
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 3. ROI by Segment × Propensity
    pivot_roi = results_df.pivot_table(
        values='roi_percent', 
        index='value_cluster', 
        columns='propensity_segment'
    )
    
    ax2 = axes[0, 1]
    pivot_roi.plot(kind='bar', ax=ax2, color=colors)
    ax2.set_title('Expected ROI (%) by Segment & Propensity', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Customer Segment')
    ax2.set_ylabel('ROI (%)')
    ax2.legend(title='Propensity', bbox_to_anchor=(1.05, 1))
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 4. Investment vs Return scatter - with segment labels
    ax3 = axes[1, 0]
    prop_colors = {'Low (<30%)': '#e74c3c', 'Medium (30-60%)': '#f39c12', 'High (>60%)': '#2ecc71'}
    segment_markers = {'Casual Walk-in': 'o', 'Golden Whales': 's', 'High Potential': '^', 'Drifting Risk': 'D'}
    
    # Plot each point with marker by segment, color by propensity
    for _, row in results_df.iterrows():
        ax3.scatter(
            row['intervention_cost_total'], 
            row['additional_revenue'],
            s=150,  # Fixed size for clarity
            alpha=0.7, 
            color=prop_colors.get(row['propensity_segment'], '#3498db'),
            marker=segment_markers.get(row['value_cluster'], 'o'),
            edgecolors='black',
            linewidths=0.5
        )
    
    # Break-even line
    max_val = max(results_df['intervention_cost_total'].max(), results_df['additional_revenue'].max())
    ax3.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3)
    
    # Create custom legend with small, uniform markers
    # Propensity legend (colors)
    prop_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=p) 
                    for p, c in prop_colors.items()]
    # Segment legend (markers)
    seg_handles = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='gray', markersize=8, label=s) 
                   for s, m in segment_markers.items() if s in results_df['value_cluster'].values]
    # Break-even line
    breakeven_handle = plt.Line2D([0], [0], linestyle='--', color='black', alpha=0.3, label='Break-even')
    
    # Combine legends
    legend1 = ax3.legend(handles=prop_handles, title='Propensity', loc='upper left', fontsize=8)
    ax3.add_artist(legend1)
    ax3.legend(handles=seg_handles + [breakeven_handle], title='Segment', loc='lower right', fontsize=8)
    
    ax3.set_title('Investment vs Expected Return', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Total Investment ($)')
    ax3.set_ylabel('Expected Additional Revenue ($)')
    ax3.grid(True, alpha=0.3)
    
    # 5. Top opportunities bar chart with proper margins
    ax4 = axes[1, 1]
    top_10 = results_df.nlargest(10, 'net_profit').copy()
    top_10['label'] = top_10['value_cluster'].str[:12] + '\n' + top_10['propensity_segment']
    
    bar_colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_10['net_profit']]
    bars = ax4.barh(range(len(top_10)), top_10['net_profit'], color=bar_colors)
    ax4.set_yticks(range(len(top_10)))
    ax4.set_yticklabels(top_10['label'], fontsize=9)
    ax4.set_xlabel('Net Profit ($)')
    ax4.set_title('Top Opportunities by Net Profit\n(Green = Profitable, Red = Loss)', fontsize=13, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Calculate proper x-axis limits to fit labels
    min_profit = top_10['net_profit'].min()
    max_profit = top_10['net_profit'].max()
    x_margin = max(abs(min_profit), abs(max_profit)) * 0.25  # 25% margin for labels
    ax4.set_xlim(min_profit - x_margin, max_profit + x_margin)
    
    # Add value labels with proper positioning
    for i, (bar, val) in enumerate(zip(bars, top_10['net_profit'])):
        # Position label outside the bar with small offset
        offset = x_margin * 0.1
        x_pos = val + offset if val >= 0 else val - offset
        ax4.text(x_pos, i, f'${val:,.0f}', va='center', fontsize=8,
                ha='left' if val >= 0 else 'right')
    
    plt.tight_layout()
    
    file_path = f"{output_path}/revenue_impact_analysis.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved visualization to: {file_path}")
    
    return file_path


def plot_user_distribution(user_df, output_path):
    """Plot user distribution across segments and propensity levels."""
    logger.info("Creating user distribution visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Users by segment
    ax1 = axes[0, 0]
    segment_counts = user_df['segment_name'].value_counts()
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    segment_counts.plot(kind='bar', ax=ax1, color=colors[:len(segment_counts)])
    ax1.set_title('Users by Customer Segment', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Segment')
    ax1.set_ylabel('Number of Users')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels
    for i, v in enumerate(segment_counts):
        ax1.text(i, v + 50, f'{v:,}', ha='center', fontsize=10)
    
    # 2. Users by propensity segment
    ax2 = axes[0, 1]
    prop_counts = user_df['propensity_segment'].value_counts()
    prop_colors = {'Low (<30%)': '#e74c3c', 'Medium (30-60%)': '#f39c12', 'High (>60%)': '#2ecc71'}
    prop_counts.plot(kind='bar', ax=ax2, color=[prop_colors.get(x, '#3498db') for x in prop_counts.index])
    ax2.set_title('Users by Propensity Level', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Propensity Segment')
    ax2.set_ylabel('Number of Users')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add count labels
    for i, v in enumerate(prop_counts):
        ax2.text(i, v + 50, f'{v:,}', ha='center', fontsize=10)
    
    # 3. Heatmap: Users by Segment × Propensity
    ax3 = axes[1, 0]
    pivot_users = pd.crosstab(user_df['segment_name'], user_df['propensity_segment'])
    # Reorder columns
    col_order = ['Low (<30%)', 'Medium (30-60%)', 'High (>60%)']
    pivot_users = pivot_users.reindex(columns=[c for c in col_order if c in pivot_users.columns])
    
    sns.heatmap(pivot_users, annot=True, fmt='d', cmap='YlOrRd', ax=ax3, 
                cbar_kws={'label': 'Number of Users'})
    ax3.set_title('User Count by Segment × Propensity', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Propensity Segment')
    ax3.set_ylabel('Customer Segment')
    
    # 4. Stacked bar: Segment breakdown by propensity
    ax4 = axes[1, 1]
    pivot_users_pct = pivot_users.div(pivot_users.sum(axis=1), axis=0) * 100
    pivot_users_pct.plot(kind='barh', stacked=True, ax=ax4, 
                         color=['#e74c3c', '#f39c12', '#2ecc71'])
    ax4.set_title('Propensity Distribution within Each Segment', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Percentage of Users')
    ax4.set_ylabel('Customer Segment')
    ax4.legend(title='Propensity', bbox_to_anchor=(1.05, 1))
    
    # Add percentage labels
    for i, (idx, row) in enumerate(pivot_users_pct.iterrows()):
        cumsum = 0
        for j, val in enumerate(row):
            if val > 5:  # Only show label if > 5%
                ax4.text(cumsum + val/2, i, f'{val:.0f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            cumsum += val
    
    plt.tight_layout()
    
    file_path = f"{output_path}/user_distribution.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved user distribution to: {file_path}")
    
    return file_path


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_revenue_impact_analysis(test_df, propensity_scores, cluster_path=CLUSTER_PATH, 
                                 output_dir=OUTPUT_DIR, intervention_config=None):
    """
    Run complete user-level revenue impact analysis.
    
    Args:
        test_df: Test DataFrame with orders
        propensity_scores: Array of propensity scores (aligned with test_df)
        cluster_path: Path to cluster data CSV
        output_dir: Directory to save outputs
        intervention_config: Optional custom intervention config
        
    Returns:
        tuple: (results_df, summary, user_df)
    """
    logger.info("\n" + "="*80)
    logger.info("REVENUE IMPACT ANALYSIS (USER-LEVEL)")
    logger.info("="*80)
    
    # Load cluster data
    try:
        cluster_df = pd.read_csv(cluster_path)
        if 'user_id' not in cluster_df.columns:
            cluster_df = cluster_df.rename(columns={'id': 'user_id'})
        logger.info(f"✓ Loaded cluster data: {len(cluster_df):,} users")
    except FileNotFoundError:
        logger.error(f"  ⚠️ Cluster data not found at {cluster_path}")
        return None, None, None
    
    # Aggregate to user level
    user_df, order_df = aggregate_to_user_level(test_df, propensity_scores, cluster_df)
    
    # Calculate revenue impact
    results_df = calculate_revenue_impact(user_df, intervention_config)
    
    # Generate summary
    summary = generate_summary(results_df, user_df, order_df)
    
    # Save outputs
    output_path = f"{output_dir}/revenue_impact"
    os.makedirs(output_path, exist_ok=True)
    
    # Save detailed results
    results_df.to_csv(f"{output_path}/revenue_impact_detailed.csv", index=False)
    logger.info(f"\n✓ Saved detailed results to: {output_path}/revenue_impact_detailed.csv")
    
    # Save user-level data
    user_df.to_csv(f"{output_path}/user_level_data.csv", index=False)
    logger.info(f"✓ Saved user-level data to: {output_path}/user_level_data.csv")
    
    # Save summary
    with open(f"{output_path}/revenue_impact_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Saved summary to: {output_path}/revenue_impact_summary.json")
    
    # Create visualizations (pass intervention_config for dynamic subtitle)
    actual_config = intervention_config if intervention_config else DEFAULT_INTERVENTION_CONFIG
    plot_revenue_impact(results_df, output_path, actual_config)
    plot_user_distribution(user_df, output_path)
    
    logger.info("\n" + "="*80)
    logger.info("✓ REVENUE IMPACT ANALYSIS COMPLETE")
    logger.info("="*80)
    
    return results_df, summary, user_df


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    import joblib
    
    print("="*80)
    print("REVENUE IMPACT ANALYSIS (USER-LEVEL) - Standalone Mode")
    print("="*80)
    
    # Load test data
    test_df = pd.read_csv(f"{OUTPUT_DIR}/order_based_test.csv")
    print(f"✓ Loaded test data: {len(test_df):,} orders")
    
    # Load model and get predictions
    model_path = f"{OUTPUT_DIR}/models/order_based_best_model.pkl"
    model = joblib.load(model_path)
    print(f"✓ Loaded model from: {model_path}")
    
    # Prepare features
    from order_based_model import prepare_features
    X_test, y_test, feature_names = prepare_features(test_df)
    
    # Get propensity scores
    propensity_scores = model.predict_proba(X_test)[:, 1]
    print(f"✓ Generated propensity scores for {len(propensity_scores):,} orders")
    
    # Run analysis
    results_df, summary, user_df = run_revenue_impact_analysis(
        test_df, 
        propensity_scores,
        cluster_path=CLUSTER_PATH,
        output_dir=OUTPUT_DIR
    )
    
    print("\n✓ Analysis complete! Check outputs/order_based/revenue_impact/")
