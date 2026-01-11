# ---
# Visualization Functions for Case 2: Propensity Prediction
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
PLOTS_PATH = "outputs/plots"
FIGSIZE_STANDARD = (10, 8)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_GRID = (16, 12)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# MODEL EVALUATION PLOTS
# ============================================================================

def plot_roc_curves(results, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot ROC curves for all models on one plot.
    
    Args:
        results: Dictionary of evaluation results with roc_curve data
        target_name: Name of target variable
        save_path: Path to save plot
    """
    logger.info("Creating ROC curves plot...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for (model_name, metrics), color in zip(results.items(), colors):
        fpr = metrics['roc_curve']['fpr']
        tpr = metrics['roc_curve']['tpr']
        auc = metrics['roc_auc']
        
        ax.plot(fpr, tpr, color=color, lw=2, 
                label=f'{model_name} (AUC = {auc:.4f})')
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_roc_curves.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved ROC curves to: {file_path}")
    
    return file_path


def plot_pr_curves(results, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot Precision-Recall curves for all models on one plot.
    
    Args:
        results: Dictionary of evaluation results with pr_curve data
        target_name: Name of target variable
        save_path: Path to save plot
    """
    logger.info("Creating Precision-Recall curves plot...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for (model_name, metrics), color in zip(results.items(), colors):
        precision = metrics['pr_curve']['precision']
        recall = metrics['pr_curve']['recall']
        pr_auc = metrics['pr_auc']
        
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{model_name} (PR-AUC = {pr_auc:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_pr_curves.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved PR curves to: {file_path}")
    
    return file_path


def plot_metric_comparison(comparison_df, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot bar chart comparing all 6 metrics across models.
    
    Args:
        comparison_df: DataFrame with model comparison metrics
        target_name: Name of target variable
        save_path: Path to save plot
    """
    logger.info("Creating metric comparison plot...")
    
    # Prepare data for plotting
    metrics_to_plot = ['Test Accuracy', 'Test Precision', 'Test Recall', 
                       'Test F1', 'Test ROC-AUC', 'Test PR-AUC']
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    x = np.arange(len(comparison_df))
    width = 0.12
    multiplier = 0
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_to_plot)))
    
    for metric, color in zip(metrics_to_plot, colors):
        offset = width * multiplier
        bars = ax.bar(x + offset, comparison_df[metric], width, 
                     label=metric.replace('Test ', ''), color=color)
        multiplier += 1
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(comparison_df['Model'], fontsize=10)
    ax.legend(loc='lower right', ncol=2, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=7, rotation=90, padding=3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_metric_comparison.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved metric comparison to: {file_path}")
    
    return file_path


def plot_confusion_matrix(confusion_matrix, model_name, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot confusion matrix heatmap for best model.
    
    Args:
        confusion_matrix: 2x2 confusion matrix array
        model_name: Name of the model
        target_name: Name of target variable
        save_path: Path to save plot
    """
    logger.info("Creating confusion matrix plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'],
                ax=ax, annot_kws={'size': 14})
    
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_confusion_matrix.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved confusion matrix to: {file_path}")
    
    return file_path


# ============================================================================
# SEGMENT × PROPENSITY VISUALIZATIONS
# ============================================================================

def plot_segment_propensity_boxplot(propensity_df, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot box plot of propensity scores by segment.
    
    Args:
        propensity_df: DataFrame with user_id, value_cluster, propensity_score
        target_name: Name of target variable
        save_path: Path to save plot
    """
    logger.info("Creating segment propensity box plot...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    
    # Define segment names
    segment_names = {
        0: 'Casual Walk-in',
        1: 'Golden Whales',
        2: 'High Potential',
        3: 'Drifting Risk'
    }
    
    propensity_df['segment_name'] = propensity_df['value_cluster'].map(segment_names)
    
    # Create box plot
    sns.boxplot(data=propensity_df, x='segment_name', y='propensity_score',
                palette='Set2', ax=ax)
    
    ax.set_xlabel('Customer Segment (Case 1)', fontsize=12)
    ax.set_ylabel('Propensity Score', fontsize=12)
    ax.set_title('Propensity Score Distribution by Segment', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_segment_boxplot.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved segment box plot to: {file_path}")
    
    return file_path


def plot_segment_propensity_violin(propensity_df, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot violin plot of propensity scores by segment.
    
    Args:
        propensity_df: DataFrame with user_id, value_cluster, propensity_score
        target_name: Name of target variable
        save_path: Path to save plot
    """
    logger.info("Creating segment propensity violin plot...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    
    # Define segment names
    segment_names = {
        0: 'Casual Walk-in',
        1: 'Golden Whales',
        2: 'High Potential',
        3: 'Drifting Risk'
    }
    
    propensity_df['segment_name'] = propensity_df['value_cluster'].map(segment_names)
    
    # Create violin plot
    sns.violinplot(data=propensity_df, x='segment_name', y='propensity_score',
                   palette='Set2', ax=ax, inner='box')
    
    ax.set_xlabel('Customer Segment (Case 1)', fontsize=12)
    ax.set_ylabel('Propensity Score', fontsize=12)
    ax.set_title('Propensity Score Density by Segment', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_segment_violin.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved segment violin plot to: {file_path}")
    
    return file_path


def plot_segment_propensity_heatmap(propensity_df, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot heatmap of segment × propensity bins.
    
    Args:
        propensity_df: DataFrame with user_id, value_cluster, propensity_score
        target_name: Name of target variable
        save_path: Path to save plot
    """
    logger.info("Creating segment × propensity heatmap...")
    
    # Define segment names
    segment_names = {
        0: 'Casual Walk-in',
        1: 'Golden Whales',
        2: 'High Potential',
        3: 'Drifting Risk'
    }
    
    # Create propensity bins
    propensity_df['propensity_bin'] = pd.cut(
        propensity_df['propensity_score'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low (0-0.3)', 'Medium (0.3-0.6)', 'High (0.6-1.0)']
    )
    
    propensity_df['segment_name'] = propensity_df['value_cluster'].map(segment_names)
    
    # Create cross-tabulation
    cross_tab = pd.crosstab(
        propensity_df['segment_name'],
        propensity_df['propensity_bin'],
        normalize='index'
    ) * 100
    
    # Reorder rows
    row_order = ['Casual Walk-in', 'Golden Whales', 'High Potential', 'Drifting Risk']
    cross_tab = cross_tab.reindex(row_order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': '% of Segment'})
    
    ax.set_xlabel('Propensity Level', fontsize=12)
    ax.set_ylabel('Customer Segment', fontsize=12)
    ax.set_title('Segment × Propensity Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_segment_heatmap.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved segment heatmap to: {file_path}")
    
    return file_path


def plot_strategic_matrix(propensity_df, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot strategic matrix showing segment × propensity → marketing action.
    
    Args:
        propensity_df: DataFrame with user_id, value_cluster, propensity_score
        target_name: Name of target variable
        save_path: Path to save plot
    """
    logger.info("Creating strategic matrix...")
    
    # Define strategic actions
    strategic_actions = {
        ('Casual Walk-in', 'Low'): 'Onboarding\nTrial Offers',
        ('Casual Walk-in', 'Medium'): 'Engagement\nCampaigns',
        ('Casual Walk-in', 'High'): 'Conversion\nPush',
        ('Golden Whales', 'Low'): 'URGENT\nWin-back',
        ('Golden Whales', 'Medium'): 'VIP\nRetention',
        ('Golden Whales', 'High'): 'Loyalty\nRewards',
        ('High Potential', 'Low'): 'Re-engagement\nOffers',
        ('High Potential', 'Medium'): 'Subscription\nUpsell',
        ('High Potential', 'High'): 'Accelerate\nto VIP',
        ('Drifting Risk', 'Low'): 'Win-back\nDiscounts',
        ('Drifting Risk', 'Medium'): 'Feedback\nSurvey',
        ('Drifting Risk', 'High'): 'Retention\nCampaign'
    }
    
    # Define priority colors
    priority_colors = {
        ('Casual Walk-in', 'Low'): '#FFF9C4',
        ('Casual Walk-in', 'Medium'): '#FFECB3',
        ('Casual Walk-in', 'High'): '#FFE082',
        ('Golden Whales', 'Low'): '#FFCDD2',
        ('Golden Whales', 'Medium'): '#BBDEFB',
        ('Golden Whales', 'High'): '#C8E6C9',
        ('High Potential', 'Low'): '#FFECB3',
        ('High Potential', 'Medium'): '#B3E5FC',
        ('High Potential', 'High'): '#A5D6A7',
        ('Drifting Risk', 'Low'): '#FFCDD2',
        ('Drifting Risk', 'Medium'): '#FFCDD2',
        ('Drifting Risk', 'High'): '#FFE0B2'
    }
    
    segments = ['Casual Walk-in', 'Golden Whales', 'High Potential', 'Drifting Risk']
    propensity_levels = ['Low', 'Medium', 'High']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create grid
    for i, segment in enumerate(segments):
        for j, prop_level in enumerate(propensity_levels):
            color = priority_colors.get((segment, prop_level), '#FFFFFF')
            action = strategic_actions.get((segment, prop_level), '')
            
            rect = plt.Rectangle((j, len(segments)-1-i), 1, 1, 
                                 facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            ax.text(j + 0.5, len(segments)-1-i + 0.5, action,
                   ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 4)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(['Low Propensity\n(0-0.3)', 'Medium Propensity\n(0.3-0.6)', 
                       'High Propensity\n(0.6-1.0)'], fontsize=10)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(segments[::-1], fontsize=10)
    ax.set_xlabel('Propensity Score', fontsize=12)
    ax.set_ylabel('Customer Segment (Case 1)', fontsize=12)
    ax.set_title('Strategic Marketing Action Matrix', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor='#FFCDD2', edgecolor='black', label='High Priority'),
        plt.Rectangle((0,0), 1, 1, facecolor='#FFE082', edgecolor='black', label='Medium Priority'),
        plt.Rectangle((0,0), 1, 1, facecolor='#C8E6C9', edgecolor='black', label='Maintain/Reward')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    file_path = f"{save_path}/{target_name}_strategic_matrix.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved strategic matrix to: {file_path}")
    
    return file_path


# ============================================================================
# SHAP VISUALIZATIONS (will be populated after SHAP analysis)
# ============================================================================

def plot_shap_summary(shap_values, X_test, feature_names, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot SHAP summary plot (beeswarm).
    
    Args:
        shap_values: SHAP values array
        X_test: Test features DataFrame
        feature_names: List of feature names
        target_name: Name of target variable
        save_path: Path to save plot
    """
    try:
        import shap
        
        logger.info("Creating SHAP summary plot...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                         show=False, max_display=20)
        
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        os.makedirs(save_path, exist_ok=True)
        file_path = f"{save_path}/{target_name}_shap_summary.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved SHAP summary to: {file_path}")
        
        return file_path
    except ImportError:
        logger.warning("  ⚠️ SHAP not installed. Skipping SHAP summary plot.")
        return None


def plot_shap_bar(shap_values, feature_names, target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Plot SHAP bar plot (mean absolute values).
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        target_name: Name of target variable
        save_path: Path to save plot
    """
    try:
        import shap
        
        logger.info("Creating SHAP bar plot...")
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame for plotting
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=True).tail(20)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.barh(shap_df['feature'], shap_df['importance'], color='steelblue')
        ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
        ax.set_title('Top 20 Feature Importance (SHAP)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(save_path, exist_ok=True)
        file_path = f"{save_path}/{target_name}_shap_bar.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved SHAP bar plot to: {file_path}")
        
        return file_path
    except ImportError:
        logger.warning("  ⚠️ SHAP not installed. Skipping SHAP bar plot.")
        return None


# ============================================================================
# GENERATE ALL EVALUATION PLOTS
# ============================================================================

def generate_all_evaluation_plots(results, comparison_df, best_metrics, best_model_name,
                                  target_name='will_purchase', save_path=PLOTS_PATH):
    """
    Generate all model evaluation plots.
    
    Args:
        results: Dictionary of evaluation results
        comparison_df: Model comparison DataFrame
        best_metrics: Best model metrics
        best_model_name: Name of best model
        target_name: Name of target variable
        save_path: Path to save plots
        
    Returns:
        dict: Paths to generated plots
    """
    logger.info("="*80)
    logger.info("GENERATING EVALUATION PLOTS")
    logger.info("="*80)
    
    plot_paths = {}
    
    # ROC curves
    plot_paths['roc_curves'] = plot_roc_curves(results, target_name, save_path)
    
    # PR curves
    plot_paths['pr_curves'] = plot_pr_curves(results, target_name, save_path)
    
    # Metric comparison
    plot_paths['metric_comparison'] = plot_metric_comparison(comparison_df, target_name, save_path)
    
    # Confusion matrix (best model)
    plot_paths['confusion_matrix'] = plot_confusion_matrix(
        best_metrics['confusion_matrix'], best_model_name, target_name, save_path
    )
    
    logger.info("\n✓ All evaluation plots generated")
    
    return plot_paths


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # This module is typically imported and used by other scripts
    print("Visualization module loaded successfully.")
    print("Use generate_all_evaluation_plots() to create model evaluation plots.")
    print("Use plot_segment_propensity_* functions for segment analysis.")
