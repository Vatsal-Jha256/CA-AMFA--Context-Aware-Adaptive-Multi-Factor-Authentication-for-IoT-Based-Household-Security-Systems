import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import seaborn as sns

# Set up publication-quality plot parameters
def set_publication_style():
    """Set global matplotlib parameters for truly publication-quality figures"""
    plt.rcParams.update({
        # Font settings - switched to a more readable font
        'font.family': 'Arial',
        'font.size': 9,  # Reduced from 11 for less crowding
        'font.weight': 'normal',  # Changed from bold for better readability
        
        # Axes settings - reduced grid prominence
        'axes.linewidth': 1.0,  # Thinner lines
        'axes.grid': False,  # Turn off default grid
        
        # Tick settings - more subtle
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.visible': False,  # Hide minor ticks
        'ytick.minor.visible': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Title and label settings - more streamlined
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        
        # Legend settings - more subtle
        'legend.frameon': False,  # No frame around legend
        'legend.fontsize': 8,
        
        # Figure settings
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'figure.figsize': (5, 3.5),  # Smaller default size
        'figure.dpi': 100,
        
        # Output settings (for PDF editing)
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    
    # Set a more muted color cycle with better differentiation
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', 
        '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9'
    ])
    
    # Create output directory if it doesn't exist
    os.makedirs("simulation_graphs", exist_ok=True)

def stylize_axes(ax):
    """Apply enhanced publication-quality styling to axis"""
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make remaining spines thinner
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Style ticks
    ax.xaxis.set_tick_params(direction='out', width=1.0, length=4, pad=3)
    ax.yaxis.set_tick_params(direction='out', width=1.0, length=4, pad=3)
    
    # Only add subtle y-grid when needed
    ax.grid(False)  # Turn off grid by default
    
    # Use white background instead of light gray
    ax.set_facecolor('white')
    
    # Make tick labels smaller and normal weight
    for label in ax.get_xticklabels():
        label.set_fontsize(8)
        label.set_fontweight('normal')
    
    for label in ax.get_yticklabels():
        label.set_fontsize(8)
        label.set_fontweight('normal')
    
    return ax

def plot_performance_comparison(results_df, output_file='performance_comparison.pdf'):
    """Plot performance comparison between methods with publication-quality styling"""
    # Select only the most important metrics for clarity
    key_metrics = ['accuracy', 'f1_score', 'far', 'frr']
    
    # Create a subset of the data with only key metrics
    plot_data = results_df[['Method'] + key_metrics].set_index('Method')
    
    # Create a color palette with higher contrast
    colors = sns.color_palette("muted", len(key_metrics))
    
    # Plot
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax = plot_data.plot(kind='bar', color=colors, ax=ax)
    
    # Add minimalist title and labels
    ax.set_title('Performance Comparison', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=10)
    ax.set_xlabel('Method', fontsize=10)
    
    # Rotate x-tick labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Add subtle value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=7, padding=3)
    
    # Style the axis
    stylize_axes(ax)
    
    # Add subtle y-grid
    ax.grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
    
    # Place legend outside the plot for clarity
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_risk_factor_weights(weight_history, factor_names, output_file='factor_weights_evolution.pdf'):
    """Plot risk factor weights over time with publication-quality styling"""
    methods = list(weight_history.keys())
    
    # Limit to top 3-4 most significant factors for clarity
    if len(factor_names) > 4:
        # Here you would implement logic to identify most important factors
        # For example, based on average weight magnitude or variance
        factor_names = factor_names[:4]  # Simplified example: take first 4
    
    # Create a figure with subplots for each factor
    fig, axes = plt.subplots(len(factor_names), 1, figsize=(5, 1.5*len(factor_names)), sharex=True)
    
    # Ensure axes is always a list for consistent indexing
    if len(factor_names) == 1:
        axes = [axes]
    
    # Create a color palette with better differentiation
    colors = sns.color_palette("deep", len(methods))
    
    for i, factor in enumerate(factor_names):
        for j, method in enumerate(methods):
            factor_history = weight_history[method][factor]
            if factor_history:
                # Improve data density - downsample if too many points
                if len(factor_history) > 100:
                    step = len(factor_history) // 100
                    factor_history = factor_history[::step]
                
                timestamps = [pd.to_datetime(h[0], unit='s') for h in factor_history]
                weights = [h[1] for h in factor_history]
                
                # Apply smoothing for clearer trends (rolling average)
                if len(weights) > 10:
                    weights_series = pd.Series(weights)
                    weights = weights_series.rolling(window=5, min_periods=1).mean()
                
                axes[i].plot(timestamps, weights, label=method, color=colors[j], linewidth=1.5)
        
        # Add minimal labels
        axes[i].set_title(f'{factor.capitalize()}', fontsize=9, fontweight='bold')
        axes[i].set_ylabel('Weight', fontsize=8)
        
        # Style the axis
        stylize_axes(axes[i])
        
        # Add legend to the first subplot only
        if i == 0:
            axes[i].legend(loc='best', fontsize=7, frameon=False)
    
    # Add common x-label
    fig.text(0.5, 0.01, 'Time', ha='center', fontsize=9)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_authentication_methods(auth_decisions, output_file='auth_methods_evolution.pdf'):
    """Plot authentication method usage over time with publication-quality styling"""
    methods = list(auth_decisions.keys())
    
    # Limit to comparing just 2-3 methods for clarity if there are many
    if len(methods) > 3:
        methods = methods[:3]  # Take first 3 for simplicity
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Define colors for different authentication levels - more distinguishable
    auth_colors = ['#0173B2', '#DE8F05', '#029E73']  # Blue, Orange, Green
    auth_labels = ['Password Only', 'Password + OTP', 'Password + OTP + Face']
    
    # For each method, aggregate data by day instead of hour for better visibility
    all_data = []
    
    for method in methods:
        decisions = auth_decisions[method]
        if not decisions:
            continue
            
        # Group by day instead of hour
        from collections import defaultdict
        day_groups = defaultdict(list)
        
        for decision in decisions:
            # Round timestamp to nearest day
            timestamp = decision['timestamp']
            day = int(timestamp) // 86400 * 86400  # Seconds in a day
            day_groups[day].append(decision)
        
        # Process each day
        day_data = {'timestamp': [], 'auth_level_counts': [[], [], []]}
        
        for day, day_decisions in sorted(day_groups.items()):
            day_data['timestamp'].append(pd.to_datetime(day, unit='s'))
            
            # Count authentication methods
            level1 = sum(1 for d in day_decisions if len(d['auth_methods']) == 1)
            level2 = sum(1 for d in day_decisions if len(d['auth_methods']) == 2)
            level3 = sum(1 for d in day_decisions if len(d['auth_methods']) == 3)
            
            day_data['auth_level_counts'][0].append(level1)
            day_data['auth_level_counts'][1].append(level2) 
            day_data['auth_level_counts'][2].append(level3)
        
        all_data.append((method, day_data))
    
    # Instead of stacked areas (which can be hard to read), use grouped bars for key dates
    # or line plots for trends over time
    
    # Example: create line plots for each authentication level
    for method, data in all_data:
        if not data['timestamp']:
            continue
            
        # For each method, plot percentage of each auth level
        for i in range(3):
            # Calculate percentage
            total_counts = [sum(x) for x in zip(*data['auth_level_counts'])]
            percentages = [count/total*100 if total > 0 else 0 
                          for count, total in zip(data['auth_level_counts'][i], total_counts)]
            
            # Plot as line
            ax.plot(data['timestamp'], percentages, 
                   linestyle=['-', '--', ':'][i],  # Different line styles
                   color=auth_colors[i], 
                   label=f"{method} - {auth_labels[i]}")
    
    # Add labels
    ax.set_title('Authentication Method Usage', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=10)
    ax.set_xlabel('Date', fontsize=10)
    
    # Style the axis
    stylize_axes(ax)
    
    # Format x-axis to show fewer dates
    from matplotlib.dates import AutoDateLocator, DateFormatter
    locator = AutoDateLocator(minticks=3, maxticks=7)
    formatter = DateFormatter('%m/%d')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    # Add subtle grid
    ax.grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
    
    # Place legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()
def plot_risk_scores_over_time(auth_decisions, output_file='risk_scores_evolution.pdf'):
    """Plot risk scores over time for all methods with publication-quality styling"""
    methods = list(auth_decisions.keys())
    
    # Limit to comparing just 2-3 methods for clarity if there are many
    if len(methods) > 3:
        methods = methods[:3]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Create a more distinguishable color palette
    colors = sns.color_palette("deep", len(methods))
    
    for i, method in enumerate(methods):
        decisions = auth_decisions[method]
        if not decisions:
            continue
            
        # Group by day for clearer trends
        from collections import defaultdict
        day_groups = defaultdict(list)
        
        for decision in decisions:
            timestamp = decision['timestamp']
            day = int(timestamp) // 86400 * 86400
            day_groups[day].append(decision)
            
        # Process each day
        days = []
        avg_risks = []
        
        for day, day_decisions in sorted(day_groups.items()):
            days.append(pd.to_datetime(day, unit='s'))
            avg_risks.append(sum(d['risk_score'] for d in day_decisions) / len(day_decisions))
        
        # Plot average risk score by day
        ax.plot(days, avg_risks, label=method, linewidth=2, color=colors[i])
    
    # Add labels
    ax.set_title('Daily Average Risk Scores', fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Risk Score', fontsize=10)
    
    # Style the axis
    stylize_axes(ax)
    
    # Format x-axis to show fewer dates
    from matplotlib.dates import AutoDateLocator, DateFormatter
    locator = AutoDateLocator(minticks=3, maxticks=7)
    formatter = DateFormatter('%m/%d')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    # Add subtle grid
    ax.grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
    
    # Add legend
    ax.legend(loc='best', fontsize=8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()
def plot_security_metrics(auth_decisions, output_file='security_metrics_evolution.pdf'):
    """Plot security-related metrics over time with publication-quality styling"""
    methods = list(auth_decisions.keys())
    
    # Focus on the most important metrics only
    metrics = ['false_positives', 'false_negatives']  # Security critical metrics
    
    # Create a single figure for both metrics, side by side
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    
    # Create a color palette
    colors = sns.color_palette("deep", len(methods))
    
    for i, metric in enumerate(metrics):
        for j, method in enumerate(methods):
            decisions = auth_decisions[method]
            if not decisions:
                continue
                
            # Group by week for better trend visibility
            from collections import defaultdict
            week_groups = defaultdict(list)
            
            for decision in decisions:
                timestamp = decision['timestamp']
                week = int(timestamp) // (86400 * 7) * (86400 * 7)  # Round to week
                
                # Count this metric
                is_counted = 1 if decision['legitimate'] == (metric == 'false_negatives') and \
                               decision['success'] != (metric == 'false_negatives') else 0
                
                week_groups[week].append(is_counted)
            
            # Process each week
            weeks = []
            rates = []
            
            for week, counts in sorted(week_groups.items()):
                weeks.append(pd.to_datetime(week, unit='s'))
                # Calculate rate
                rate = sum(counts) / len(counts)
                rates.append(rate * 100)  # Convert to percentage
            
            # Plot rate by week
            if weeks:
                axes[i].plot(weeks, rates, label=method, linewidth=1.5, color=colors[j])
        
        # Add labels
        metric_label = 'False Accept Rate' if metric == 'false_positives' else 'False Reject Rate'
        axes[i].set_title(metric_label, fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Rate (%)', fontsize=9)
        
        if i == len(metrics) - 1:
            axes[i].set_xlabel('Date', fontsize=9)
        
        # Style the axis
        stylize_axes(axes[i])
        
        # Format x-axis
        from matplotlib.dates import AutoDateLocator, DateFormatter
        locator = AutoDateLocator(minticks=3, maxticks=5)
        formatter = DateFormatter('%m/%d')
        axes[i].xaxis.set_major_locator(locator)
        axes[i].xaxis.set_major_formatter(formatter)
        
        # Add subtle grid
        axes[i].grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
        
        # Add legend to only one subplot
        if i == 0:
            axes[i].legend(loc='best', fontsize=7)
    
    # Add overall title
    fig.suptitle('Security Performance Over Time', fontsize=11, fontweight='bold')
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_user_experience(auth_decisions, output_file='user_experience_metrics.pdf'):
    """Plot user experience metrics over time with publication-quality styling"""
    methods = list(auth_decisions.keys())
    
    # Create a single figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    
    # Create a color palette with better differentiation
    colors = sns.color_palette("deep", len(methods))
    
    # Plot 1: Authentication Success Rate for legitimate users
    for i, method in enumerate(methods):
        legitimate_decisions = [d for d in auth_decisions[method] if d.get('legitimate', True)]
        
        if not legitimate_decisions:
            continue
            
        # Group by week for clearer trends
        from collections import defaultdict
        week_groups = defaultdict(list)
        
        for decision in legitimate_decisions:
            timestamp = decision['timestamp']
            week = int(timestamp) // (86400 * 7) * (86400 * 7)  # Round to week
            week_groups[week].append(decision)
        
        # Process weekly data
        weeks = []
        success_rates = []
        
        for week, week_decisions in sorted(week_groups.items()):
            weeks.append(pd.to_datetime(week, unit='s'))
            success_rate = sum(1 for d in week_decisions if d['success']) / len(week_decisions)
            success_rates.append(success_rate * 100)  # Convert to percentage
        
        # Plot success rate
        if weeks:
            axes[0].plot(weeks, success_rates, label=method, linewidth=1.5, color=colors[i])
    
    # Style first plot
    axes[0].set_title('Authentication Success Rate', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Success Rate (%)', fontsize=9)
    axes[0].set_ylim(0, 100)
    
    # Plot 2: Average Auth Methods for legitimate users (simpler visualization)
    for i, method in enumerate(methods):
        legitimate_decisions = [d for d in auth_decisions[method] if d.get('legitimate', True)]
        
        if not legitimate_decisions:
            continue
            
        # Group by week
        week_groups = defaultdict(list)
        
        for decision in legitimate_decisions:
            timestamp = decision['timestamp']
            week = int(timestamp) // (86400 * 7) * (86400 * 7)
            week_groups[week].append(decision)
        
        # Process weekly data
        weeks = []
        avg_methods = []
        
        for week, week_decisions in sorted(week_groups.items()):
            weeks.append(pd.to_datetime(week, unit='s'))
            avg = sum(len(d['auth_methods']) for d in week_decisions) / len(week_decisions)
            avg_methods.append(avg)
        
        # Plot average methods
        if weeks:
            axes[1].plot(weeks, avg_methods, label=method, linewidth=1.5, color=colors[i])
    
    # Style both plots
    axes[1].set_title('Avg. Authentication Methods', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Number of Methods', fontsize=9)
    axes[1].set_ylim(0.9, 3.1)
    
    for i in range(2):
        # Style the axis
        stylize_axes(axes[i])
        
        # Format x-axis
        from matplotlib.dates import AutoDateLocator, DateFormatter
        locator = AutoDateLocator(minticks=3, maxticks=5)
        formatter = DateFormatter('%m/%d')
        axes[i].xaxis.set_major_locator(locator)
        axes[i].xaxis.set_major_formatter(formatter)
        axes[i].set_xlabel('Date', fontsize=9)
        
        # Add subtle grid
        axes[i].grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
    
    # Add legend to the first subplot only
    axes[0].legend(loc='best', fontsize=7)
    
    # Add overall title
    fig.suptitle('User Experience Metrics', fontsize=11, fontweight='bold')
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()
def plot_environmental_adaptation(metrics, environments, output_file='environmental_adaptation.pdf'):
    """Plot performance in different environments with publication-quality styling"""
    methods = list(metrics.keys())
    
    # Filter environments - if too many, focus on the most distinctive ones
    if len(environments) > 5:
        environments = environments[:5]  # Take first 5
    
    # Prepare data
    env_data = []
    
    for method in methods:
        for env in environments:
            env_key = f'env_{env}'
            if env_key in metrics[method] and metrics[method][env_key]:
                success_rate = sum(metrics[method][env_key]) / len(metrics[method][env_key])
                env_data.append({
                    'Method': method,
                    'Environment': env.replace('_', ' ').title(),
                    'Success Rate': success_rate * 100  # Convert to percentage
                })
    
    if not env_data:
        print("No environment-specific data available for plotting")
        return
        
    df = pd.DataFrame(env_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create grouped bar chart with better spacing
    ax = sns.barplot(x='Environment', y='Success Rate', hue='Method', data=df)
    
    # Add labels with more appropriate sizing
    plt.title('Performance Across Environments', fontsize=11, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=10)
    plt.xlabel('Environment', fontsize=10)
    plt.ylim(0, 100)
    
    # Better legend positioning
    plt.legend(title='Method', frameon=False, fontsize=8, loc='best')
    
    # Style the axis
    stylize_axes(ax)
    
    # Handle x-tick rotations for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add data labels on bars for clarity
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=7, padding=3)
    
    # Add subtle grid
    ax.grid(axis='y', linestyle=':', alpha=0.3, color='#cccccc')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()
def generate_latex_table(results, output_file='simulation_results/results_table.tex'):
    """Generate LaTeX table from simulation results for publication"""
    methods = list(results.keys())
    
    # Define the metrics we want to include in our table
    key_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score', 'far', 'frr', 'eer', 
        'avg_auth_factors', 'avg_risk_score'
    ]
    
    # Create a DataFrame for easier manipulation
    table_data = []
    for method in methods:
        row = {'Method': method.title()}
        for metric in key_metrics:
            if metric in results[method]:
                row[metric] = results[method][metric]
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Format the column names for LaTeX
    column_names = {
        'Method': 'Method',
        'accuracy': 'Accuracy', 
        'precision': 'Precision', 
        'recall': 'Recall',
        'f1_score': 'F1 Score', 
        'far': 'FAR', 
        'frr': 'FRR', 
        'eer': 'EER',
        'avg_auth_factors': 'Avg. Auth Factors',
        'avg_risk_score': 'Avg. Risk Score'
    }
    
    df = df.rename(columns=column_names)
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.3f")
    
    # Enhance the LaTeX table with better formatting
    latex_table = latex_table.replace('tabular', 'tabular*{\\textwidth}')
    latex_table = latex_table.replace('\\begin{tabular*}{\\textwidth}', '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l' + 'c' * (len(df.columns) - 1) + '}')
    
    # Add table caption and label
    latex_header = "\\begin{table}[htbp]\n\\centering\n\\caption{Performance Comparison of Risk Assessment Methods}\n\\label{tab:performance_comparison}\n"
    latex_footer = "\\end{table}"
    
    latex_table = latex_header + latex_table + latex_footer
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_file}")
    return latex_table

def generate_comparison_table(results, baseline_method='fixed', output_file='simulation_results/comparison_table.tex'):
    """Generate LaTeX table comparing methods against a baseline (typically fixed weights)"""
    methods = list(results.keys())
    
    # Define the metrics we want to include in our comparison
    key_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score', 'far', 'frr', 'eer', 
        'avg_auth_factors'
    ]
    
    # Create a DataFrame for comparison
    table_data = []
    
    # Get baseline values
    baseline_values = {}
    if baseline_method in results:
        for metric in key_metrics:
            if metric in results[baseline_method]:
                baseline_values[metric] = results[baseline_method][metric]
    
    # Calculate percentage improvements
    for method in methods:
        if method == baseline_method:
            continue
            
        row = {'Method': method.title()}
        for metric in key_metrics:
            if metric in results[method] and metric in baseline_values and baseline_values[metric] != 0:
                # Calculate percentage improvement
                improvement = (results[method][metric] - baseline_values[metric]) / baseline_values[metric] * 100
                
                # Handle metrics where lower is better (FAR, FRR, EER)
                if metric in ['far', 'frr', 'eer', 'avg_auth_factors']:
                    improvement = -improvement
                    
                row[metric] = improvement
        table_data.append(row)
    
    if not table_data:
        print("No comparison data available")
        return None
    
    df = pd.DataFrame(table_data)
    
    # Format the column names for LaTeX
    column_names = {
        'Method': 'Method',
        'accuracy': 'Accuracy \\%', 
        'precision': 'Precision \\%', 
        'recall': 'Recall \\%',
        'f1_score': 'F1 Score \\%', 
        'far': 'FAR \\%', 
        'frr': 'FRR \\%', 
        'eer': 'EER \\%',
        'avg_auth_factors': 'Auth Factors \\%'
    }
    
    df = df.rename(columns=column_names)
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.2f")
    
    # Add highlighting for positive values
    for metric in key_metrics:
        if metric in column_names:
            metric_latex = column_names[metric].replace('\\%', '\\\\%')
            latex_table = latex_table.replace(metric_latex, metric_latex + ' Improvement')
    
    # Enhance the LaTeX table with better formatting
    latex_table = latex_table.replace('tabular', 'tabular*{\\textwidth}')
    latex_table = latex_table.replace('\\begin{tabular*}{\\textwidth}', '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l' + 'c' * (len(df.columns) - 1) + '}')
    
    # Add table caption and label
    latex_header = f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{Percentage Improvement Compared to {baseline_method.title()} Method}}\n\\label{{tab:improvement_comparison}}\n"
    latex_footer = "\\end{table}"
    
    latex_table = latex_header + latex_table + latex_footer
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX comparison table saved to {output_file}")
    return latex_table

def plot_table(data, output_file='results_table.pdf'):
    """Create a visual table using matplotlib with cleaner presentation"""
    # Convert data dictionary to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        # Process the data dictionary
        methods = list(data.keys())
        
        # Choose only the most important metrics for clarity
        key_metrics = ['accuracy', 'f1_score', 'far', 'frr', 'eer']
        
        # Create DataFrame
        table_data = []
        for method in methods:
            row = {'Method': method}
            for metric in key_metrics:
                if metric in data[method]:
                    row[metric] = data[method][metric]
                else:
                    row[metric] = None
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
    else:
        df = data.copy()
        # Filter to key metrics if DataFrame has too many columns
        if len(df.columns) > 6:  # Including 'Method'
            key_metrics = ['Method', 'accuracy', 'f1_score', 'far', 'frr', 'eer']
            available_cols = [col for col in key_metrics if col in df.columns]
            df = df[available_cols]
    
    # Rename columns for better display
    column_names = {
        'Method': 'Method',
        'accuracy': 'Accuracy', 
        'precision': 'Precision', 
        'recall': 'Recall',
        'f1_score': 'F1 Score', 
        'far': 'FAR', 
        'frr': 'FRR', 
        'eer': 'EER',
        'avg_auth_factors': 'Auth Factors',
        'avg_risk_score': 'Risk Score'
    }
    
    df = df.rename(columns={k: v for k, v in column_names.items() if k in df.columns})
    
    # Set Method as index if it exists
    if 'Method' in df.columns:
        df = df.set_index('Method')
    
    # Format numeric columns to 3 decimal places
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].map('{:.3f}'.format)
    
    # Create figure and axis with appropriate size
    fig, ax = plt.subplots(figsize=(len(df.columns) + 1, len(df) * 0.5 + 1))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create table with minimal styling
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc='center',
        loc='center'
    )
    
    # Style the table more appropriately for academic publication
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.3)
    
    # Apply custom styling - simpler, more professional
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#e6e6e6')  # Light gray
        elif j == -1:  # Row labels
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')  # Lighter gray
        
        # Add subtle borders
        cell.set_edgecolor('#cccccc')
    
    # Add title
    plt.title('Performance Metrics', fontsize=11, fontweight='bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visual table saved to simulation_graphs/{output_file}")

def plot_heatmap_comparison(data, baseline_method='fixed', output_file='comparison_heatmap.pdf'):
    """Create a heatmap showing the percentage improvement against a baseline method"""
    # Convert data dictionary to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        methods = list(data.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'far', 'frr', 'eer', 'avg_auth_factors']
        
        # Get baseline values
        baseline = {}
        if baseline_method in data:
            for metric in metrics:
                if metric in data[baseline_method]:
                    baseline[metric] = data[baseline_method][metric]
        
        # Calculate improvements
        improvement_data = {}
        for method in methods:
            if method == baseline_method:
                continue
                
            improvement_data[method] = {}
            for metric in metrics:
                if metric in data[method] and metric in baseline and baseline[metric] != 0:
                    improvement = (data[method][metric] - baseline[metric]) / baseline[metric] * 100
                    
                    # For metrics where lower is better
                    if metric in ['far', 'frr', 'eer', 'avg_auth_factors']:
                        improvement = -improvement
                        
                    improvement_data[method][metric] = improvement
        
        # Convert to DataFrame
        df = pd.DataFrame(improvement_data).T
    else:
        df = data.copy()
    
    # Rename columns for better display
    column_names = {
        'accuracy': 'Accuracy', 
        'precision': 'Precision', 
        'recall': 'Recall',
        'f1_score': 'F1 Score', 
        'far': 'FAR', 
        'frr': 'FRR', 
        'eer': 'EER',
        'avg_auth_factors': 'Auth Factors'
    }
    
    df = df.rename(columns={k: v for k, v in column_names.items() if k in df.columns})
    
    # Create figure and axis
    plt.figure(figsize=(10, len(df) + 1))
    
    # Create a custom colormap - red to white to green
    cmap = sns.diverging_palette(10, 120, as_cmap=True)
    
    # Create heatmap
    ax = sns.heatmap(df, annot=True, cmap=cmap, center=0, fmt='.1f',
                linewidths=.5, cbar_kws={'label': 'Percentage Improvement (%)'},
                vmin=-50, vmax=50)
    
    # Add title and labels
    plt.title(f'Percentage Improvement Compared to {baseline_method.title()}', fontsize=14, fontweight='bold')
    plt.ylabel('Method')
    
    # Style the plot
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison heatmap saved to simulation_graphs/{output_file}")

def plot_radar_chart(data, output_file='radar_chart.pdf'):
    """Create a radar chart comparing different methods across key metrics"""
    methods = list(data.keys())
    
    # Choose metrics for radar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    available_metrics = [m for m in metrics if all(m in data[method] for method in methods)]
    
    if not available_metrics:
        print("Error: Not enough metrics available for radar chart")
        return
    
    # Extract data
    values = {}
    for method in methods:
        values[method] = [data[method][metric] for metric in available_metrics]
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of metrics
    N = len(available_metrics)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create color palette
    colors = sns.color_palette("deep", len(methods))
    
    # Plot each method
    for i, method in enumerate(methods):
        values_method = values[method]
        values_method += values_method[:1]  # Close the loop
        ax.plot(angles, values_method, linewidth=2, linestyle='solid', label=method.title(), color=colors[i])
        ax.fill(angles, values_method, alpha=0.1, color=colors[i])
    
    # Add metrics labels
    nice_names = {
        'accuracy': 'Accuracy', 
        'precision': 'Precision', 
        'recall': 'Recall',
        'f1_score': 'F1 Score'
    }
    
    metric_labels = [nice_names.get(m, m) for m in available_metrics]
    plt.xticks(angles[:-1], metric_labels, fontsize=12, fontweight='bold')
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title('Performance Metrics Comparison', size=14, fontweight='bold', y=1.1)
    
    # Style the chart
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"simulation_graphs/{output_file}", dpi=300, bbox_inches='tight')
    plt.savefig(f"simulation_graphs/{output_file.replace('.pdf', '.png')}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Radar chart saved to simulation_graphs/{output_file}")

def generate_all_visualizations(results, auth_decisions, weight_history, factor_names, metrics, environments):
    """Generate all visualizations with publication-quality styling"""
    # Set publication style
    set_publication_style()
    
    # Convert results to DataFrame for plotting
    methods = list(results.keys())
    metrics_list = list(results[methods[0]].keys())
    
    # Create a dataframe for the results
    data = []
    for method in methods:
        row = [method]
        for metric in metrics_list:
            row.append(results[method][metric])
        data.append(row)
        
    columns = ['Method'] + metrics_list
    results_df = pd.DataFrame(data, columns=columns)
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    generate_latex_table(results)
    generate_comparison_table(results)
    
    # Generate table visualizations
    print("Generating visual result tables...")
    plot_table(results_df)
    plot_heatmap_comparison(results)
    plot_radar_chart(results)
    
    # Generate all plots
    print("Generating performance comparison chart...")
    plot_performance_comparison(results_df)
    
    print("Generating risk factor weights evolution chart...")
    plot_risk_factor_weights(weight_history, factor_names)
    
    print("Generating authentication methods chart...")
    plot_authentication_methods(auth_decisions)
    
    print("Generating risk scores evolution chart...")
    plot_risk_scores_over_time(auth_decisions)
    
    print("Generating security metrics evolution chart...")
    plot_security_metrics(auth_decisions)
    
    print("Generating user experience metrics chart...")
    plot_user_experience(auth_decisions)
    
    print("Generating environmental adaptation chart...")
    plot_environmental_adaptation(metrics, environments)
    
    print(f"All visualizations generated successfully in simulation_graphs/")
    print(f"LaTeX tables generated in simulation_results/") 