import argparse
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt


def plot_summary_analysis(results_dirpath: str) -> None:
    # Get all result CSV files
    result_files = [f for f in os.listdir(results_dirpath) if f.startswith('results_') and f.endswith('.csv')]
    
    if not result_files:
        print("No result files found in the specified directory.")
        return
    
    # Extract experiment names
    experiment_names = [f.replace('results_', '').replace('.csv', '') for f in result_files]
    
    # Initialize lists to store metrics
    max_f1_values = []
    max_f1_topk_values = []
    max_precision_values = []
    max_recall_values = []
    max_jaccard_values = []
    max_f1_errors = []  # Add list to store error values
    
    # Process each result file
    for exp_name in experiment_names:
        # Load the results
        df = pd.read_csv(f'{results_dirpath}/results_{exp_name}.csv')
        
        # Calculate mean metrics across all samples
        metrics_mean = df.mean(axis=0)
        metrics_err = df.std(axis=0) / np.sqrt(len(df))
        
        # Extract F1 scores for different top-k values
        f1_values = []
        ref_cnts = []
        
        max_topk = max([int(name.split('top')[-1]) for name in metrics_mean.index.tolist() if name.startswith('f1_top')])
        for topk in range(1, max_topk + 1):
            topk_str = f'top{topk}'
            if f'f1_{topk_str}' in metrics_mean:
                ref_cnts.append(topk)
                f1_values.append(metrics_mean[f'f1_{topk_str}'])
        
        # Find the index of maximum F1 score
        max_f1_idx = np.argmax(f1_values)
        max_f1_topk = ref_cnts[max_f1_idx]
        max_f1 = f1_values[max_f1_idx]
        
        # Get corresponding precision, recall, and Jaccard at max F1
        max_precision = metrics_mean[f'prec_top{max_f1_topk}']
        max_recall = metrics_mean[f'rec_top{max_f1_topk}']
        max_jaccard = metrics_mean[f'jaccard_top{max_f1_topk}']
        max_f1_error = metrics_err[f'f1_top{max_f1_topk}']  # Get error for max F1
        
        # Store the values
        max_f1_values.append(max_f1)
        max_f1_topk_values.append(max_f1_topk)
        max_precision_values.append(max_precision)
        max_recall_values.append(max_recall)
        max_jaccard_values.append(max_jaccard)
        max_f1_errors.append(max_f1_error)  # Store the error
    
    # Create a DataFrame for the summary
    summary_df = pd.DataFrame({
        'Experiment': experiment_names,
        'Max F1': max_f1_values,
        'Max F1 Error': max_f1_errors,  # Add errors to the DataFrame
        'Top-k at Max F1': max_f1_topk_values,
        'Precision at Max F1': max_precision_values,
        'Recall at Max F1': max_recall_values,
        'Jaccard at Max F1': max_jaccard_values
    })
    
    # Extract model name and method from experiment name
    summary_df['Model'] = summary_df['Experiment'].apply(lambda x: x.split('-')[0] if '-' in x else x)
    summary_df['Method'] = summary_df['Experiment'].apply(lambda x: x.split('-')[1] if '-' in x else 'unknown')
    
    # Keep only one keyword entry (as it's model-independent)
    keyword_entries = summary_df[summary_df['Method'] == 'keyword']
    if not keyword_entries.empty:
        # Keep only the first keyword entry
        keyword_entry = keyword_entries.iloc[0:1]
        # Remove all keyword entries
        summary_df = summary_df[summary_df['Method'] != 'keyword']
        # Add back just the single keyword entry
        summary_df = pd.concat([summary_df, keyword_entry])
        # Rename the experiment for the keyword entry
        summary_df.loc[summary_df['Method'] == 'keyword', 'Experiment'] = 'keyword'
    
    # Sort by Max F1 in descending order
    summary_df = summary_df.sort_values('Max F1', ascending=False)
    
    # Save the summary to CSV
    summary_df.to_csv(f'{results_dirpath}/summary_analysis.csv', index=False)
    
    # Define color map for different models
    unique_models = summary_df[summary_df['Method'] != 'keyword']['Model'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    model_colors = dict(zip(unique_models, colors))
    
    # Create a bar plot for Max F1 scores with colors by model
    plt.figure(figsize=(12, 6))
    
    # Plot non-keyword entries with model colors
    non_keyword_df = summary_df[summary_df['Method'] != 'keyword']
    bars1 = plt.bar(non_keyword_df['Experiment'], non_keyword_df['Max F1'], 
                  color=[model_colors[model] for model in non_keyword_df['Model']],
                  yerr=non_keyword_df['Max F1 Error'],  # Add error bars
                  capsize=5)  # Add caps to error bars
    
    # Plot keyword entry with grey color
    keyword_df = summary_df[summary_df['Method'] == 'keyword']
    if not keyword_df.empty:
        bars2 = plt.bar(keyword_df['Experiment'], keyword_df['Max F1'], 
                      color='grey',
                      yerr=keyword_df['Max F1 Error'],  # Add error bars for keyword
                      capsize=5)  # Add caps to error bars
        bars = bars1.patches + bars2.patches
    else:
        bars = bars1.patches
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add legend for models
    legend_handles = [plt.Rectangle((0,0),1,1, color=model_colors[model]) for model in unique_models]
    if not keyword_df.empty:
        legend_handles.append(plt.Rectangle((0,0),1,1, color='grey'))
        unique_models = list(unique_models) + ['keyword']
    plt.legend(legend_handles, unique_models, title="Models")
    
    plt.xlabel('Experiment')
    plt.ylabel('Max F1 Score')
    plt.title('Maximum F1 Score Across Different Experiments')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{results_dirpath}/max_f1_comparison.pdf')
    
    # Print summary
    print("Summary analysis completed. Results saved to:")
    print(f"  {results_dirpath}/summary_analysis.csv")
    print(f"  {results_dirpath}/max_f1_comparison.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dirpath', type=str, default='/Users/luis/Desktop/ETH/Courses/SS25-DSL/benchmark_results')
    args = parser.parse_args()

    plot_summary_analysis(args.results_dirpath)