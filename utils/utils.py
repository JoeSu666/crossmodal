# import necessary libraries
import os
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def get_attribute(obj, attr):
    """Safely get an attribute from an object, returning default if not found."""
    try:
        return getattr(obj, attr)
    except AttributeError:
        raise AttributeError(f"Object {obj} has no attribute '{attr}'")

def save_cv_results(fold_results, cv_stats, output_dir, model_name):
    """Save cross-validation results to CSV files"""
    # Create results directory
    results_dir = os.path.join(output_dir, 'cv_results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual fold results
    fold_data = []
    for fold_name, results in fold_results.items():
        fold_num = fold_name.split('_')[1]
        test_results = results['test_results']
        fold_data.append({
            'fold': fold_num,
            'test_wacc': test_results['test/wacc'],
            'test_wf1': test_results['test/wf1'],
            'test_macroauc': test_results['test/macroauc'],
            'test_loss': test_results.get('test/loss', 0.0),
            'best_checkpoint': results['best_checkpoint']
        })
    
    fold_df = pd.DataFrame(fold_data)
    fold_csv_path = os.path.join(results_dir, f'{model_name}_fold_results_{timestamp}.csv')
    fold_df.to_csv(fold_csv_path, index=False)
    
    # Save summary statistics
    summary_data = [{
        'model': model_name,
        'timestamp': timestamp,
        **cv_stats
    }]
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(results_dir, f'{model_name}_cv_summary_{timestamp}.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"\n💾 Results saved:")
    print(f"  Fold results: {fold_csv_path}")
    print(f"  CV summary: {summary_csv_path}")

def print_cv_results(fold_results, cv_stats, model_name, folds):
    """Print comprehensive cross-validation results"""
    print(f"\n{'='*100}")
    print(f"CROSS-VALIDATION RESULTS: {model_name}")
    print(f"{'='*100}")
    
    # Individual fold results
    print(f"\n📊 Individual Fold Results:")
    print(f"{'Fold':<6} {'Test Acc':<12} {'Test F1':<12} {'Test AUC':<12} {'Test Loss':<12}")
    print(f"{'-'*60}")
    
    for fold in folds:
        results = fold_results[f"fold_{fold}"]['test_results']
        print(f"{fold:<6} {results['test/wacc']:<12.4f} {results['test/wf1']:<12.4f} "
              f"{results['test/macroauc']:<12.4f} {results.get('test/loss', 0.0):<12.4f}")
    
    # Summary statistics
    print(f"\n📈 Cross-Validation Summary Statistics:")
    print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"{'-'*75}")
    
    metrics = ['test_wacc', 'test_wf1', 'test_macroauc', 'test_loss']
    metric_names = ['Accuracy', 'F1-Score', 'AUC', 'Loss']
    
    for metric, name in zip(metrics, metric_names):
        if f"{metric}_mean" in cv_stats:
            mean_val = cv_stats[f"{metric}_mean"]
            std_val = cv_stats[f"{metric}_std"]
            min_val = cv_stats[f"{metric}_min"]
            max_val = cv_stats[f"{metric}_max"]
            print(f"{name:<15} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")
    
    # Confidence intervals (assuming normal distribution)
    print(f"\n🎯 95% Confidence Intervals:")
    for metric, name in zip(metrics, metric_names):
        if f"{metric}_mean" in cv_stats:
            mean_val = cv_stats[f"{metric}_mean"]
            std_val = cv_stats[f"{metric}_std"]
            ci_lower = mean_val - 1.96 * (std_val / np.sqrt(len(folds)))
            ci_upper = mean_val + 1.96 * (std_val / np.sqrt(len(folds)))
            print(f"{name:<15} [{ci_lower:.4f}, {ci_upper:.4f}]")

def calculate_cv_statistics(all_metrics):
    """Calculate mean and std for cross-validation metrics"""
    df = pd.DataFrame(all_metrics)
    
    stats = {}
    metric_columns = ['test_wacc', 'test_wf1', 'test_macroauc', 'test_loss']
    
    for metric in metric_columns:
        if metric in df.columns:
            stats[f"{metric}_mean"] = df[metric].mean()
            stats[f"{metric}_std"] = df[metric].std()
            stats[f"{metric}_min"] = df[metric].min()
            stats[f"{metric}_max"] = df[metric].max()
    
    return stats