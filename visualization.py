import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import auc
import os

def plot_roc_curves(save_dir='results'):
    model_types = ['gru', 'lstm', 'conv1d']
    
    for model_type in model_types:
        plt.figure(figsize=(10, 8))
        
        for i in range(7):  # Assuming 7 classes as per the original code
            roc_data = pd.read_csv(os.path.join(save_dir, f'{model_type}_roc_data_class_{i}.csv'))
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xscale('log')
        plt.xlim([1e-9, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves for {model_type.upper()} Model (Log Scale)', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'roc_curves_{model_type.lower()}.pdf'), format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

def plot_learning_curves(save_dir='results'):
    plt.figure(figsize=(12, 10))
    
    for model_type in ['gru', 'lstm', 'conv1d']:
        metrics = pd.read_csv(os.path.join(save_dir, f'{model_type}_training_metrics.csv'))
        
        plt.subplot(2, 1, 1)
        plt.plot(metrics['Epoch'], metrics['Train Loss'], label=f'{model_type.upper()} - Train')
        plt.plot(metrics['Epoch'], metrics['Validation Loss'], label=f'{model_type.upper()} - Validation')
        
        plt.subplot(2, 1, 2)
        plt.plot(metrics['Epoch'], metrics['Train Accuracy'], label=f'{model_type.upper()} - Train')
        plt.plot(metrics['Epoch'], metrics['Validation Accuracy'], label=f'{model_type.upper()} - Validation')
    
    plt.subplot(2, 1, 1)
    plt.title('Learning Curves - Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    
    plt.subplot(2, 1, 2)
    plt.title('Learning Curves - Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves_comparison.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_roc_curves()
    plot_learning_curves()