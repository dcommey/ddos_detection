import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_learning_curves(save_dir='results'):
    model_types = ['gru', 'lstm', 'conv1d']
    
    for model_type in model_types:
        metrics = pd.read_csv(os.path.join(save_dir, f'{model_type}_training_metrics.csv'))
        
        epochs = list(range(1, len(metrics) + 1))

        data_loss = pd.DataFrame({
            'Epoch': epochs * 2,
            'Loss': metrics['Train Loss'].tolist() + metrics['Validation Loss'].tolist(),
            'Type': ['Train Loss'] * len(metrics) + ['Validation Loss'] * len(metrics)
        })

        data_accuracy = pd.DataFrame({
            'Epoch': epochs * 2,
            'Accuracy': metrics['Train Accuracy'].tolist() + metrics['Validation Accuracy'].tolist(),
            'Type': ['Train Accuracy'] * len(metrics) + ['Validation Accuracy'] * len(metrics)
        })

        sns.set(style="whitegrid")
        
        # Individual loss curve
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data_loss, x='Epoch', y='Loss', hue='Type', style='Type', markers=True, dashes=False, markersize=8, linewidth=2.5)
        plt.title(f'{model_type.upper()} Training and Validation Loss', weight='bold', color='0.2', fontsize=16)
        plt.xlabel('Epochs', weight='bold', color='0.2', fontsize=14)
        plt.ylabel('Loss', weight='bold', color='0.2', fontsize=14)
        plt.legend(frameon=False, prop={'weight':'bold', 'size':12}, labelcolor='0.2')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_type}_loss_curve.pdf'), format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Individual accuracy curve
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data_accuracy, x='Epoch', y='Accuracy', hue='Type', style='Type', markers=True, dashes=False, markersize=8, linewidth=2.5)
        plt.title(f'{model_type.upper()} Training and Validation Accuracy', weight='bold', color='0.2', fontsize=16)
        plt.xlabel('Epochs', weight='bold', color='0.2', fontsize=14)
        plt.ylabel('Accuracy', weight='bold', color='0.2', fontsize=14)
        plt.legend(frameon=False, prop={'weight':'bold', 'size':12}, labelcolor='0.2')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_type}_accuracy_curve.pdf'), format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Side-by-side loss and accuracy curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        sns.lineplot(data=data_loss, x='Epoch', y='Loss', hue='Type', style='Type', markers=True, dashes=False, markersize=8, linewidth=2.5, ax=ax1)
        ax1.set_title(f'{model_type.upper()} Training and Validation Loss', weight='bold', color='0.2', fontsize=16)
        ax1.set_xlabel('Epochs', weight='bold', color='0.2', fontsize=14)
        ax1.set_ylabel('Loss', weight='bold', color='0.2', fontsize=14)
        ax1.legend(frameon=False, prop={'weight':'bold', 'size':12}, labelcolor='0.2')

        sns.lineplot(data=data_accuracy, x='Epoch', y='Accuracy', hue='Type', style='Type', markers=True, dashes=False, markersize=8, linewidth=2.5, ax=ax2)
        ax2.set_title(f'{model_type.upper()} Training and Validation Accuracy', weight='bold', color='0.2', fontsize=16)
        ax2.set_xlabel('Epochs', weight='bold', color='0.2', fontsize=14)
        ax2.set_ylabel('Accuracy', weight='bold', color='0.2', fontsize=14)
        ax2.legend(frameon=False, prop={'weight':'bold', 'size':12}, labelcolor='0.2')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_type}_learning_curves.pdf'), format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    plot_learning_curves()