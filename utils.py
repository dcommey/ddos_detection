import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report, precision_score, recall_score, f1_score, precision_recall_curve
import os

def plot_metrics(metrics, model_type, save_dir='results'):
    epochs = list(range(1, len(metrics['client_train_loss']) + 1))

    data_loss = pd.DataFrame({
        'Epoch': epochs * 2,
        'Loss': metrics['client_train_loss'] + metrics['global_val_loss'],
        'Type': ['Train Loss'] * len(metrics['client_train_loss']) + ['Validation Loss'] * len(metrics['global_val_loss'])
    })

    data_accuracy = pd.DataFrame({
        'Epoch': epochs * 2,
        'Accuracy': metrics['client_train_accuracy'] + metrics['global_val_accuracy'],
        'Type': ['Train Accuracy'] * len(metrics['client_train_accuracy']) + ['Validation Accuracy'] * len(metrics['global_val_accuracy'])
    })

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.lineplot(data=data_loss, x='Epoch', y='Loss', hue='Type', style='Type', markers=True, dashes=False, markersize=8, linewidth=2.5)
    plt.title(f'{model_type.upper()} Training and Validation Loss', weight='bold', color='0.2', fontsize=14)
    plt.xlabel('Epochs', weight='bold', color='0.2', fontsize=14)
    plt.ylabel('Loss', weight='bold', color='0.2', fontsize=14)

    plt.subplot(1, 2, 2)
    sns.lineplot(data=data_accuracy, x='Epoch', y='Accuracy', hue='Type', style='Type', markers=True, dashes=False, markersize=8, linewidth=2.5)
    plt.title(f'{model_type.upper()} Training and Validation Accuracy', weight='bold', color='0.2', fontsize=14)
    plt.xlabel('Epochs', weight='bold', color='0.2', fontsize=14)
    plt.ylabel('Accuracy', weight='bold', color='0.2', fontsize=14)

    plt.legend(frameon=False, prop={'weight':'bold', 'size':12}, labelcolor='0.2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_type}_training_validation_metrics.png'), bbox_inches='tight', dpi=300)
    plt.close()

def evaluate_model(model, test_features, test_labels, model_type, save_dir='results'):
    y_pred_probabilities = model.predict(test_features)
    y_pred_classes = np.argmax(y_pred_probabilities, axis=1)
    y_true_classes = np.argmax(test_labels, axis=1)

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    print(f"{model_type.upper()} Accuracy: {accuracy:.4f}")

    report = classification_report(y_true_classes, y_pred_classes)
    print(f"{model_type.upper()} Classification Report:\n", report)

    # Save classification report
    with open(os.path.join(save_dir, f'{model_type}_classification_report.txt'), 'w') as f:
        f.write(report)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_type.upper()} Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted labels', fontsize=12)
    plt.ylabel('True labels', fontsize=12)
    plt.savefig(os.path.join(save_dir, f'{model_type}_confusion_matrix.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Save ROC curve data
    plt.figure(figsize=(10, 8))
    for i in range(test_labels.shape[1]):
        fpr, tpr, _ = roc_curve(test_labels[:, i], y_pred_probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        # Save ROC data for each class separately
        pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(
            os.path.join(save_dir, f'{model_type}_roc_data_class_{i}.csv'), index=False
        )
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_type.upper()} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, f'{model_type}_roc_curve.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # F1 Scores
    plot_f1_scores(y_true_classes, y_pred_classes, model_type, range(test_labels.shape[1]), save_dir)

    # Precision-Recall Curve
    plot_precision_recall_curve(test_labels, y_pred_probabilities, model_type, save_dir)

def save_metrics_to_csv(metrics, model_type, save_dir='results'):
    df = pd.DataFrame({
        'Epoch': range(1, len(metrics['client_train_loss']) + 1),
        'Train Loss': metrics['client_train_loss'],
        'Train Accuracy': metrics['client_train_accuracy'],
        'Validation Loss': metrics['global_val_loss'],
        'Validation Accuracy': metrics['global_val_accuracy'],
        'Epoch Time': metrics['epoch_times']
    })
    df.to_csv(os.path.join(save_dir, f'{model_type}_training_metrics.csv'), index=False)
    
    # Save total training time
    with open(os.path.join(save_dir, f'{model_type}_total_training_time.txt'), 'w') as f:
        f.write(f"Total training time: {metrics['total_training_time']:.2f} seconds")

def plot_f1_scores(y_true, y_pred, model_type, labels, save_dir='results'):
    f1_scores = f1_score(y_true, y_pred, average=None)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=f1_scores)
    plt.title(f'{model_type.upper()} F1 Scores by Class', fontsize=16)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_type}_f1_scores.png'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_prob, model_type, save_dir='results'):
    n_classes = y_true.shape[1]
    precision = dict()
    recall = dict()
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {i}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_type.upper()} Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_dir, f'{model_type}_precision_recall_curve.png'), bbox_inches='tight', dpi=300)
    plt.close()

def save_model_summary(model, model_type, save_dir='results'):
    with open(os.path.join(save_dir, f'{model_type}_model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def plot_training_time(metrics, model_type, save_dir='results'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics['epoch_times']) + 1), metrics['epoch_times'])
    plt.title(f'{model_type.upper()} Training Time per Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.savefig(os.path.join(save_dir, f'{model_type}_training_time.png'), bbox_inches='tight', dpi=300)
    plt.close()

def compare_models(model_metrics, save_dir='results'):
    # Compare final validation accuracies
    final_accuracies = {model: metrics['global_val_accuracy'][-1] for model, metrics in model_metrics.items()}
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(final_accuracies.keys()), y=list(final_accuracies.values()))
    plt.title('Final Validation Accuracy Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.savefig(os.path.join(save_dir, 'model_comparison_accuracy.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Compare training times
    training_times = {model: metrics['total_training_time'] for model, metrics in model_metrics.items()}
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(training_times.keys()), y=list(training_times.values()))
    plt.title('Total Training Time Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.savefig(os.path.join(save_dir, 'model_comparison_time.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Save comparison results to CSV
    comparison_df = pd.DataFrame({
        'Model': list(final_accuracies.keys()),
        'Final Validation Accuracy': list(final_accuracies.values()),
        'Total Training Time': list(training_times.values())
    })
    comparison_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)

def plot_all_metrics(model_metrics, save_dir='results'):
    # Plot training and validation loss for all models
    plt.figure(figsize=(12, 6))
    for model, metrics in model_metrics.items():
        plt.plot(metrics['client_train_loss'], label=f'{model} Train')
        plt.plot(metrics['global_val_loss'], label=f'{model} Validation')
    plt.title('Training and Validation Loss Comparison', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'all_models_loss_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Plot training and validation accuracy for all models
    plt.figure(figsize=(12, 6))
    for model, metrics in model_metrics.items():
        plt.plot(metrics['client_train_accuracy'], label=f'{model} Train')
        plt.plot(metrics['global_val_accuracy'], label=f'{model} Validation')
    plt.title('Training and Validation Accuracy Comparison', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'all_models_accuracy_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()

def generate_metrics_table(model_metrics, save_dir='results'):
    table_data = []
    for model_type, metrics in model_metrics.items():
        # Load the classification report
        with open(os.path.join(save_dir, f'{model_type}_classification_report.txt'), 'r') as f:
            report = f.read()
        
        # Extract overall metrics from the report
        lines = report.split('\n')
        weighted_avg = lines[-2].split()
        precision, recall, f1 = map(float, weighted_avg[2:5])
        
        # Get accuracy from the metrics
        accuracy = metrics['global_val_accuracy'][-1]
        
        table_data.append([model_type.upper(), accuracy, precision, recall, f1])
    
    # Create the table string
    table_string = "Model & Accuracy & Precision & Recall & F1 Score \\\\\n% \\hline\n"
    for row in table_data:
        table_string += f"{row[0]} & {row[1]:.4f} & {row[2]:.4f} & {row[3]:.4f} & {row[4]:.4f} \\\\\n"
    
    # Save the table
    with open(os.path.join(save_dir, 'model_metrics_table.txt'), 'w') as f:
        f.write(table_string)
    
    print("Metrics table saved to results/model_metrics_table.txt")