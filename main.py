import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight

from models import init_server, FederatedTrain
from data_processing import preprocess_data, split_data_among_clients, initialize_clients
from utils import (plot_metrics, save_metrics_to_csv, evaluate_model, 
                   plot_training_time, compare_models, plot_all_metrics, 
                   save_model_summary)

def main():
    # Load and preprocess data
    train_data = pd.read_csv('ddos_data_train.csv')
    test_data = pd.read_csv('ddos_data_test.csv')
    
    train_data, test_data, label_mapping = preprocess_data(train_data, test_data)
    
    # Set up experiment parameters
    num_clients = 10
    inshape = 30
    nclass = len(label_mapping)
    lr = 1e-4
    epochs = 50
    
    # Prepare data
    train_features, val_features, train_labels, val_labels, test_features, test_labels = split_data(train_data, test_data, inshape, nclass)
    
    clients = split_data_among_clients(train_data, num_clients)
    initialize_clients(clients, inshape, nclass)
    
    # List of models to compare
    model_types = ['rnn', 'gru', 'lstm', 'conv1d']
    
    all_metrics = {}

    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} model")
        
        # Initialize server
        server = init_server(model_type, lr, inshape, nclass)
        
        # Save model summary
        save_model_summary(server['model'], model_type)
        
        # Train the federated model
        metrics = FederatedTrain(clients, server, val_features, val_labels, epochs=epochs)
        
        # Evaluate and save results
        evaluate_model(server['model'], test_features, test_labels, model_type)
        plot_metrics(metrics, model_type)
        plot_training_time(metrics, model_type)
        save_metrics_to_csv(metrics, model_type)

        all_metrics[model_type] = metrics

    # Compare all models
    compare_models(all_metrics)
    plot_all_metrics(all_metrics)

def split_data(train_data, test_data, inshape, nclass):
    train_features = train_data[:, :-1].astype('float32').reshape(-1, inshape, 1)
    test_features = test_data[:, :-1].astype('float32').reshape(-1, inshape, 1)
    
    train_labels = to_categorical(train_data[:, -1].astype('int32'), nclass)
    test_labels = to_categorical(test_data[:, -1].astype('int32'), nclass)
    
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels, test_size=0.20, stratify=train_labels, random_state=42
    )
    
    return train_features, val_features, train_labels, val_labels, test_features, test_labels

if __name__ == "__main__":
    main()