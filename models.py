from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Flatten, SimpleRNN, GRU, LSTM
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import time

def model_rnn(lr=1e-4, N=64, inshape=40, nclass=12):
    in1 = Input(shape=(inshape, 1))
    x = SimpleRNN(N, activation='tanh')(in1)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    out1 = Dense(nclass, activation='softmax')(x)
    model = Model(inputs=in1, outputs=out1)
    
    model.compile(optimizer=optimizers.Adam(lr),
                  loss=losses.categorical_crossentropy,
                  metrics=['acc'])
    return model

def model_gru(lr=1e-4, N=64, inshape=40, nclass=12):
    in1 = Input(shape=(inshape, 1))
    x = GRU(N, activation='tanh')(in1)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    out1 = Dense(nclass, activation='softmax')(x)
    model = Model(inputs=in1, outputs=out1)
    
    model.compile(optimizer=optimizers.Adam(lr),
                  loss=losses.categorical_crossentropy,
                  metrics=['acc'])
    return model

def model_lstm(lr=1e-4, N=64, inshape=40, nclass=12):
    in1 = Input(shape=(inshape, 1))
    x = LSTM(N, activation='tanh')(in1)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    out1 = Dense(nclass, activation='softmax')(x)
    model = Model(inputs=in1, outputs=out1)
    
    model.compile(optimizer=optimizers.Adam(lr),
                  loss=losses.categorical_crossentropy,
                  metrics=['acc'])
    return model

def model_conv1D(lr=1e-4, N=64, inshape=40, nclass=12):
    in1 = Input(shape=(inshape, 1))
    x = Conv1D(N, 3, padding='same', activation='relu')(in1)
    x = Conv1D(N, 3, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    out1 = Dense(nclass, activation='softmax')(x)
    model = Model(inputs=in1, outputs=out1)
    
    model.compile(optimizer=optimizers.Adam(lr),
                  loss=losses.categorical_crossentropy,
                  metrics=['acc'])
    return model

def init_server(model_type, lr, inshape, nclass):
    if model_type == 'rnn':
        model = model_rnn(lr=lr, N=64, inshape=inshape, nclass=nclass)
    elif model_type == 'gru':
        model = model_gru(lr=lr, N=64, inshape=inshape, nclass=nclass)
    elif model_type == 'lstm':
        model = model_lstm(lr=lr, N=64, inshape=inshape, nclass=nclass)
    elif model_type == 'conv1d':
        model = model_conv1D(lr=lr, N=64, inshape=inshape, nclass=nclass)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    server = {
        'model': model,
        'dataset_name': 'ddos_dataset'
    }
    return server

def aggregation_weighted_sum(server, clients):
    server_weights = server['model'].get_weights()
    total_weights = [np.zeros_like(weight) for weight in server_weights]
    total_samples = sum([len(client['training'][0]) for client in clients])
    
    for client in clients:
        client_weights = client['model'].get_weights()
        num_samples = len(client['training'][0])
        for i, weight in enumerate(client_weights):
            total_weights[i] += (num_samples / total_samples) * weight
    
    aggregated_model = clone_model(server['model'])
    aggregated_model.set_weights(total_weights)
    
    return aggregated_model

def FederatedTrain(clients, server, val_features, val_labels, epochs=10):
    main_save_dir = './ddos_saved_models'
    if not os.path.exists(main_save_dir):
        os.makedirs(main_save_dir)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    metrics = {
        'client_train_loss': [],
        'client_train_accuracy': [],
        'global_val_loss': [],
        'global_val_accuracy': [],
        'epoch_times': []
    }
    
    total_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        client_train_losses = []
        client_train_accuracies = []

        for client in clients:
            client_save_dir = os.path.join(main_save_dir, client["name"])
            if not os.path.exists(client_save_dir):
                os.makedirs(client_save_dir)
                
            client['model'] = clone_model(server['model'])
            client['model'].set_weights(server['model'].get_weights())
            
            client['model'].compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])

            client_features, client_labels = client['training']
            client_callbacks_list = [early_stopping, 
                                     ModelCheckpoint(os.path.join(client_save_dir, f'best_model_epoch_{epoch}.hdf5'),
                                                     save_best_only=True, monitor='val_loss', mode='min')]
            history = client['model'].fit(client_features, client_labels,
                                          validation_split=0.1,
                                          epochs=50, 
                                          batch_size=256,
                                          callbacks=client_callbacks_list,
                                          verbose=0)
            
            client_train_losses.append(history.history['loss'][-1])
            client_train_accuracies.append(history.history['accuracy'][-1])

            client['model'].load_weights(os.path.join(client_save_dir, f'best_model_epoch_{epoch}.hdf5'))

        metrics['client_train_loss'].append(np.mean(client_train_losses))
        metrics['client_train_accuracy'].append(np.mean(client_train_accuracies))

        server['model'] = aggregation_weighted_sum(server, clients)

        server['model'].compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
        
        val_loss, val_acc = server['model'].evaluate(val_features, val_labels, verbose=1)
        metrics['global_val_loss'].append(val_loss)
        metrics['global_val_accuracy'].append(val_acc)
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        metrics['epoch_times'].append(epoch_time)
        
        print(f"Global Validation Accuracy after epoch {epoch+1}: {val_acc:.4f}")
        print(f"Global Validation Loss after epoch {epoch+1}: {val_loss:.4f}")
        print(f"Epoch {epoch+1} time: {epoch_time:.2f} seconds")
    
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    metrics['total_training_time'] = total_training_time
    
    print(f"Total training time: {total_training_time:.2f} seconds")
    
    return metrics

