import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preprocess_data(train_data, test_data):
    label_mapping = {
        'BENIGN': 0,
        'DrDoS_MSSQL': 1,
        'DrDoS_NTP': 2,
        'DrDoS_SNMP': 3,
        'DrDoS_UDP': 4,
        'Syn': 5,
        'TFTP': 6,
    }
    
    classes_to_remove = [1, 4, 2, 7, 11]  # Corresponding to DrDoS_DNS, DrDoS_NetBIOS, DrDoS_LDAP, DrDoS_SSDP, UDPLag
    
    train_data = train_data[~train_data.iloc[:, -1].isin(classes_to_remove)]
    test_data = test_data[~test_data.iloc[:, -1].isin(classes_to_remove)]
    
    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()
    
    old_to_new_labels = {0: 0, 3: 1, 5: 2, 6: 3, 8: 4, 9: 5, 10: 6}
    train_data[:, -1] = np.vectorize(old_to_new_labels.get)(train_data[:, -1])
    test_data[:, -1] = np.vectorize(old_to_new_labels.get)(test_data[:, -1])
    
    return train_data, test_data, label_mapping

def split_data(train_data, test_data, inshape, nclass):
    train_features = train_data[:, :-1].astype('float32').reshape(-1, inshape, 1)
    test_features = test_data[:, :-1].astype('float32').reshape(-1, inshape, 1)
    
    train_labels = to_categorical(train_data[:, -1].astype('int32'), nclass)
    test_labels = to_categorical(test_data[:, -1].astype('int32'), nclass)
    
    train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.20, stratify=train_labels, random_state=42)
    
    return train_features, val_features, train_labels, val_labels, test_features, test_labels

def split_data_among_clients(data, num_clients):
    np.random.shuffle(data)
    
    chunk_size = len(data) // num_clients
    client_data = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_clients)]
    
    clients = []
    for i in range(num_clients):
        client = {
            'name': 'client_' + str(i),
            'data': client_data[i]
        }
        clients.append(client)
        
    return clients

def initialize_clients(clients, inshape, nclass):
    for client in clients:
        data = client['data']
        features = data[:, :-1].astype('float32').reshape(-1, inshape, 1)
        labels = to_categorical(data[:, -1].astype('int32'), nclass)
        client['training'] = (features, labels)
        client['model'] = None