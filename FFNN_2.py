import os
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import kurtosis
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def load_mat_files_from_folder(folder_path):
    """
    Lädt alle .mat-Dateien aus dem angegebenen Ordner und gibt sie als Liste zurück.
    """
    mat_data = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Der Pfad {folder_path} existiert nicht.")
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            mat_file_path = os.path.join(folder_path, file_name)
            try:
                mat_data.append(scipy.io.loadmat(mat_file_path))
            except Exception as e:
                print(f"Fehler beim Laden der Datei {mat_file_path}: {e}")
    
    return mat_data

def extract_data(mat_data, keys):
    """
    Extrahiert die Daten aus der Liste der geladenen .mat-Daten.
    """
    extracted_data = {key: [next((value for k, value in data.items() if key in k), None) for data in mat_data] for key in keys}
    return extracted_data

def create_time_vectors(data, sample_rate=10000):
    """
    Erstellt Zeitvektoren für die gegebene Datenliste basierend auf der Abtastrate.
    """
    return [np.arange(0, len(d)/sample_rate, 1/sample_rate) for d in data if d is not None]

# Funktion zur Berechnung von Wölbung, Standardabweichung und Maximum
def compute_features(data):
    kurtosis_values = []
    std_values = []
    max_values = []
    var_values = []
    rms_values = []
    
    interval_length = Umlaufzeit
    num_intervals = int(len(data) / (interval_length * 10000))
    for i in range(num_intervals):
        start_index = int(i * interval_length * 10000)
        end_index = int((i + 1) * interval_length * 10000)
        interval_data = data[start_index:end_index]
        
        # Berechnung der Wölbung
        interval_kurtosis = kurtosis(interval_data)
        kurtosis_values.append(interval_kurtosis)
        
        # Berechnung der Standardabweichung
        interval_std = np.std(interval_data)
        std_values.append(interval_std)

        # Berechnung des Maximums
        interval_max = np.max(interval_data)
        max_values.append(interval_max)
        
        # Berechnung der Varianz
        interval_var = np.var(interval_data)
        var_values.append(interval_var)
        
        #Berechnung der RMS (Root Mean Square)
        interval_rms = np.sqrt(np.mean(data**2))
        rms_values.append(interval_rms)
        
        features = [
            np.array(kurtosis_values), 
            np.array(std_values).reshape(-1, 1), 
            np.array(max_values).reshape(-1, 1),
            np.array(var_values).reshape(-1, 1),
            np.array(rms_values).reshape(-1, 1)
                    ]
        
        
    return features

# Funktion zur Normalisierung der Merkmale
def normalize_features(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Daten zusammenführen
def prepare_data(features_dict, labels_dict):
    all_data = []
    all_labels = []
    for key in features_dict:
        base_key = key.split('_')[0]  # Extrahieren Sie den Basis-Key (z.B. 'IR', 'B', etc.)
        num_features = len(features_dict[key])  # Anzahl der Merkmale bestimmen
        for i in range(len(features_dict[key][0])):
            feature_vector = [features_dict[key][f][i] for f in range(num_features)]
            all_data.append(feature_vector)
            all_labels.append(labels_dict[base_key])  # Verwenden Sie den Basis-Key für das Label
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    return all_data, all_labels

# Definition des neuronalen Netzes
class FFNN(pl.LightningModule):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FFNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('test_loss', loss)
        
        # Additional metrics
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        precision = precision_score(labels.cpu(), preds.cpu(), average='weighted')
        recall = recall_score(labels.cpu(), preds.cpu(), average='weighted')
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        
        self.log('test_acc', acc)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
        
        return {'test_loss': loss, 'test_acc': acc, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)


# Pfade zu den Ordnern mit den .mat-Dateien
folder_paths_FL = {
    'IR': r'.\Fehlerhafte_Lager\IR',
    'B': r'.\Fehlerhafte_Lager\B',
    'OR3': r'.\Fehlerhafte_Lager\OR@3',
    'OR6': r'.\Fehlerhafte_Lager\OR@6',
    'OR12': r'.\Fehlerhafte_Lager\OR@12'
}
folder_path_NL = r'.\Normale_Lager\NL'

# Laden der Daten
mat_data_FL = {key: load_mat_files_from_folder(path) for key, path in folder_paths_FL.items()}
mat_data_NL = load_mat_files_from_folder(folder_path_NL)

# Schlüssel zum Extrahieren
keys = ['DE_time', 'FE_time']

# Extrahieren der Daten
extracted_NL_data = extract_data(mat_data_NL, keys)
extracted_FL_data = {key: extract_data(data, keys) for key, data in mat_data_FL.items()}

# Berechnung der Merkmale
Umlaufzeit = 0.033*3
features_NL_DE = {f"NL_{i+1}": compute_features(data) for i, data in enumerate(extracted_NL_data['DE_time']) if data is not None}
features_FL_DE = {f"{key}_{i+1}": compute_features(data) for key, dataset in extracted_FL_data.items() for i, data in enumerate(dataset['DE_time']) if data is not None}


# Beispielplot der Merkmale
plt.figure(figsize=(10, 6))
# Plotting normal Lager DE Merkmale
plt.plot(features_NL_DE["NL_2"][1][:400], "*", label='Normale Lager DE Std') 

# Plotting fehlerhafte Lager DE Merkmale 
plt.plot(features_FL_DE["IR_1"][1], "^", label='Fehlerhafte Lager IR DE Std') 
plt.plot(features_FL_DE["B_1"][1], "^", label='Fehlerhafte Lager B DE Std') 
plt.plot(features_FL_DE["OR3_1"][1], "^", label='Fehlerhafte Lager OR3 DE Std') 
plt.plot(features_FL_DE["OR6_1"][1], "^", label='Fehlerhafte Lager OR6 DE Std') 
plt.plot(features_FL_DE["OR12_1"][1], "^", label='Fehlerhafte Lager OR12 DE Std') 

plt.xlabel('Datenpunkt')
plt.ylabel('Merkmale')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
# Plotting normal Lager DE Merkmale
plt.plot(features_NL_DE["NL_2"][0][:400], "*", label='Normale Lager DE W') 

# Plotting fehlerhafte Lager DE Merkmale 
plt.plot(features_FL_DE["IR_1"][0], "^", label='Fehlerhafte Lager IR DE W') 
plt.plot(features_FL_DE["B_1"][0], "^", label='Fehlerhafte Lager B DE W') 
plt.plot(features_FL_DE["OR3_1"][0], "^", label='Fehlerhafte Lager OR3 DE W') 
plt.plot(features_FL_DE["OR6_1"][0], "^", label='Fehlerhafte Lager OR6 DE W') 
plt.plot(features_FL_DE["OR12_1"][0], "^", label='Fehlerhafte Lager OR12 DE W') 

plt.xlabel('Datenpunkt')
plt.ylabel('Merkmale')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
# Plotting normal Lager DE Merkmale
plt.plot(features_NL_DE["NL_2"][2][:400], "*", label='Normale Lager DE max') 

# Plotting fehlerhafte Lager DE Merkmale 
plt.plot(features_FL_DE["IR_1"][2], "^", label='Fehlerhafte Lager IR DE max') 
plt.plot(features_FL_DE["B_1"][2], "^", label='Fehlerhafte Lager B DE max') 
plt.plot(features_FL_DE["OR3_1"][2], "^", label='Fehlerhafte Lager OR3 DE max') 
plt.plot(features_FL_DE["OR6_1"][2], "^", label='Fehlerhafte Lager OR6 DE max') 
plt.plot(features_FL_DE["OR12_1"][2], "^", label='Fehlerhafte Lager OR12 DE max') 

plt.xlabel('Datenpunkt')
plt.ylabel('Merkmale')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
# Plotting normal Lager DE Merkmale
plt.plot(features_NL_DE["NL_2"][3][:400], "*", label='Normale Lager DE var') 

# Plotting fehlerhafte Lager DE Merkmale 
plt.plot(features_FL_DE["IR_1"][3], "^", label='Fehlerhafte Lager IR DE var') 
plt.plot(features_FL_DE["B_1"][3], "^", label='Fehlerhafte Lager B DE var') 
plt.plot(features_FL_DE["OR3_1"][3], "^", label='Fehlerhafte Lager OR3 DE var') 
plt.plot(features_FL_DE["OR6_1"][3], "^", label='Fehlerhafte Lager OR6 DE var') 
plt.plot(features_FL_DE["OR12_1"][3], "^", label='Fehlerhafte Lager OR12 DE var') 

plt.xlabel('Datenpunkt')
plt.ylabel('Merkmale')
plt.legend()
plt.grid(True)
plt.show()




# Labels für fehlerhafte Lager
labels_FL = {
    'IR': 0,
    'B': 1,
    'OR3': 2,
    'OR6': 3,
    'OR12': 4
}

# Daten zusammenführen und Tensoren erstellen
all_data_DE, all_labels_DE = prepare_data(features_FL_DE, labels_FL)
# all_data_DE = normalize_features(all_data_DE)

# Speicher freigeben
del mat_data_FL
del mat_data_NL
del extracted_NL_data
del extracted_FL_data
gc.collect()

# Tensoren erstellen
data_tensor_DE = torch.tensor(all_data_DE[:,:,0], dtype=torch.float32)
labels_tensor_DE = torch.tensor(all_labels_DE, dtype=torch.long)

# Dataset erstellen
dataset_DE = TensorDataset(data_tensor_DE, labels_tensor_DE)

# Aufteilen in Trainings-, Validierungs- und Testdatensätze
train_size_DE = int(0.6 * len(dataset_DE))
val_size_DE = int(0.2 * len(dataset_DE))
test_size_DE = len(dataset_DE) - train_size_DE - val_size_DE
train_dataset_DE, val_dataset_DE, test_dataset_DE = random_split(dataset_DE, [train_size_DE, val_size_DE, test_size_DE])

# DataLoader erstellen
batch_size = 45
batch_size_test = test_size_DE
train_loader_DE = DataLoader(train_dataset_DE, batch_size, shuffle=True)
val_loader_DE = DataLoader(val_dataset_DE, batch_size)
test_loader_DE = DataLoader(test_dataset_DE, batch_size_test)
#%%
# Hyperparameter definieren
input_size = all_data_DE.shape[1]  # Anzahl der Merkmale
hidden_sizes = [100, 50, 25]  # Beliebige Anzahl von versteckten Schichten
output_size = len(labels_FL)  # Anzahl der Klassen

# TensorBoard Logger einrichten
logger = TensorBoardLogger("tb_logs", name="FFNN")

# Modell instanziieren
model = FFNN(input_size, hidden_sizes, output_size)

# Training des Modells mit Early Stopping
max_epochs = 100
# trainer = pl.Trainer(
#     max_epochs=max_epochs,
#     logger=logger,
#     callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=7)]
# )
# trainer.fit(model, train_loader_DE, val_loader_DE)

# Training des Modells
trainer = pl.Trainer(max_epochs = max_epochs, logger=logger)
trainer.fit(model, train_loader_DE, val_loader_DE)

# Funktion zum Testen des Modells
def test_model(model, test_loader, title):
    test_data, test_labels = next(iter(test_loader))
    with torch.no_grad():
        outputs = model(test_data)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == test_labels).float().mean()
        print(f'{title} Accuracy: {accuracy.item():.4f}')
        
        # Plot der Vorhersagen gegenüber den tatsächlichen Labels
        Intervall_A , Intervall_E = [0, 50]
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(test_labels))[Intervall_A:Intervall_E], test_labels.numpy()[Intervall_A:Intervall_E], color='blue', label='Actual Labels')
        plt.scatter(range(len(predictions))[Intervall_A:Intervall_E], predictions.numpy()[Intervall_A:Intervall_E], color='red', marker='x', label='Predicted Labels')
        plt.xlabel('Datenpunkt')
        plt.ylabel('Label')
        plt.title(f'Vorhersagen des Modells gegenüber den tatsächlichen Labels ({title})')
        plt.legend()
        plt.show()

# Testen des Modells
trainer.test(model, test_loader_DE)
test_model(model, test_loader_DE, "Test_DE")

