import os
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from scipy.stats import kurtosis
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt


# Pfad zum Ordner mit den .mat-Dateien
folder_path_FL = r'C:\Users\ha368\OneDrive\Desktop\Python\Datenset\Fehlerhafte_Lager'
folder_path_NL = r'C:\Users\ha368\OneDrive\Desktop\Python\Datenset\Normale_Lager'

# Liste, um die geladenen Fehlerhafte Lager-Daten zu speichern
mat_data_FL = []

# Alle Dateien im Ordner durchgehen
for file_name in os.listdir(folder_path_FL):
    if file_name.endswith('.mat'):  # Sicherstellen, dass die Datei eine .mat-Datei ist
        # Den vollständigen Pfad zur .mat-Datei erstellen
        mat_file_path_FL = os.path.join(folder_path_FL, file_name)
        
        # Datei laden und die Daten zur Liste hinzufügen
        mat_data_FL.append(scipy.io.loadmat(mat_file_path_FL))
 
# Liste, um die geladenen normale Lager-Daten zu speichern
mat_data_NL = []
        
for file_name in os.listdir(folder_path_NL):
    if file_name.endswith('.mat'):  # Sicherstellen, dass die Datei eine .mat-Datei ist
        # Den vollständigen Pfad zur .mat-Datei erstellen
        mat_file_path = os.path.join(folder_path_NL, file_name)
        
        # Datei laden und die Daten zur Liste hinzufügen
        mat_data_NL.append(scipy.io.loadmat(mat_file_path))        
        
        
# Zeitvektoren initialisieren
t_NL_DE = []
t_NL_FE = []

# Für die normale Lager-Daten
NL_data_DE = [next(value for key, value in mat_data_NL[i].items() if 'DE_time' in key) for i in range(len(mat_data_NL))]
NL_data_FE = [next(value for key, value in mat_data_NL[i].items() if 'FE_time' in key) for i in range(len(mat_data_NL))]
NL_data_RPM = [next(value for key, value in mat_data_NL[i].items() if 'RPM' in key) for i in range(len(mat_data_NL))]

# Liste der Indizes sortieren basierend auf den 'RPM'-Werten in absteigender Reihenfolge
sorted_indices_NL = sorted(range(len(NL_data_RPM)), key=lambda k: NL_data_RPM[k], reverse=True)

# Daten neu anordnen basierend auf den sortierten Indizes (Abstiegsortierung nach der RPM)
NL_data_DE = [NL_data_DE[i] for i in sorted_indices_NL]
NL_data_FE = [NL_data_FE[i] for i in sorted_indices_NL]

for data in NL_data_DE:
    # Zeitvektor erstellen und hinzufügen
    t_NL_DE.append(np.arange(0, len(data)/10000, 0.0001))
    

for data in NL_data_FE:
    # Zeitvektor erstellen und hinzufügen
    t_NL_FE.append(np.arange(0, len(data)/10000, 0.0001))
    
    
# Zeitvektoren initialisieren
t_FL_DE = []
t_FL_FE = []    
    
# Für die Fehlerhafte Lager-Daten
FL_data_DE = [next(value for key, value in mat_data_FL[i].items() if 'DE_time' in key) for i in range(len(mat_data_FL))]
FL_data_FE = [next(value for key, value in mat_data_FL[i].items() if 'FE_time' in key) for i in range(len(mat_data_FL))]
FL_data_RPM = [next(value for key, value in mat_data_FL[i].items() if 'RPM' in key) for i in range(len(mat_data_FL))]

# Liste der Indizes sortieren basierend auf den 'RPM'-Werten in absteigender Reihenfolge
sorted_indices_FL = sorted(range(len(FL_data_RPM)), key=lambda k: FL_data_RPM[k], reverse=True)

# Daten neu anordnen basierend auf den sortierten Indizes (Abstiegsortierung nach der RPM)
FL_data_DE = [FL_data_DE[i] for i in sorted_indices_FL]
FL_data_FE = [FL_data_FE[i] for i in sorted_indices_FL]

for data in FL_data_DE:
    # Zeitvektor erstellen und hinzufügen
    t_FL_DE.append(np.arange(0, len(data)/10000, 0.0001))

for data in FL_data_FE:
    # Zeitvektor erstellen und hinzufügen
    t_FL_FE.append(np.arange(0, len(data)/10000, 0.0001))
    
# # plot erstellen
# plt.figure(figsize=(10, 6))  
# plt.plot(t_FL_FE[3], FL_data_FE[3], label='FL_FE_4') 
# plt.plot(t_FL_FE[0], FL_data_FE[0], label='FL_FE_1')  
# plt.plot(t_FL_FE[1], FL_data_FE[1], label='FL_FE_2')  
# plt.plot(t_FL_FE[2], FL_data_FE[2], label='FL_FE_3')  
# plt.plot(t_NL_FE[0], NL_data_FE[0], label='NL_FE_1')  
# # plt.plot(t_NL_DE[0], NL_data_DE[0], label='NL_DE_1')  
# plt.xlabel('Zeit')  
# plt.ylabel('Beschleunigung')  
# plt.legend()  
# plt.grid(True)  
# plt.show(block=True)      

# Funktion zur Berechnung von Wölbung und Standardabweichung
def compute_features(data):
    kurtosis_values = []
    std_values = []
    interval_length = 0.033
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
        
    return np.array(kurtosis_values), np.array(std_values).reshape(-1, 1)

# Berechnung der Merkmale des normalen Lagers
features_NL_DE_ = {f"{i+1}": compute_features(data) for i, data in enumerate(NL_data_DE)}

# Berechnung der Merkmale des fehlerhaften Lagers
features_FL_DE_ = {f"{i+1}": compute_features(data) for i, data in enumerate(FL_data_DE)}
# Berechnung der Merkmale des fehlerhaften Lagers FE
features_FL_FE_ = {f"{i+1}": compute_features(data) for i, data in enumerate(FL_data_FE)}

# Daten zusammenführen
all_data = []
all_labels = []
all_data_FE = []
all_labels_FE = []

for key in features_NL_DE_:
    for i in range(len(features_NL_DE_[key][0])):
        all_data.append([features_NL_DE_[key][0][i], features_NL_DE_[key][1][i]])
    all_labels.extend([0] * len(features_NL_DE_[key][0]))

for key in features_FL_DE_:
    for i in range(len(features_FL_DE_[key][0])):
        all_data.append([features_FL_DE_[key][0][i], features_FL_DE_[key][1][i]])
    all_labels.extend([1] * len(features_FL_DE_[key][0]))

for key in features_FL_FE_:
    for i in range(len(features_FL_FE_[key][0])):
        all_data_FE.append([features_FL_FE_[key][0][i], features_FL_FE_[key][1][i]])
    all_labels_FE.extend([1] * len(features_FL_FE_[key][0]))    

all_data = np.array(all_data)  
all_labels = np.array(all_labels).reshape(-1, 1)    

all_data_FE = np.array(all_data_FE)  
all_labels_FE = np.array(all_labels_FE).reshape(-1, 1)    

# Tensoren erstellen
data_tensor = torch.tensor(all_data[:,:,0], dtype=torch.float32)
labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
data_tensor_FE = torch.tensor(all_data_FE[:,:,0], dtype=torch.float32)
labels_tensor_FE = torch.tensor(all_labels_FE, dtype=torch.float32)


# Dataset erstellen
dataset = TensorDataset(data_tensor, labels_tensor)
dataset_FE = TensorDataset(data_tensor_FE, labels_tensor_FE)

# Aufteilen in Trainings-, Validierungs- und Testdatensätze
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoader erstellen
train_loader = DataLoader(train_dataset, batch_size =15, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=15)
test_loader  = DataLoader(test_dataset,  batch_size=70)
# test_loader  = DataLoader(dataset_FE,  batch_size=70)

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
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = nn.BCELoss()(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = nn.BCELoss()(outputs, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = nn.BCELoss()(outputs, labels)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)

# Hyperparameter definieren
input_size = 2  # Da wir jetzt zwei Merkmale haben (Wölbung und Standardabweichung)
hidden_sizes = [50, 25]  # Beliebige Anzahl von versteckten Schichten
output_size = 1

# Modell instanziieren
model = FFNN(input_size, hidden_sizes, output_size)

# Training des Modells
trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, train_loader, val_loader)

# Testen des Modells
trainer.test(model, test_loader)

# Evaluation und grafische Darstellung der Ergebnisse
test_data, test_labels = next(iter(test_loader))

with torch.no_grad():
    predictions = (model(test_data) > 0.5).float()
    accuracy = (predictions == test_labels).float().mean()
    print(f'Test Accuracy: {accuracy.item():.32}')

    # Plot der Vorhersagen gegenüber den tatsächlichen Labels
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(test_labels)), test_labels.numpy(), color='blue', label='Actual Labels')
    plt.scatter(range(len(predictions)), predictions.numpy(), color='red', marker='x', label='Predicted Labels')
    plt.xlabel('Datenpunkt')
    plt.ylabel('Label')
    plt.title('Vorhersagen des Modells gegenüber den tatsächlichen Labels')
    plt.legend()
    plt.show()


# # Evaluation und grafische Darstellung der Ergebnisse
# test_data_FE, test_labels_FE = next(iter(test_loader_FE))

# with torch.no_grad():
#     predictions_FE = (model(test_data_FE) > 0.5).float()
#     accuracy_FE = (predictions_FE == test_labels_FE).float().mean()
#     print(f'Test_FE Accuracy: {accuracy_FE.item():.32}')

#     # Plot der Vorhersagen gegenüber den tatsächlichen Labels
#     plt.figure(figsize=(10, 6))
#     plt.scatter(range(len(test_labels_FE)), test_labels_FE.numpy(), color='blue', label='Actual Labels')
#     plt.scatter(range(len(predictions_FE)), predictions_FE.numpy(), color='red', marker='x', label='Predicted Labels')
#     plt.xlabel('Datenpunkt')
#     plt.ylabel('Label')
#     plt.title('Vorhersagen des Modells gegenüber den tatsächlichen Labels')
#     plt.legend()
#     plt.show()