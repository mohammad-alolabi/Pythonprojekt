import gc
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from data_processing import load_mat_files_from_folder, extract_data, create_time_vectors, plot_multiple_measurements
from feature_engineering_and_model_2 import compute_features, prepare_data, model_2, test_model

# Konfigurierbare Parameter
Umlaufzeit = 0.033 * 3
train_size, val_size = 0.6, 0.2
batch_size = 40
max_epochs = 100
learning_rate = 0.0001
hidden_sizes = [200, 200, 200, 200]


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

# Berechnung der Merkmale des normalen Lagers DE
features_NL_DE = {f"NL_{i+1}": compute_features(data, Umlaufzeit) for i, data in enumerate(extracted_NL_data['DE_time']) if data is not None}
# Berechnung der Merkmale des normalen Lagers FE
features_NL_FE = {f"NL_{i+1}": compute_features(data, Umlaufzeit) for i, data in enumerate(extracted_NL_data['FE_time']) if data is not None}
# Berechnung der Merkmale des fehlerhaften Lagers DE
features_FL_DE = {f"{key}_{i+1}": compute_features(data, Umlaufzeit) for key, dataset in extracted_FL_data.items() for i, data in enumerate(dataset['DE_time']) if data is not None}
# Berechnung der Merkmale des fehlerhaften Lagers FE
features_FL_FE = {f"{key}_{i+1}": compute_features(data, Umlaufzeit) for key, dataset in extracted_FL_data.items() for i, data in enumerate(dataset['FE_time']) if data is not None}

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
all_data_FE, all_labels_FE = prepare_data(features_FL_FE, labels_FL)

# Speicher freigeben
del mat_data_FL
del mat_data_NL
del extracted_NL_data
del extracted_FL_data
gc.collect()

# Tensoren erstellen
data_tensor_DE = torch.tensor(all_data_DE[:, :, 0], dtype=torch.float32)
labels_tensor_DE = torch.tensor(all_labels_DE, dtype=torch.long)
data_tensor_FE = torch.tensor(all_data_FE[:, :, 0], dtype=torch.float32)
labels_tensor_FE = torch.tensor(all_labels_FE, dtype=torch.long)

# Dataset erstellen
dataset_DE = TensorDataset(data_tensor_DE, labels_tensor_DE)
dataset_FE = TensorDataset(data_tensor_FE, labels_tensor_FE)

# Aufteilen in Trainings-, Validierungs- und Testdatensätze
train_size_DE = int(train_size * len(dataset_DE))
val_size_DE = int(val_size * len(dataset_DE))
test_size_DE = len(dataset_DE) - train_size_DE - val_size_DE
train_dataset_DE, val_dataset_DE, test_dataset_DE = random_split(dataset_DE, [train_size_DE, val_size_DE, test_size_DE])

train_size_FE = int(train_size * len(dataset_FE))
val_size_FE = int(val_size * len(dataset_FE))
test_size_FE = len(dataset_FE) - train_size_FE - val_size_FE
train_dataset_FE, val_dataset_FE, test_dataset_FE = random_split(dataset_FE, [train_size_FE, val_size_FE, test_size_FE])
#%%
# DataLoader erstellen
batch_size_test = test_size_DE
train_loader_DE = DataLoader(train_dataset_DE, batch_size, shuffle=True)
val_loader_DE = DataLoader(val_dataset_DE, batch_size)
test_loader_DE = DataLoader(test_dataset_DE, batch_size_test)
batch_size_FE = test_size_FE
train_loader_FE = DataLoader(train_dataset_FE, batch_size, shuffle=True)
val_loader_FE = DataLoader(val_dataset_FE, batch_size)
test_loader_FE = DataLoader(test_dataset_FE, batch_size_FE)

# Hyperparameter definieren
input_size = all_data_DE.shape[1]  # Anzahl der Merkmale
output_size = len(labels_FL)       # Anzahl der Klassen bzw. die labls

# TensorBoard Logger einrichten
logger = TensorBoardLogger("tb_logs", name="model_2")

# Modell instanziieren
model = model_2(input_size, hidden_sizes, output_size, learning_rate)


# Training des Modells
trainer = pl.Trainer(max_epochs = max_epochs, logger=logger)
trainer.fit(model, train_loader_DE, val_loader_DE)
# trainer.fit(model, train_loader_FE, val_loader_FE)

# Testen des Modells
trainer.test(model, test_loader_DE)
# trainer.test(model, test_loader_FE)
test_model(model, test_loader_DE, "Test_DE")
test_model(model, test_loader_FE, "Test_FE")
