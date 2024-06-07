import os
import gc
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from data_processing import load_mat_files_from_folder, extract_data, create_time_vectors, plot_multiple_measurements
from feature_engineering_and_model_1 import compute_features, prepare_data, model_1, test_model

# Konfigurierbare Parameter
Umlaufzeit = 0.033
train_size, val_size = 0.6, 0.2
batch_size = 230
max_epochs = 50
learning_rate = 0.01
hidden_sizes = [50, 25]  

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

# Erstellen der Zeitvektoren
t_NL_DE = create_time_vectors(extracted_NL_data['DE_time'])
t_NL_FE = create_time_vectors(extracted_NL_data['FE_time'])

t_FL_DE = {key: create_time_vectors(data['DE_time']) for key, data in extracted_FL_data.items()}
t_FL_FE = {key: create_time_vectors(data['FE_time']) for key, data in extracted_FL_data.items()}

# Beispiel: Plotten der ersten normalen Lager DE-Messung und der ersten fehlerhaften Lager DE-Messung für IR
plot_multiple_measurements(
    [t_FL_DE['OR6'][0], t_FL_DE['OR3'][0], t_FL_DE['IR'][0], t_FL_DE['OR12'][0], t_FL_DE['B'][0], t_NL_DE[1]],
    [extracted_FL_data['OR6']['DE_time'][0], extracted_FL_data['OR3']['DE_time'][0], 
     extracted_FL_data['IR']['DE_time'][0], extracted_FL_data['OR12']['DE_time'][0], 
     extracted_FL_data['B']['DE_time'][0], extracted_NL_data['DE_time'][1]],
    ['Fehlerhafte Lager OR6 DE-Messung', 'Fehlerhafte Lager OR3 DE-Messung', 'Fehlerhafte Lager IR DE-Messung', 
     'Fehlerhafte Lager OR12 DE-Messung', 'Fehlerhafte Lager B DE-Messung', 'Normale Lager DE-Messung']
)

# Berechnung der Merkmale des normalen Lagers DE
features_NL_DE = {f"NL_{i+1}": compute_features(data, Umlaufzeit) for i, data in enumerate(extracted_NL_data['DE_time']) if data is not None}
# Berechnung der Merkmale des normalen Lagers FE
features_NL_FE = {f"NL_{i+1}": compute_features(data, Umlaufzeit) for i, data in enumerate(extracted_NL_data['FE_time']) if data is not None}
# Berechnung der Merkmale des fehlerhaften Lagers DE
features_FL_DE = {f"{key}_{i+1}": compute_features(data, Umlaufzeit) for key, dataset in extracted_FL_data.items() for i, data in enumerate(dataset['DE_time']) if data is not None}
# Berechnung der Merkmale des fehlerhaften Lagers FE
features_FL_FE = {f"{key}_{i+1}": compute_features(data, Umlaufzeit) for key, dataset in extracted_FL_data.items() for i, data in enumerate(dataset['FE_time']) if data is not None}

# Speicher freigeben
del mat_data_FL
del mat_data_NL
del extracted_NL_data
del extracted_FL_data
del t_FL_DE, t_FL_FE, t_NL_DE, t_NL_FE, keys
gc.collect()

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
plt.plot(features_NL_DE["NL_2"][0][:400], "*", label='Normale Lager DE Wölbung') 

# Plotting fehlerhafte Lager DE Merkmale 
plt.plot(features_FL_DE["IR_1"][0], "^", label='Fehlerhafte Lager IR DE Wölbung') 
plt.plot(features_FL_DE["B_1"][0], "^", label='Fehlerhafte Lager B DE Wölbung') 
plt.plot(features_FL_DE["OR3_1"][0], "^", label='Fehlerhafte Lager OR3 DE Wölbung') 
plt.plot(features_FL_DE["OR6_1"][0], "^", label='Fehlerhafte Lager OR6 DE Wölbung') 
plt.plot(features_FL_DE["OR12_1"][0], "^", label='Fehlerhafte Lager OR12 DE Wölbung') 

plt.xlabel('Datenpunkt')
plt.ylabel('Merkmale')
plt.legend()
plt.grid(True)
plt.show()

# Daten zusammenführen und Tensoren erstellen
all_data_DE, all_labels_DE = prepare_data(features_NL_DE, features_FL_DE)
all_data_FE, all_labels_FE = prepare_data(features_NL_FE, features_FL_FE)

# Tensoren erstellen
data_tensor_DE = torch.tensor(all_data_DE[:, :, 0], dtype=torch.float32)
labels_tensor_DE = torch.tensor(all_labels_DE, dtype=torch.float32)
data_tensor_FE = torch.tensor(all_data_FE[:, :, 0], dtype=torch.float32)
labels_tensor_FE = torch.tensor(all_labels_FE, dtype=torch.float32)

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

# DataLoader erstellen
batch_size_DE = test_size_DE
train_loader_DE = DataLoader(train_dataset_DE, batch_size, shuffle=True)
val_loader_DE = DataLoader(val_dataset_DE, batch_size)
test_loader_DE = DataLoader(test_dataset_DE, batch_size_DE)
batch_size_FE = test_size_FE
train_loader_FE = DataLoader(train_dataset_FE, batch_size, shuffle=True)
val_loader_FE = DataLoader(val_dataset_FE, batch_size)
test_loader_FE = DataLoader(test_dataset_FE, batch_size_FE)

# Hyperparameter definieren
input_size = all_data_DE.shape[1]  
output_size = 1

# TensorBoard Logger einrichten
logger = TensorBoardLogger("tb_logs", name="model_1")

# Modell instanziieren
model = model_1(input_size, hidden_sizes, output_size, learning_rate)

# Training des Modells mit Early Stopping
trainer = pl.Trainer(
    max_epochs=max_epochs,
    logger=logger,
    callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=7)]
)
trainer.fit(model, train_loader_DE, val_loader_DE)

# Testen des Modells mit verschiedenen Testdaten
trainer.test(model, test_loader_DE)
test_model(model, test_loader_DE, "Test")
test_model(model, test_loader_FE, "Test_FE")
