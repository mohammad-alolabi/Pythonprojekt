import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import kurtosis
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.preprocessing import MinMaxScaler


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

# # Plot für normale Lager-Daten erstellen
# plt.figure(figsize=(10, 6))
# for i, data in enumerate(NL_data_DE):
#     plt.plot(t_NL_DE[i], data, label=f'NL_{i+1}')
# plt.xlabel('Zeit')
# plt.ylabel('Signalwert')
# plt.title('Plot der normalen Lager-Daten (nach absteigendem RPM)')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Plot für fehlerhafte Lager-Daten erstellen
# plt.figure(figsize=(10, 6))
# for i, data in enumerate(FL_data_DE):
#     plt.plot(t_FL_DE[i], data, label=f'FL_{i+1}')
# plt.xlabel('Zeit')
# plt.ylabel('Signalwert')
# plt.title('Plot der fehlerhaften Lager-Daten')
# plt.grid(True)
# plt.legend()
# plt.show()

# plot erstellen
plt.figure(figsize=(10, 6))  
plt.plot(t_FL_DE[3], FL_data_DE[3], label='FL_DE_4') 
plt.plot(t_FL_DE[1], FL_data_DE[1], label='FL_DE_2')  
plt.plot(t_FL_DE[0], FL_data_DE[0], label='FL_DE_1')  
plt.plot(t_FL_DE[2], FL_data_DE[2], label='FL_DE_3')  
plt.plot(t_NL_DE[0], NL_data_DE[0], label='NL_DE_1')  
plt.xlabel('Zeit')  
plt.ylabel('Beschleunigung')  
plt.legend()  
plt.grid(True)  
plt.show(block=True)  

#parameter des Lagers DE und FE
D_DE = 52  # Außendurchmesser der Lager in Antriebsseite (mm)
D_FE = 40  # Außendurchmesser der Lager in Lufterseite (mm)
U_DE = math.pi * D_DE  # Umfang berechnen in Antriebsseite (mm)
U_FE = math.pi * D_FE  # Umfang berechnen in Lufterseite (mm)

v_NL_DE = []
v_NL_FE = []
T_NL_DE = []
T_NL_FE = []
for rpm in NL_data_RPM:
    v_NL_DE.append(math.pi * D_DE * rpm / 60)  # Umfanggeschwindigkeit in Antriebsseite (mm/s)
    T_NL_DE.append(U_DE / v_NL_DE[-1])         # Umlaufzeit in Antriebsseite (s)
    v_NL_FE.append(math.pi * D_FE * rpm / 60)  # Umfanggeschwindigkeit in Lufterseite (mm/s)
    T_NL_FE.append(U_FE / v_NL_FE[-1])         # Umlaufzeit in Lufterseite (s)
    
v_FL_DE = []
v_FL_FE = []
T_FL_DE = []
T_FL_FE = []
for rpm in FL_data_RPM:
    v_FL_DE.append(math.pi * D_DE * rpm / 60)  # Umfanggeschwindigkeit in Antriebsseite (mm/s)
    T_FL_DE.append(U_DE / v_FL_DE[-1])         # Umlaufzeit in Antriebsseite (s)
    v_FL_FE.append(math.pi * D_FE * rpm / 60)  # Umfanggeschwindigkeit in Lufterseite (mm/s)
    T_FL_FE.append(U_FE / v_FL_FE[-1])         # Umlaufzeit in Lufterseite (s)


# Merkmale extraieren (Wölbung)
def compute_kurtosis(data):
    # Initialisiere eine Liste, um die Wölbung für jedes Intervall zu speichern
    kurtosis_values = []
    # Angenomene Umlaufzeit
    interval_length = 0.033
    # Berechne die Anzahl der Intervalle basierend auf der Gesamtdauer der Daten und der Intervalllänge
    num_intervals = int(len(data) / (interval_length * 10000))
    # Schleife über jedes Intervall
    for i in range(num_intervals):
        # Bestimme den Start- und Endindex des aktuellen Intervalls
        start_index = int(i * interval_length * 10000)
        end_index = int((i + 1) * interval_length * 10000)
    
        # Extrahiere die Daten für das aktuelle Intervall für jede Data in FL_data_DE
        interval_data = data[start_index:end_index]
    
        # Berechne die Wölbung (Kurtosis) für das aktuelle Intervall und füge sie der Liste hinzu
        interval_kurtosis = kurtosis(interval_data) 
        kurtosis_values.append(interval_kurtosis)
    return np.array(kurtosis_values)


# Berechnung der Merkmale des normalen Lagers
kurtosis_NL_DE_ = {}
for i, data in enumerate(NL_data_DE):
    kurtosis_NL_DE_[f"{i+1}"] = compute_kurtosis(data)

# Berechnung der Merkmale des fehlerhaften Lagers
kurtosis_FL_DE_ = {}
for i, data in enumerate(FL_data_DE):
    kurtosis_FL_DE_[f"{i+1}"] = compute_kurtosis(data)

# Plot der Wölbungswerte über die Zeitintervalle
plt.figure(figsize=(10, 6))
plt.plot(kurtosis_NL_DE_["1"], '*' , label=f'NL_DE_{1}')
# plt.plot(kurtosis_NL_DE_["2"], '*' , label=f'NL_DE_{2}')
# plt.plot(kurtosis_NL_DE_["3"], '*' , label=f'NL_DE_{3}')
# plt.plot(kurtosis_NL_DE_["4"], '*' , label=f'NL_DE_{4}')
plt.plot(kurtosis_FL_DE_["1"], '^' , label=f'FL_DE_{1}')
# plt.plot(kurtosis_FL_DE_["2"], '*' , label=f'FL_DE_{2}')
# plt.plot(kurtosis_FL_DE_["3"], '*' , label=f'FL_DE_{3}')
# plt.plot(kurtosis_FL_DE_["4"], '*' , label=f'FL_DE_{4}')
plt.xlabel('')
plt.ylabel('Wölbung')
plt.grid(True)
plt.legend()
plt.show()




# # Normalisierung der Merkmale
# scaler = MinMaxScaler()
# for key in kurtosis_NL_DE_:
#     kurtosis_NL_DE_[key] = scaler.fit_transform(kurtosis_NL_DE_[key].reshape(-1, 1)).flatten()
# for key in kurtosis_FL_DE_:
#     kurtosis_FL_DE_[key] = scaler.fit_transform(kurtosis_FL_DE_[key].reshape(-1, 1)).flatten()

# # Beispiel für den Zugriff auf normierte Wölbungswerte
# print(kurtosis_NL_DE_["1"])  # Normierte Wölbungswerte für das erste normale Lager
# print(kurtosis_FL_DE_["1"])  # Normierte Wölbungswerte für das erste fehlerhafte Lager


# # Normaeren der Merkmale
# spe_FL_kurtosis = [kurtosis_FL_DE_["1"], kurtosis_FL_DE_["2"], kurtosis_FL_DE_["3"], kurtosis_FL_DE_["4"], kurtosis_FL_DE_["5"], kurtosis_NL_DE_["1"]]
# xmin = min(spe_FL_kurtosis)
# xmax = max(spe_FL_kurtosis)
# for i, x in enumerate(spe_FL_kurtosis):
#     spe_FL_kurtosis[i] = (x - xmin) / (xmax - xmin)

# plt.figure(figsize=(10, 6))
# plt.plot(spe_FL_kurtosis, '*' , label=f'NL_DE_{1}')
# plt.xlabel('')
# plt.ylabel('Wölbung')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Beispielhaftes Merkmal für eine Klasse 
# feature_class1 = kurtosis_NL_DE_1[0:len(kurtosis_NL_DE_1)//2]
# feature_class2 = kurtosis_FL_DE_1[0:len(kurtosis_FL_DE_1)//2]
# feature_class3 = kurtosis_FL_DE_2
# feature_class4 = kurtosis_FL_DE_3
# feature_class5 = kurtosis_FL_DE_4

# # Mitteln über die Zeitdimension, um die Merkmale zu eindimensionalen Arrays zu machen
# mean_features_class1 = np.mean(feature_class1, axis=1)
# mean_features_class2 = np.mean(feature_class2, axis=1)
# mean_features_class3 = np.mean(feature_class3, axis=1)
# mean_features_class4 = np.mean(feature_class4, axis=1)
# mean_features_class5 = np.mean(feature_class5, axis=1)

# # Normalitätstests
# normality_test_class1 = shapiro(mean_features_class1)
# normality_test_class2 = shapiro(mean_features_class2)
# normality_test_class3 = shapiro(mean_features_class3)
# normality_test_class4 = shapiro(mean_features_class4)
# normality_test_class5 = shapiro(mean_features_class5)

# print("Shapiro-Wilk-Test für Klasse 1:", normality_test_class1)
# print("Shapiro-Wilk-Test für Klasse 2:", normality_test_class2)
# print("Shapiro-Wilk-Test für Klasse 3:", normality_test_class3)
# print("Shapiro-Wilk-Test für Klasse 4:", normality_test_class5)
# print("Shapiro-Wilk-Test für Klasse 5:", normality_test_class4)

# # Test auf Gleichheit der Kovarianzmatrizen
# levene_test = levene(mean_features_class1, mean_features_class2)#, mean_features_class3, mean_features_class4, mean_features_class5)
# print("Levene-Test für die Gleichheit der Kovarianzmatrizen:", levene_test)


# Definiere das Feedforward-Netzwerk
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # self.relu = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# class FFNN(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
#         super(FFNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(hidden_size2, output_size)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         out = self.sigmoid(out)
#         return out


# Training und Evaluation in einer Schleife
for key in kurtosis_FL_DE_:
    # Trainingsdaten für normale Lager
    train_data_NL_DE = torch.tensor(kurtosis_NL_DE_["1"][:200], dtype=torch.float32)
    train_labels_NL_DE = torch.zeros(len(train_data_NL_DE), dtype=torch.float32).view(-1, 1)  # Kennzeichnen Sie Normaldaten als 0

    # Testdaten für normale Lager
    test_data_NL_DE = torch.tensor(kurtosis_NL_DE_["1"][:200], dtype=torch.float32)
    test_labels_NL_DE = torch.zeros(len(test_data_NL_DE), dtype=torch.float32).view(-1, 1)  # Kennzeichnen Sie Normaldaten als 0

    # Trainingsdaten für fehlerhafte Lager
    train_data_FL_DE = torch.tensor(kurtosis_FL_DE_[key][:200], dtype=torch.float32)
    train_labels_FL_DE = torch.ones(len(train_data_FL_DE), dtype=torch.float32).view(-1, 1)  # Kennzeichnen Sie Fehlerdaten als 1

    # Testdaten für fehlerhafte Lager
    test_data_FL_DE = torch.tensor(kurtosis_FL_DE_[key][:200], dtype=torch.float32)
    test_labels_FL_DE = torch.ones(len(test_data_FL_DE), dtype=torch.float32).view(-1, 1)  # Kennzeichnen Sie Fehlerdaten als 1

  # Modell definieren
    input_size = len(train_data_NL_DE[0])
    hidden_size1 = input_size // 2
    hidden_size2 = input_size // 2
    output_size = 1
    model = FFNN(input_size, hidden_size1, hidden_size2, output_size)

    # Verlustfunktion und Optimierer definieren
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Trainingsdaten und Labels zusammenführen
    train_data = torch.cat((train_data_NL_DE, train_data_FL_DE), dim=0)
    train_labels = torch.cat((train_labels_NL_DE, train_labels_FL_DE), dim=0)

    # Testdaten und Labels zusammenführen
    test_data = torch.cat((test_data_NL_DE, test_data_FL_DE), dim=0)
    test_labels = torch.cat((test_labels_NL_DE, test_labels_FL_DE), dim=0)

    # Trainingsschleife
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward-Pass
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        
        # Backward-Pass und Optimierung
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    with torch.no_grad():
        predicted = (model(test_data) < 0.5).float()
        accuracy = (predicted == test_labels).float().mean()
        print(f'Accuracy for {key}: {accuracy.item():.2f}')

    # Konvertiere die Torch-Tensoren in numpy arrays
    predicted = predicted.numpy().flatten()
    test_labels = test_labels.numpy().flatten()
    
    # Plot der Vorhersagen gegenüber den tatsächlichen Labels
    plt.figure(figsize=(10, 8))
    plt.scatter(range(len(test_labels)), test_labels, color='blue',  marker='o', label='Actual Labels')
    plt.xlabel('Datenpunkt')
    plt.ylabel('Label')
    plt.title('Vorhersagen des Modells gegenüber den tatsächlichen Labels')
    plt.legend()
    plt.show()

    # Plot der Vorhersagen gegenüber den tatsächlichen Labels
    plt.figure(figsize=(10, 8))
    plt.scatter(range(len(test_labels)), test_labels, color='blue',  marker='o', label='Actual Labels')
    plt.scatter(range(len(predicted)), predicted, color='red', marker='x', label='Predicted Labels')
    plt.xlabel('Datenpunkt')
    plt.ylabel('Label')
    plt.title('Vorhersagen des Modells gegenüber den tatsächlichen Labels')
    plt.legend()
    plt.show()

















