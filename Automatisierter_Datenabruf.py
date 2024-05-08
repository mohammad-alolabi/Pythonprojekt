import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import kurtosis
from scipy.stats import shapiro, anderson, levene
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

# # Normalisierung der Merkmale
# scaler = MinMaxScaler()
# for key in kurtosis_NL_DE_:
#     kurtosis_NL_DE_[key] = scaler.fit_transform(kurtosis_NL_DE_[key].reshape(-1, 1)).flatten()
# for key in kurtosis_FL_DE_:
#     kurtosis_FL_DE_[key] = scaler.fit_transform(kurtosis_FL_DE_[key].reshape(-1, 1)).flatten()

# # Beispiel für den Zugriff auf normierte Wölbungswerte
# print(kurtosis_NL_DE_["1"])  # Normierte Wölbungswerte für das erste normale Lager
# print(kurtosis_FL_DE_["1"])  # Normierte Wölbungswerte für das erste fehlerhafte Lager


# Plot der Wölbungswerte über die Zeitintervalle
plt.figure(figsize=(10, 6))
plt.plot(kurtosis_NL_DE_["1"], '*' , label=f'NL_DE_{1}')
#plt.plot(kurtosis_FL_DE_["1"], '*' , label=f'FL_DE_{1}')
#plt.plot(kurtosis_FL_DE_["2"], '*' , label=f'FL_DE_{2}')
plt.plot(kurtosis_FL_DE_["3"], '*' , label=f'FL_DE_{3}')
#plt.plot(kurtosis_FL_DE_["4"], '*' , label=f'FL_DE_{4}')
plt.xlabel('')
plt.ylabel('Wölbung')
plt.grid(True)
plt.legend()
plt.show()


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

