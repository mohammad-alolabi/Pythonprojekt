import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import kurtosis


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

# Plot für normale Lager-Daten erstellen
plt.figure(figsize=(10, 6))
for i, data in enumerate(NL_data_DE):
    plt.plot(t_NL_DE[i], data, label=f'NL_{i+1}')
plt.xlabel('Zeit')
plt.ylabel('Signalwert')
plt.title('Plot der normalen Lager-Daten (nach absteigendem RPM)')
plt.grid(True)
plt.legend()
plt.show()

# Plot für fehlerhafte Lager-Daten erstellen
plt.figure(figsize=(10, 6))
for i, data in enumerate(FL_data_DE):
    plt.plot(t_FL_DE[i], data, label=f'FL_{i+1}')
plt.xlabel('Zeit')
plt.ylabel('Signalwert')
plt.title('Plot der fehlerhaften Lager-Daten')
plt.grid(True)
plt.legend()
plt.show()

# plot erstellen
plt.figure(figsize=(10, 6)) 
plt.plot(t_FL_DE[3], FL_data_DE[3], label='FL_3') 
plt.plot(t_FL_DE[1], FL_data_DE[1], label='FL_1')   
plt.plot(t_FL_DE[4], FL_data_DE[4], label='FL_4')  
plt.plot(t_FL_DE[2], FL_data_DE[2], label='FL_2')  
plt.plot(t_NL_DE[0], NL_data_DE[0], label='NL_1')  
plt.xlabel('Zeit')  
plt.ylabel('Beschleunigung')  
plt.legend()  
plt.grid(True)  
plt.show(block=True)  
