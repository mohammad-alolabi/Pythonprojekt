import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis


# Pfad zum Ordner mit den .mat-Dateien
folder_path_FL = r'C:\Users\ha368\OneDrive\Desktop\Python\Datenset\Fehlerhafte_Lager'
folder_path_NL = r'C:\Users\ha368\OneDrive\Desktop\Python\Datenset\Normale_Lager'

# Liste, um die geladenen Fehlerhafte Lager-Daten zu speichern
mat_data_list_FL = []

# Alle Dateien im Ordner durchgehen
for file_name in os.listdir(folder_path_FL):
    if file_name.endswith('.mat'):  # Sicherstellen, dass die Datei eine .mat-Datei ist
        # Den vollständigen Pfad zur .mat-Datei erstellen
        mat_file_path_FL = os.path.join(folder_path_FL, file_name)
        
        # Datei laden und die Daten zur Liste hinzufügen
        mat_data_list_FL.append(scipy.io.loadmat(mat_file_path_FL))
 
# Liste, um die geladenen normale Lager-Daten zu speichern
mat_data_list_NL = []
        
for file_name in os.listdir(folder_path_NL):
    if file_name.endswith('.mat'):  # Sicherstellen, dass die Datei eine .mat-Datei ist
        # Den vollständigen Pfad zur .mat-Datei erstellen
        mat_file_path = os.path.join(folder_path_NL, file_name)
        
        # Datei laden und die Daten zur Liste hinzufügen
        mat_data_list_NL.append(scipy.io.loadmat(mat_file_path))        
        
        
# # Zeitvektoren initialisieren
# t_NL = []
# t_FL = []

# # Für die normale Lager-Daten
# NL_data = [mat_data_NL['X097_DE_time'], mat_data_NL['X098_DE_time'], mat_data_NL['X099_DE_time'], mat_data_NL['X100_DE_time']]

# for data in NL_data:
#     # Zeitvektor erstellen und hinzufügen
#     t_NL.append(np.arange(0, len(data)/10000, 0.0001))

# # Für die Fehlerhafte Lager-Daten
# FL_data = [mat_data_FL['X105_DE_time'], mat_data_FL['X122_DE_time'], mat_data_FL['X135_DE_time'], mat_data_FL['X148_DE_time'], mat_data_FL['X161_FE_time']]

# for data in FL_data:
#     # Zeitvektor erstellen und hinzufügen
#     t_FL.append(np.arange(0, len(data)/10000, 0.0001))
  

# Hier kannst du mit den geladenen normale Lager-Daten in mat_data_list arbeiten
NL_1 = mat_data_list_NL[1]['X097_DE_time'] 
NL_2 = mat_data_list_NL[2]['X098_DE_time'] 
NL_3 = mat_data_list_NL[3]['X099_DE_time'] 
NL_4 = mat_data_list_NL[0]['X100_DE_time'] 

# Zeitvektor erstellen
t_NL_1 = np.arange(0, len(NL_1)/10000, 0.0001)
t_NL_2 = np.arange(0, len(NL_2)/10000, 0.0001)
t_NL_3 = np.arange(0, len(NL_3)/10000, 0.0001)
t_NL_4 = np.arange(0, len(NL_4)/10000, 0.0001)

# Hier kannst du mit den geladenen Fehlerhafte Lager-Daten in mat_data_list arbeiten
FL_1 = mat_data_list_FL[0]['X105_DE_time'] 
FL_2 = mat_data_list_FL[1]['X122_DE_time'] 
FL_3 = mat_data_list_FL[2]['X135_DE_time'] 
FL_4 = mat_data_list_FL[3]['X148_DE_time'] 
FL_4 = mat_data_list_FL[4]['X161_FE_time']

# Zeitvektor erstellen
t_FL_1 = np.arange(0, len(FL_1)/10000, 0.0001)
t_FL_2 = np.arange(0, len(FL_2)/10000, 0.0001)
t_FL_3 = np.arange(0, len(FL_3)/10000, 0.0001)
t_FL_4 = np.arange(0, len(FL_4)/10000, 0.0001)

# Plot erstellen
plt.figure(figsize=(10, 6)) 

plt.plot(t_FL_3, FL_3, label='FL_3') 
plt.plot(t_FL_1, FL_1, label='FL_1')  
plt.plot(t_FL_2, FL_2, label='FL_2')   
plt.plot(t_FL_4, FL_4, label='FL_4')  
plt.plot(t_NL_1, NL_1, label='NL_1')  

plt.xlabel('Zeit')  
plt.ylabel('Beschleunigung')  
plt.legend()  
plt.grid(True)  
plt.show(block=True)  


# Spektrum berechnen
NL_1_fft = np.fft.fft(np.abs(NL_1)) / len(NL_1)
NL_2_fft = np.fft.fft(np.abs(NL_2)) / len(NL_2)
NL_3_fft = np.fft.fft(np.abs(NL_3)) / len(NL_3)
NL_4_fft = np.fft.fft(np.abs(NL_4)) / len(NL_4) 

FL_1_fft = np.fft.fft(np.abs(FL_1)) / len(FL_1) 
FL_2_fft = np.fft.fft(np.abs(FL_2)) / len(FL_2) 
FL_3_fft = np.fft.fft(np.abs(FL_3)) / len(FL_3) 
FL_4_fft = np.fft.fft(np.abs(FL_4)) / len(FL_4) 

# Frequenzachse erstellen
f_NL_1_fft = np.fft.fftfreq(len(NL_1))
f_NL_2_fft = np.fft.fftfreq(len(NL_2))
f_NL_3_fft = np.fft.fftfreq(len(NL_3))
f_NL_4_fft = np.fft.fftfreq(len(NL_4))

f_FL_1_fft = np.fft.fftfreq(len(FL_1))
f_FL_2_fft = np.fft.fftfreq(len(FL_2))
f_FL_3_fft = np.fft.fftfreq(len(FL_3))
f_FL_4_fft = np.fft.fftfreq(len(FL_4))

# # Plot des Spektrums
# plt.figure(figsize=(10, 6))
# plt.plot(f_FL_3_fft, FL_3_fft, label='FL_3_fft')
# plt.plot(f_FL_1_fft, FL_1_fft, label='FL_1_fft')
# plt.plot(f_FL_4_fft, FL_4_fft, label='FL_4_fft')
# plt.plot(f_FL_2_fft, FL_2_fft, label='FL_2_fft')
# plt.plot(f_NL_1_fft, NL_1_fft, label='NL_1_fft')
# plt.xlabel('Frequenz (Hz)')
# plt.ylabel('Amplitude')
# plt.title('Normalized Amplitude Spectrum')
# plt.grid(True)
# plt.legend()  
# plt.grid(True)  
# plt.show(block=True)  

# Berechne die Kurtosis des Rohsignals
sig_NL_kurtosis = [kurtosis(NL_1), kurtosis(NL_2), kurtosis(NL_3), kurtosis(NL_4)]
sig_FL_kurtosis = [kurtosis(FL_1), kurtosis(FL_2), kurtosis(FL_3), kurtosis(FL_4)]

# Berechne die Kurtosis des normierten Amplitudenspektrums
spe_NL_kurtosis = [kurtosis(NL_1_fft), kurtosis(NL_2_fft), kurtosis(NL_3_fft), kurtosis(NL_4_fft)]
spe_FL_kurtosis = [kurtosis(FL_1_fft), kurtosis(FL_2_fft), kurtosis(FL_3_fft), kurtosis(FL_4_fft)]

Anzahl_Mek = np.arange(1, len(sig_NL_kurtosis)+1)

# # Normaeren der Merkmale
# liste = []
# xmin = min(spe_FL_kurtosis)
# xmax = max(spe_FL_kurtosis)
# for i, x in enumerate(spe_FL_kurtosis):
#     spe_FL_kurtosis[i] = (x - xmin) / (xmax - xmin)

print("Kurtosis vom normalen Lager des Rohsignals:", sig_NL_kurtosis)
print("Kurtosis vom Fehlerhaften Lager des Rohsignals:", sig_FL_kurtosis)
print("Kurtosis vom normalen Lager des Spectrumsignals:", spe_NL_kurtosis)
print("Kurtosis vom Fehlerhaften Lager des Spectrumsignals:", spe_FL_kurtosis)
print("Normalized Daten:", spe_FL_kurtosis)

# Plot der Liste
plt.figure(figsize=(10, 6))
plt.plot(Anzahl_Mek,sig_NL_kurtosis, '-*', label='sig_NL_kurtosis')
plt.plot(Anzahl_Mek,sig_FL_kurtosis, '-*', label='sig_FL_kurtosis')
plt.xlabel('Anzahl der Merkmale')
plt.ylabel('Beschleunigung')
plt.grid(True)
plt.legend()  
plt.grid(True)  
plt.show(block=True)  

plt.figure(figsize=(10, 6))
plt.plot(Anzahl_Mek,spe_NL_kurtosis, '-*', label='spe_NL_kurtosis')
plt.plot(Anzahl_Mek,spe_FL_kurtosis, '-*', label='spe_FL_kurtosis')
plt.xlabel('Anzahl der Merkmale')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()  
plt.grid(True)  
plt.show(block=True)  

# Berechne den Mittelwert des Rohsignals
sig_NL_mean = [np.mean(NL_1), np.mean(NL_2), np.mean(NL_3), np.mean(NL_4)]
sig_FL_mean = [np.mean(FL_1), np.mean(FL_2), np.mean(FL_3), np.mean(FL_4)]

# Berechne die Mittelwert des normierten Amplitudenspektrums
spe_NL_mean = [np.mean(NL_1_fft), np.mean(NL_2_fft), np.mean(NL_3_fft), np.mean(NL_4_fft)]
spe_FL_mean = [np.mean(FL_1_fft), np.mean(FL_2_fft), np.mean(FL_3_fft), np.mean(FL_4_fft)]

print("Mittelwert vom normalen Lager des Rohsignals:", sig_NL_mean)
print("Mittelwert vom Fehlerhaften Lager des Rohsignals:", sig_FL_mean)
print("Mittelwert vom normalen Lager des Spectrumsignals:", spe_NL_mean)
print("Mittelwert vom Fehlerhaften Lager des Spectrumsignals:", spe_FL_mean)

# Plot der Liste
plt.figure(figsize=(10, 6))
plt.plot(Anzahl_Mek,sig_NL_mean, '-*', label='sig_NL_mean')
plt.plot(Anzahl_Mek,sig_FL_mean, '-*', label='sig_FL_mean')
plt.xlabel('Anzahl der Merkmale')
plt.ylabel('Beschleunigung')
plt.grid(True)
plt.legend()  
plt.grid(True)  
plt.show(block=True)  

plt.figure(figsize=(10, 6))
plt.plot(Anzahl_Mek,spe_NL_mean, '-*', label='spe_NL_mean')
plt.plot(Anzahl_Mek,spe_FL_mean, '-*', label='spe_FL_mean')
plt.xlabel('Anzahl der Merkmale')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()  
plt.grid(True)  
plt.show(block=True)  


