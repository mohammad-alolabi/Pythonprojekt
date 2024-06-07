import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def load_mat_files_from_folder(folder_path):
    """
    Lädt alle .mat-Dateien aus dem angegebenen Ordner und gibt sie als Liste zurück.

    Args:
        folder_path (str): Pfad zum Ordner, der die .mat-Dateien enthält.

    Returns:
        list: Liste der geladenen .mat-Dateien.
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

    Args:
        mat_data (list): Liste der geladenen .mat-Daten.
        keys (list): Liste der Schlüssel, die extrahiert werden sollen.

    Returns:
        dict: Dictionary der extrahierten Daten.
    """
    extracted_data = {key: [next((value for k, value in data.items() if key in k), None) for data in mat_data] for key in keys}
    return extracted_data

def create_time_vectors(data, sample_rate=10000):
    """
    Erstellt Zeitvektoren für die gegebene Datenliste basierend auf der Abtastrate.

    Args:
        data (list): Liste der Daten.
        sample_rate (int, optional): Abtastrate. Standard ist 10000.

    Returns:
        list: Liste der Zeitvektoren.
    """
    return [np.arange(0, len(d)/sample_rate, 1/sample_rate) for d in data if d is not None]

def plot_multiple_measurements(time_data_list, measurement_data_list, labels, ylabel='Vibration (mm/s)', xlabel='Time (s)'):
    """
    Plottet mehrere Messungen in einem Diagramm.

    Args:
        time_data_list (list): Liste der Zeitdaten.
        measurement_data_list (list): Liste der Messdaten.
        labels (list): Liste der Labels für die Plots.
        ylabel (str, optional): Beschriftung der y-Achse. Standard ist 'Vibration (mm/s)'.
        xlabel (str, optional): Beschriftung der x-Achse. Standard ist 'Time (s)'.
    """
    plt.figure(figsize=(10, 6))
    for t_data, m_data, label in zip(time_data_list, measurement_data_list, labels):
        if t_data is not None and m_data is not None:
            plt.plot(t_data, m_data, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
