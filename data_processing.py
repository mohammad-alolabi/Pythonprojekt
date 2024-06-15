import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def load_mat_files_from_folder(folder_path):
    """
    Lädt alle .mat-Dateien aus einem angegebenen Verzeichnis und gibt deren Daten als Liste zurück.

    Diese Funktion durchsucht das angegebene Verzeichnis nach Dateien mit der Endung '.mat' und versucht,
    deren Inhalte mithilfe der scipy.io.loadmat Funktion zu laden. Die geladenen Daten werden in einer Liste
    gespeichert und zurückgegeben. Wenn ein Verzeichnis nicht existiert oder ein Fehler beim Laden einer Datei
    auftritt, wird eine entsprechende Fehlermeldung ausgegeben.

    Args:
        folder_path (str): Der Pfad des Verzeichnisses, aus dem die .mat-Dateien geladen werden sollen.

    Returns:
        list: Eine Liste von Dictionaries, die die Daten der .mat-Dateien enthalten. Jedes Dictionary repräsentiert
              den Inhalt einer .mat-Datei.

    Raises:
        FileNotFoundError: Wenn das angegebene Verzeichnis nicht existiert.
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
    Extrahiert spezifische Daten aus einer Liste von .mat-Dateien anhand gegebener Schlüssel.

    Diese Funktion durchsucht die geladenen .mat-Daten nach Werten, die mit den angegebenen Schlüsseln
    übereinstimmen. Die Ergebnisse werden in einem Dictionary gespeichert, wobei jeder Schlüssel eine Liste
    von extrahierten Werten enthält.

    Args:
        mat_data (list): Eine Liste von Dictionaries, die die Daten aus .mat-Dateien enthalten.
        keys (list): Eine Liste von Schlüsseln, deren zugehörige Daten extrahiert werden sollen.

    Returns:
        dict: Ein Dictionary, in dem jeder Schlüssel eine Liste der extrahierten Daten enthält. Falls ein Schlüssel
              in einer Datei nicht gefunden wird, wird `None` eingefügt.
    """
    extracted_data = {key: [next((value for k, value in data.items() if key in k), None) for data in mat_data] for key in keys}
    return extracted_data

def create_time_vectors(data, sample_rate=10000):
    """
    Erstellt Zeitvektoren für die gegebenen Messdaten auf Basis einer angegebenen Abtastrate.

    Diese Funktion generiert für jedes Datenarray in der Liste einen Zeitvektor, der von 0 bis zur Länge
    der Daten in Sekunden reicht. Die Standard-Abtastrate beträgt 10.000 Hz, kann aber angepasst werden.

    Args:
        data (list): Eine Liste von Arrays, die die Messdaten enthalten.
        sample_rate (int, optional): Die Abtastrate in Hz, die zur Erstellung der Zeitvektoren verwendet wird.
                                     Standard ist 10.000 Hz.

    Returns:
        list: Eine Liste von Arrays, die die Zeitvektoren für die Messdaten enthalten.
    """
    return [np.arange(0, len(d)/sample_rate, 1/sample_rate) for d in data if d is not None]

def plot_multiple_measurements(time_data_list, measurement_data_list, labels, ylabel='Vibration (mm/s)', xlabel='Time (s)'):
    """
    Plottet mehrere Messreihen auf einem Diagramm mit gemeinsamem Zeitvektor.

    Diese Funktion erstellt ein Diagramm, das mehrere Messreihen darstellt. Jede Messreihe wird auf Basis
    der entsprechenden Zeitvektoren geplottet. Die Labels werden verwendet, um die einzelnen Messreihen
    zu identifizieren.

    Args:
        time_data_list (list): Eine Liste von Arrays, die die Zeitvektoren für die Messdaten enthalten.
        measurement_data_list (list): Eine Liste von Arrays, die die Messdaten enthalten.
        labels (list): Eine Liste von Strings, die die Beschriftungen der einzelnen Messreihen darstellen.
        ylabel (str, optional): Die Beschriftung der y-Achse. Standard ist 'Vibration (mm/s)'.
        xlabel (str, optional): Die Beschriftung der x-Achse. Standard ist 'Time (s)'.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    for t_data, m_data, label in zip(time_data_list, measurement_data_list, labels):
        if t_data is not None and m_data is not None:
            plt.plot(t_data, m_data, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
