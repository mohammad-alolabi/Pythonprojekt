import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import scipy.io

def download_file(url, folder, retries=3):
    """
    Lädt eine Datei von der angegebenen URL herunter und speichert sie im angegebenen Ordner.

    Args:
        url (str): URL der Datei.
        folder (str): Pfad zum Ordner, in dem die Datei gespeichert werden soll.
        retries (int, optional): Anzahl der Versuche bei Fehlern. Standard ist 3.

    Returns:
        None
    """
    filename = os.path.join(folder, os.path.basename(url))
    if os.path.exists(filename):
        print(f"Datei bereits vorhanden: {filename}")
        # Bearbeitung der '99.mat'-Datei, falls erforderlich
        if '99.mat' in filename and 'Normale_Lager' in folder:
            process_99_mat_file(filename)
        return
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"Datei heruntergeladen: {filename}")

            # Spezifische Bearbeitung für '99.mat' in den normalen Lagerdaten
            if '99.mat' in filename and 'Normale_Lager' in folder:
                process_99_mat_file(filename)

            return
        except requests.exceptions.RequestException as e:
            print(f"Fehler beim Herunterladen der Datei: {url} (Versuch {attempt + 1} von {retries}) - {e}")
            if attempt == retries - 1:
                print(f"Maximale Anzahl an Versuchen erreicht. Datei {filename} konnte nicht heruntergeladen werden.")

def process_99_mat_file(filename):
    """
    Bearbeitet die '99.mat'-Datei, indem bestimmte Daten entfernt und andere geändert werden.

    Args:
        filename (str): Pfad zur '99.mat'-Datei.

    Returns:
        None
    """
    data = scipy.io.loadmat(filename)

    # Daten entfernen und ändern
    if 'X098_DE_time' in data:
        del data['X098_DE_time']
    if 'X098_FE_time' in data:
        del data['X098_FE_time']
    data['X099RPM'] = 1750

    # Daten speichern
    scipy.io.savemat(filename, data)
    print(f"'99.mat' wurde bearbeitet und gespeichert: {filename}")

def download_data(base_url, main_folder, folders):
    """
    Lädt die Daten von der angegebenen Basis-URL herunter und speichert sie in den spezifischen Ordnern.

    Args:
        base_url (str): Basis-URL der Webseite mit den Daten.
        main_folder (str): Pfad zum Hauptordner, in dem die Daten gespeichert werden sollen.
        folders (list): Liste der spezifischen Ordnernamen.

    Returns:
        None
    """
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Finde die Tabelle mit den Daten
    table = soup.find('table')

    if table:
        # Gehe durch alle Zeilen der Tabelle, um die maximale Anzahl von Spalten zu ermitteln
        max_columns = 0
        rows = table.find_all('tr')
        for row in rows:
            columns = row.find_all(['td', 'th'])
            if len(columns) > max_columns:
                max_columns = len(columns)
                
        # Iteriere über die Spalten
        for col in range(max_columns):
            # Bestimme den aktuellen Ordner basierend auf der Spaltennummer
            folder = folders[col % len(folders)]
            folder_path = os.path.join(main_folder, folder)
            
            # Gehe durch alle Zeilen der Tabelle und hole die Zelle der aktuellen Spalte
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) > col:
                    cell = cells[col]
                    # Finde alle Links in der Zelle
                    for link in cell.find_all('a'):
                        href = link.get('href')
                        if href and href.endswith('.mat'):
                            url = urljoin(base_url, href)
                            download_file(url, folder_path)

        print("Alle Dateien wurden heruntergeladen und entsprechend der Kategorien gespeichert.")
    else:
        print("Tabelle mit den Daten wurde nicht gefunden.")
