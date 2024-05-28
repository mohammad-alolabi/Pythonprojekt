import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import scipy.io

# Basis-URL der Webseite
base_url_FL = "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data"
base_url_NL = "https://engineering.case.edu/bearingdatacenter/normal-baseline-data"

# Erstelle den Hauptordner, wenn er nicht existiert
main_folder_FL = "Fehlerhafte_Lager"
if not os.path.exists(main_folder_FL):
    os.makedirs(main_folder_FL)

main_folder_NL = "Normale_Lager"
if not os.path.exists(main_folder_NL):
    os.makedirs(main_folder_NL)

# Spezifische Ordnernamen in der gewünschten Reihenfolge
folders_FL = ["OR@6", "OR@3", "OR@12", "IR", "B"]
folders_NL = ["NL"]

# Erstelle die spezifischen Ordner, wenn sie nicht existieren
for folder in folders_FL:
    folder_path = os.path.join(main_folder_FL, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

for folder in folders_NL:
    folder_path = os.path.join(main_folder_NL, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Funktion zum Herunterladen einer Datei mit Wiederholungsmechanismus
def download_file(url, folder, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            filename = os.path.join(folder, os.path.basename(url))
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

# Funktion zur Bearbeitung der '99.mat'-Datei
def process_99_mat_file(filename):
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

# Funktion zum Herunterladen der Daten für eine bestimmte URL und spezifische Ordner
def download_data(base_url, main_folder, folders):
    # Lade die Webseite herunter
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Finde die Tabelle mit den Daten
    table = soup.find('table')

    # Überprüfe, ob die Tabelle gefunden wurde
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

# Daten für normale Lager herunterladen
download_data(base_url_NL, main_folder_NL, folders_NL)
# Daten für fehlerhafte Lager herunterladen
download_data(base_url_FL, main_folder_FL, folders_FL)
