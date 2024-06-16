import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import scipy.io

def create_directory(path) :
    """
    Hier erstellt diese Funktion ein Verzeichnis, falls es noch nicht existiert.

    Args:
        path (str): Der Pfad des zu erstellenden Verzeichnisses.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)

def download_file(url, folder, retries=5):
    """
    Diese Funktion lädt eine Datei von der angegebenen URL herunter und speichert sie im angegebenen Ordner. 
    Falls ein Fehler beim Herunterladen einer Datei kommt, wird es nochmal nach der Anzahl der 
    Wiederholungsversuche 'retries' versucht, die Datei herunterzuladen. Nach dem Erreichen maximale 
    Anzahl an Versuchen wird angezeigt, dass die Datei nicht heruntergeladen werden kann. Danach wird versucht, 
    die nächste Datei herunterzuladen. In diesem Fall ist empfohlen, nach Beendigung des Downloadprozesses den 
    Prozess nochmal zu wiederholen. Der Vorteil hier ist, dass die Funktion so effizient ist, dass am Anfang geprüft wird, ob die Datei bereits vorhanden
    ist oder nicht. Falls ja, wird sie nicht wieder heruntergeladen und es wird angezeigt, dass die Datei bereits vorhanden
    ist; falls nein, wird sie dann heruntergeladen.

    Args:
        url (str): Die URL der Datei, die heruntergeladen werden soll.
        folder (str): Der Ordner, in dem die Datei gespeichert werden soll.
        retries (int, optional): Die Anzahl der Versuche bei fehlgeschlagenem Download. Standard ist 5 (Kann aber angepasst werden).

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
    In der Datei '99.mat' wurde herausgefunden, dass es noch zusätzliche  Sensordaten gibt, die zur anderen
    Datei gehören. Um Konflikte zu vermeiden, wird die '99.mat'-Datei bearbeitet, indem bestimmte Daten entfernt 
    und andere geändert werden. Diese Funktion ist spezifisch für den Kontext der normalen Lagerdaten.

    Args:
        filename (str): Der Pfad zur '99.mat'-Datei.

    Returns:
        None
    """
    data = scipy.io.loadmat(filename)

    # Entfernen spezifischer Daten
    if 'X098_DE_time' in data:
        del data['X098_DE_time']
    if 'X098_FE_time' in data:
        del data['X098_FE_time']
    data['X099RPM'] = 1750

    # Speichern der bearbeiteten Daten
    scipy.io.savemat(filename, data)
    print(f"'99.mat' wurde bearbeitet und gespeichert: {filename}")

def download_data(base_url, main_folder, folders):
    """
    Diese Funktion lädt Daten von einer Webseite herunter, die unter der Basis-URL verfügbar sind, 
    und speichert sie in verschiedenen Unterordnern, die den spezifischen Kategorien entsprechen. 
    Sie durchsucht die Webseite nach Tabellen, extrahiert die Download-Links und lädt die Dateien 
    herunter, die dann in die entsprechenden Ordner gespeichert werden. Falls die Tabelle nicht 
    gefunden wird, gibt die Funktion eine entsprechende Fehlermeldung aus.

    Args:
        base_url (str): Die Basis-URL der Webseite, die die Daten bereitstellt.
        main_folder (str): Der Hauptordner, in dem die Daten gespeichert werden sollen.
        folders (list): Eine Liste von Ordnernamen für spezifische Kategorien.

    Returns:
        None
    """
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Suche die Tabelle mit den Daten
    table = soup.find('table')

    if table:
        # Bestimme die maximale Anzahl von Spalten
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
            create_directory(folder_path)
            
            # Iteriere über die Zeilen und hole die Links
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) > col:
                    cell = cells[col]
                    for link in cell.find_all('a'):
                        href = link.get('href')
                        if href and href.endswith('.mat'):
                            url = urljoin(base_url, href)
                            download_file(url, folder_path)

        print("Alle Dateien wurden heruntergeladen und in den spezifischen Ordnern gespeichert.")
    else:
        print("Tabelle mit den Daten wurde nicht gefunden.")

