import os
from Daten_herunterladen import download_data

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

# Daten für normale Lager herunterladen
download_data(base_url_NL, main_folder_NL, folders_NL)
# Daten für fehlerhafte Lager herunterladen
download_data(base_url_FL, main_folder_FL, folders_FL)
