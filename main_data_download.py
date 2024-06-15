from data_download import create_directory, download_data

# Basis-URL der Webseite
base_url_FL = "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data"
base_url_NL = "https://engineering.case.edu/bearingdatacenter/normal-baseline-data"

# Erstelle den Hauptordner, wenn er nicht existiert
main_folder_FL = "Fehlerhafte_Lager"
create_directory(main_folder_FL)

main_folder_NL = "Normale_Lager"
create_directory(main_folder_NL)

# Spezifische Ordnernamen in der gewünschten Reihenfolge
folders_FL = ["OR@6", "OR@3", "OR@12", "IR", "B"]
folders_NL = ["NL"]

# Daten für normale Lager herunterladen
download_data(base_url_NL, main_folder_NL, folders_NL)
# Daten für fehlerhafte Lager herunterladen
download_data(base_url_FL, main_folder_FL, folders_FL)
