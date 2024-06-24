# Klassifikation von normalen und fehlerhaften Kugellagern mit neuronalen Netzwerken

## Projektübersicht

Dieses Projekt verwendet ein Feedforward-Neuronales Netzwerk zur Klassifikation von normalen und fehlerhaften Kugellagern. Das Ziel ist es, normale und defekte Kugellager anhand von Testdaten zu klassifizieren und verschiedene Fehlerarten zu identifizieren.
 Die verwendeten Testdaten stammen vom [Bearing Data Center der Case Western Reserve University](#(https://engineering.case.edu/bearingdatacenter/download-data-file).

## Inhaltsverzeichnis

- [Installation](#installation)
- [Projektstruktur](#projektstruktur)
- [Dokumentation](#dokumentation)
- [Beispiele](#Beispiele)
- [Ergebnisse](#ergebnisse)

## Installation

Um das Projekt zu nutzen, müssen die folgenden Bibliotheken installiert werden:

```bash
pip install requests beautifulsoup4
pip install numpy 
pip install scipy
pip install torchmetrics
pip install tensorboard
pip install sklearn
pip install torch
pip install pytorch-lightning 
pip install matplotlib 
pip install scikit-learn
```

## Projektstruktur

- data_download.py: Funktionen zum Erstellen von Verzeichnissen und Herunterladen der Daten.

- data_processing.py: Funktionen zum Laden und grafische Darstellung der Daten.

- feature_engineering_and_model_1.py: Funktionen zur Merkmalsextraktion und das erste Modell für die Klassifikation der normalen und fehlerhaften Lager.

- feature_engineering_and_model_2.py: Funktionen zur erweiterten Merkmalsextraktion und das zweite Modell zur Klassifikation der Fehlerarten.

- main_data_download.py: Hauptskript zum Herunterladen der Daten und zur Strukturierung der Verzeichnisse.

- main_modell_1.py: Skript zur Anwendung des ersten Modells zur Unterscheidung zwischen normalen und fehlerhaften Lagern.

- main_modell_2.py: Skript zur Anwendung des zweiten Modells zur Klassifikation der verschiedenen Fehlerarten.

- docs/: Enthält die Dokumentation des Projekts

## Dokumentation

Eine ausführliche Dokumentation ist im docs-Ordner vorhanden. Die index.html-Datei soll im Browser geöffnet werden, um die Dokumentation zu lesen.

## Beispiele

Beispiele zur Nutzung der Funktionen sind in den Python-Dateien und in der Dokumentation bereits gestellt.

## Ergebnisse

Die wichtigsten Ergebnisse des Projekts, die in der Dokumentation detailiert sind, umfassen:

- Hohe Genauigkeit bei der Klassifikation von normalen und fehlerhaften Lagern.

- Erfolgreiche Identifikation verschiedener Fehlerarten.

- Detaillierte Analyse der Hyperparameter-Einflüsse.
