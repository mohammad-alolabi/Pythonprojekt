Einleitung
==========

In dieser Arbeit wird ein neuronales Netzwerk angewendet, um spezifische Probleme von Kugellagern 
anhand von Testdaten zu klassifizieren. Die verwendeten Beschleunigungsdaten wurden vom Bearing Data 
Center der Case Western Reserve University bereitgestellt. Diese Daten umfassen Vibrationsmessungen an 
verschiedenen Stellen, sowohl in der Nähe als auch entfernt von den Motorlagern.

Wie in der folgenden Abbildung (1) dargestellt, besteht der Prüfstand aus einem 2-PS-Motor (links), einem 
Drehmomentwandler/-encoder (Mitte) und einem Dynamometer (rechts). Vibrationsmessungen wurden mit drei 
Beschleunigungssensoren durchgeführt, die in der 12-Uhr-Position am Gehäuse der Antriebsseite (DE) 
und der Ventilatorseite (FE) angebracht waren. Für die Antriebsseite (DE) und die Lüfterseite (FE) 
wurden SKF-Rillenkugellager der Typen 6205-2RS JEM bzw. 6203-2RS JEM verwendet. Weitere Informationen 
zur Datenerfassung und den Arten von Fehlern sind detailliert auf der Webseite des Bearing Data Centers [1]_ dokumentiert.

.. figure:: /_static/Figure1.png
   :alt: Prüfstand zur Erfassung von Vibrationsdaten
   :align: center

   Abbildung (1): Lagerprüfstand. Quelle: Neupane [2]_.

In dieser Arbeit werden die Vibrationsmessungen beider Sensoren sowie die Daten normaler und 
fehlerhafter Lager an der Antriebsseite genutzt. Die Daten wurden mit einer Abtastrate von 
12.000 Proben/Sekunde erfasst, um ein Modell zur Klassifikation zu erstellen. Zunächst werden 
die Funktionen und Module bereitgestellt, um die Daten herunterzuladen und zu verarbeiten. Diese 
Module umfassen:

- `data_download.py`: Funktionen zum Erstellen von Verzeichnissen und Herunterladen der Daten.
- `data_processing.py`: Funktionen zum Laden und grafische Darstellung der Daten.
- `feature_engineering_and_model_1.py`: Funktionen zur Merkmalsextraktion und das erste Modell für die Klassifikation der normalen und fehlerhaften Lager.
- `feature_engineering_and_model_2.py`: Funktionen zur erweiterten Merkmalsextraktion und das zweite Modell zur Klassifikation der Fehlerarten.

Danach werden diese Funktionen in den folgenden Skripten aufgerufen und ausgeführt:

- `main_data_download.py`: Hauptskript zum Herunterladen der Daten und zur Strukturierung der Verzeichnisse.
- `main_modell_1.py`: Skript zur Anwendung des ersten Modells zur Unterscheidung zwischen normalen und fehlerhaften Lagern.
- `main_modell_2.py`: Skript zur Anwendung des zweiten Modells zur Klassifikation der verschiedenen Fehlerarten.

Die entsprechenden Skripte und Module werden detailliert in den Kapiteln zur Dokumentation des 
Python-Codes und zur Verwendung beschrieben, gefolgt von einer Diskussion der Ergebnisse.

.. rubric:: Referenzen

.. [1] Bearing Data Center, Case Western Reserve University. "Webseite." https://engineering.case.edu/bearingdatacenter. last visit 11.06.2024

.. [2] Neupane, Dhiraj and Seok, Jongwon "Bearing Fault Detection and Diagnosis Using Case Western Reserve University Dataset With Deep Learning Approaches: A Review." *Information*, vol. 2020, IEEE Access, pag. 93155-93178, doi: 10.1109/ACCESS.2020.2990528
