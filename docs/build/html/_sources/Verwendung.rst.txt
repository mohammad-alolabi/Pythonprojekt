Verwendung
==========

In diesem Kapitel wird die Installation und Nutzung der Skripte und Module beschrieben. Die Installation umfasst die notwendigen Bibliotheken, die installiert werden müssen, bevor der Code ausgeführt werden kann. Im nächsten Abschnitt wird erläutert, wie die Skripte und Module in einer bestimmten Reihenfolge ausgeführt werden sollten.

Installation
------------

Vor der Nutzung der Skripte müssen folgende Bibliotheken installiert werden:

.. code-block:: bash

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

Nutzung der Module
-------------------

Zunächst sollte das Skript `main_data_download.py` ausgeführt werden, um das Modul `data_download.py` zu verwenden und die Daten herunterzuladen, die dann in den Modulen weiterverarbeitet werden können. Das Skript `main_data_download.py` ist das Hauptskript für den Download der Daten. Es nutzt Funktionen aus `data_download.py`, um Verzeichnisse für normale und fehlerhafte Kugellager zu erstellen und die entsprechenden Daten herunterzuladen. Es definiert die Basis-URLs, die Struktur der Unterverzeichnisse und ruft dann die Funktion `download_data` auf, um den Downloadprozess zu starten.

Nach dem Herunterladen der Daten können die Module `data_processing.py`, `feature_engineering_and_model_1.py`, und `feature_engineering_and_model_2.py` verwendet werden, um die Daten zu verarbeiten, Merkmale zu extrahieren und die Modelle zu trainieren und zu testen. 

1. **Daten herunterladen**:
   
   Das Skript `data_download_script.py` lädt die Daten herunter und organisiert sie in Verzeichnissen für normale und fehlerhafte Lager. Die URLs und Zielverzeichnisse sind in `data_download.py` definiert.

   Beispielcode:

   .. code-block:: python
       
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


2. **Daten verarbeiten und visualisieren**:

   Dafür wird das Modul `data_processing.py` verwendet, um die heruntergeladenen .mat-Dateien zu laden, Daten zu extrahieren, ggf. Daten zu verarbeiten, Zeitvektoren zu erstellen und die Daten zu visualisieren. 

   Beispielcode:

   .. code-block:: python
       
     from data_processing import load_mat_files_from_folder, extract_data, create_time_vectors, plot_multiple_measurements

     # Pfade zu den Ordnern mit den .mat-Dateien
     folder_paths_FL = {
         'IR': r'.\Fehlerhafte_Lager\IR',
         'B': r'.\Fehlerhafte_Lager\B',
         'OR3': r'.\Fehlerhafte_Lager\OR@3',
         'OR6': r'.\Fehlerhafte_Lager\OR@6',
         'OR12': r'.\Fehlerhafte_Lager\OR@12'
     }
     folder_path_NL = r'.\Normale_Lager\NL'

     # Laden der Daten
     mat_data_FL = {key: load_mat_files_from_folder(path) for key, path in folder_paths_FL.items()}
     mat_data_NL = load_mat_files_from_folder(folder_path_NL)

     # Schlüssel zum Extrahieren
     keys = ['DE_time', 'FE_time']

     # Extrahieren der Daten
     extracted_NL_data = extract_data(mat_data_NL, keys)
     extracted_FL_data = {key: extract_data(data, keys) for key, data in mat_data_FL.items()}

     # Erstellen der Zeitvektoren
     t_NL_DE = create_time_vectors(extracted_NL_data['DE_time'])
     t_NL_FE = create_time_vectors(extracted_NL_data['FE_time'])

     t_FL_DE = {key: create_time_vectors(data['DE_time']) for key, data in extracted_FL_data.items()}
     t_FL_FE = {key: create_time_vectors(data['FE_time']) for key, data in extracted_FL_data.items()}

     # Beispiel: Plotten der ersten normalen Lager DE-Messung und der ersten fehlerhaften Lager DE-Messung für IR
     plot_multiple_measurements(
         [t_FL_DE['OR6'][0], t_FL_DE['OR3'][0], t_FL_DE['IR'][0], t_FL_DE['OR12'][0], t_FL_DE['B'][0], t_NL_DE[1]],
         [extracted_FL_data['OR6']['DE_time'][0], extracted_FL_data['OR3']['DE_time'][0],
          extracted_FL_data['IR']['DE_time'][0], extracted_FL_data['OR12']['DE_time'][0],
          extracted_FL_data['B']['DE_time'][0], extracted_NL_data['DE_time'][1]],
         ['Fehlerhafte Lager OR6 DE-Messung', 'Fehlerhafte Lager OR3 DE-Messung', 'Fehlerhafte Lager IR DE-Messung',
          'Fehlerhafte Lager OR12 DE-Messung', 'Fehlerhafte Lager B DE-Messung', 'Normale Lager DE-Messung']
     )

Das unten resultierende Diagramm zeigt die Vibrationen der Kugellager in verschiedenen Zuständen.
Es werden die Vibrationen von fehlerhaften Lagern in unterschiedlichen Zuständen (IR, B, OR\@3, OR\@6, OR\@12) und die Vibrationen eines normalen Lagers dargestellt.

.. figure:: /_static/Figure2.png
   :alt:
   :align: center

   Abbildung (2): Vibrationen der Kugellager in verschiedenen Zuständen.


3. **Merkmale berechnen und Modell trainieren**:

   Hier werden `feature_engineering_and_model_1.py` und `feature_engineering_and_model_2.py` verwendet, um Merkmale zu berechnen, die Daten für das Modell vorzubereiten und das neuronale Netzwerk zu trainieren.

   **Erstes Modell zur Klassifikation der normalen und fehlerhaften Lager (`feature_engineering_and_model_1.py`)**:
   
   - **Berechnung der Merkmale**: Die Funktion `compute_features` berechnet Merkmale wie die Wölbung (Kurtosis) und die Standardabweichung über festgelegte Intervalle.
         
        - Um zusätzliche Merkmale wie z.B. den Spitzenwert hinzuzufügen, soll die Funktion entsprechend erweitert werden:

           .. code-block:: python
       
               peak_values = []
               for i in range(num_intervals):
                   #... (In For-Schleife bleibt alles unveränderlich.)
                   #...
                   #...

                   # Berechnung des Spitzenwerts
                   interval_peak = np.max(interval_data)  
                   peak_values.append(interval_peak)

               features = [
                        #...
                        #...
                       np.array(peak_values).reshape(-1, 1)
                           ]
     
   - **Daten vorbereiten**: Die Funktion `prepare_data` kombiniert die berechneten Merkmale und erstellt die zugehörigen Labels. Diese Funktion erzeugt eine Feature-Matrix und einen Label-Vektor zur Klassifikation zwischen normalen und fehlerhaften Lagern.

   - **Modell**: Das Feedforward-Neuronale Netzwerk (`model_1`) wird verwendet, um die Klassifikation durchzuführen. Der Optimierer und die Verlustfunktion können angepasst werden, um die Trainingseffizienz zu verbessern.

         - **Verlustfunktion**: Die Binary Cross-Entropy Loss (BCELoss) wird verwendet. Diese Verlustfunktion ist ideal für binäre Klassifikationsprobleme, da sie den Unterschied zwischen den vorhergesagten Wahrscheinlichkeiten und den tatsächlichen Binärlabels misst. Der Verlustwert wird als Mittelwert der negativen logarithmischen Differenz zwischen den tatsächlichen Labels und den vorhergesagten Wahrscheinlichkeiten berechnet.

         .. math::

              L_{BCE}(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]

         Hierbei sind:

            - :math:`N`: Die Anzahl der Datenpunkte.
            - :math:`y_i`: Das tatsächliche Label für die :math:`i`-te Datenpunkte.
            - :math:`\hat{y}_i`: Die vorhergesagte Wahrscheinlichkeit für die :math:`i`-te Datenpunkte.

         Um die Verlustfunktion zu ändern, aktualisiere die Methoden `training_step`, `validation_step`, und `test_step`, um eine neue Verlustfunktion zu verwenden. Beispielsweise könnte `BCELoss` durch `CrossEntropyLoss` oder eine benutzerdefinierte Verlustfunktion ersetzt werden.
    
            Beispiel:
                
            .. code-block:: python
            
                 loss = nn.CrossEntropyLoss()(outputs, labels.long())
            
         Um den Optimierer zu ändern, aktualisiere die Methode `configure_optimizers`, um einen anderen Optimierer zu verwenden, wie z.B. `SGD`, `Adam` oder `RMSprop`. Du kannst zusätzliche Parameter für die Optimierer einstellen oder einen neuen Lernratenplan (`scheduler`) hinzufügen.

            Beispiel:

            .. code-block:: python
            
                 optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
         
    
   **Zweites Modell zur Klassifikation der Fehlerarten (`feature_engineering_and_model_2.py`)**:
   
   - **Berechnung der Merkmale**: Neben der Wölbung und Standardabweichung werden zusätzliche Merkmale wie Maximum, Varianz und Root Mean Square (RMS) berechnet. Die Funktion `compute_features` liefert ein erweitertes Set von Merkmalen, um eine differenzierte Analyse und Klassifikation von verschiedenen Fehlerarten zu ermöglichen.

        - Um zusätzliche Merkmale wie z.B. den Spitzenwert hinzuzufügen, soll die Funktion entsprechend wie vorher bei dem ersten Modell erweitert werden.

   - **Daten vorbereiten**: Die Funktion `prepare_data` erzeugt eine Feature-Matrix und einen Label-Vektor zur Klassifikation der Fehlerarten der Lager. Die Labels müssen spezifisch für jede Fehlerart zugewiesen werden. Dies ermöglicht die Klassifikation mehrerer Klassen, wie im folgenden Beispiel gezeigt:

     .. code-block:: python

         labels_FL = {
             'IR': 0,
             'B': 1,
             'OR3': 2,
             'OR6': 3,
             'OR12': 4
         }
         all_data, all_labels = prepare_data(features_FL_DE, labels_FL)

   - **Modell**: Das zweite Modell (`model_2`) verwendet ein Feedforward-Neuronales Netzwerk zur Klassifikation von mehreren Klassen. Es wird `CrossEntropyLoss` verwendet, um die Klassifikation von verschiedenen Fehlerarten zu ermöglichen. Diese Verlustfunktion misst den Unterschied zwischen den vorhergesagten Wahrscheinlichkeiten und den tatsächlichen Klassenlabels über mehrere Klassen hinweg.

        - **Verlustfunktion**: Die Cross-Entropy Loss (CELoss) wird verwendet. Diese Verlustfunktion ist ideal für Multi-Klassen-Klassifikationsprobleme, da sie die Wahrscheinlichkeitsdifferenzen zwischen den vorhergesagten und tatsächlichen Klassenlabels über mehrere Klassen misst.

        .. math::

             L_{CE}(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})

        Hierbei sind:

            - :math:`N`: Die Anzahl der Datenpunkte.
            - :math:`C`: Die Anzahl der Klassen.
            - :math:`y_{i,c}`: Das tatsächliche Label für die :math:`i`-te Datenpunkte und die Klasse :math:`c`.
            - :math:`\hat{y}_{i,c}`: Die vorhergesagte Wahrscheinlichkeit für die :math:`i`-te Datenpunkte und die Klasse :math:`c`.

         Um die Verlustfunktion anzupassen, ändere die Methoden `training_step`, `validation_step`, und `test_step`, um eine neue Verlustfunktion zu verwenden. 

            Beispiel:
                
            .. code-block:: python
            
                 loss = nn.CrossEntropyLoss()(outputs, labels)
            
         Um den Optimierer zu ändern, passe die Methode `configure_optimizers` an, um verschiedene Optimierer zu verwenden, wie z.B. `SGD`, `Adam` oder `RMSprop`. Weitere Parameter für die Optimierer können konfiguriert werden, oder ein neuer Lernratenplan (`scheduler`) kann hinzugefügt werden.

            Beispiel:

            .. code-block:: python
            
                 optim.RMSprop(self.parameters(), lr=self.learning_rate)

   **Konfigurierbare Parameter**

      Die Skripte `main_modell_1.py` und `main_modell_2.py` enthalten mehrere konfigurierbare Parameter, die die Leistung und das Verhalten des entsprechenden Modells beeinflussen (Genaue Untersuchung der Parametereinflüsse ist im nächsten Kapitel dokumentiert).:

      - `Umlaufzeit`: Zeit, die ein Wälzkörper auf der Lageraußenfläche benötigt, um eine vollständige Umdrehung zu vollziehen. Standard ist 0.033 Sekunde und wird berechnet durch :math:`T = \frac{U}{v}` wobei:
   
         - `U` ist der Umfang des Lagers.
         - `v` ist die Umfangsgeschwindigkeit, berechnet durch: :math:`v = \frac{π \cdot D \cdot rpm}{60}`
         - `D` ist der Außendurchmesser des Lagers in Metern.
         - `rpm` ist die Drehzahl in Umdrehungen pro Minute.

      - `train_size`, `val_size`: Die Anteile der Trainings- und Validierungsdaten.
      - `batch_size`: Die Größe der Batches für das Training.
      - `max_epochs`: Die maximale Anzahl der Trainings-Epochen.
      - `learning_rate`: Die Lernrate für den Optimierer.
      - `hidden_sizes`: Eine Liste mit den Größen und der Anzahl der versteckten Schichten im neuronalen Netzwerk.

 
Beispielcode:

Hier ist ein kleines Beispiel für die Ausführung des ersten Modells. Ein vollständiger Code für beide Modelle ist in `main_modell_1.py` bzw. `main_modell_2.py` vorhanden.

 .. code-block:: python

     from feature_engineering_and_model_1 import compute_features
     import matplotlib.pyplot as plt

     # Konfigurierbare Parameter
     Umlaufzeit = 0.033

     # Berechnung der Merkmale des normalen Lagers DE
     features_NL_DE = {f"NL_{i+1}": compute_features(data, Umlaufzeit) for i, data in enumerate(extracted_NL_data['DE_time']) if data is not None}
     # Berechnung der Merkmale des normalen Lagers FE
     features_NL_FE = {f"NL_{i+1}": compute_features(data, Umlaufzeit) for i, data in enumerate(extracted_NL_data['FE_time']) if data is not None}
     # Berechnung der Merkmale des fehlerhaften Lagers DE
     features_FL_DE = {f"{key}_{i+1}": compute_features(data, Umlaufzeit) for key, dataset in extracted_FL_data.items() for i, data in enumerate(dataset['DE_time']) if data is not None}
     # Berechnung der Merkmale des fehlerhaften Lagers FE
     features_FL_FE = {f"{key}_{i+1}": compute_features(data, Umlaufzeit) for key, dataset in extracted_FL_data.items() for i, data in enumerate(dataset['FE_time']) if data is not None}

     # Beispielplot der Merkmale
     plt.figure(figsize=(10, 6))
     # Plotting normal Lager DE Merkmale
     plt.plot(features_NL_DE["NL_2"][0][:400], "*", label='Normale Lager DE Wölbung') 

     # Plotting fehlerhafte Lager DE Merkmale 
     plt.plot(features_FL_DE["IR_1"][0], "^", label='Fehlerhafte Lager IR DE Wölbung') 
     plt.plot(features_FL_DE["B_1"][0], "^", label='Fehlerhafte Lager B DE Wölbung') 
     plt.plot(features_FL_DE["OR3_1"][0], "^", label='Fehlerhafte Lager OR3 DE Wölbung') 
     plt.plot(features_FL_DE["OR6_1"][0], "^", label='Fehlerhafte Lager OR6 DE Wölbung') 
     plt.plot(features_FL_DE["OR12_1"][0], "^", label='Fehlerhafte Lager OR12 DE Wölbung') 

     plt.xlabel('Datenpunkt')
     plt.ylabel('Merkmale')
     plt.legend()
     plt.grid(True)
     plt.show()

Das unten resultierende Diagramm zeigt das Merkmal (Wölbung) über den Datenpunkten. Das Merkmal wird von fehlerhaften Lagern in unterschiedlichen Zuständen und einem normalen Lager dargestellt.

.. figure:: /_static/Figure3.png
   :alt:
   :align: center

   Abbildung (3): Merkmal (Wölbung) über den Datenpunkten in verschiedenen Zuständen.

**Merkmale zusammenführen und Modell trainieren, validieren und testen**:
 
 .. code-block:: python
     
     from feature_engineering_and_model_1 import prepare_data, model_1, test_model
     import torch
     from torch.utils.data import DataLoader, TensorDataset, random_split
     import pytorch_lightning as pl 

     # Konfigurierbare Parameter
     train_size, val_size = 0.6, 0.2
     batch_size = 230
     max_epochs = 50
     learning_rate = 0.01
     hidden_sizes = [50, 25]  

     # Merkmale zusammenführen und Tensoren erstellen
     all_data_DE, all_labels_DE = prepare_data(features_NL_DE, features_FL_DE)
     data_tensor_DE = torch.tensor(all_data_DE[:, :, 0], dtype=torch.float32)
     labels_tensor_DE = torch.tensor(all_labels_DE, dtype=torch.float32)

     # Dataset erstellen  
     dataset_DE = TensorDataset(data_tensor_DE, labels_tensor_DE)

     # Aufteilen in Trainings-, Validierungs- und Testdatensätze
     train_size_DE = int(train_size * len(dataset_DE))
     val_size_DE = int(val_size * len(dataset_DE))
     test_size_DE = len(dataset_DE) - train_size_DE - val_size_DE
     train_dataset_DE, val_dataset_DE, test_dataset_DE = random_split(dataset_DE, [train_size_DE, val_size_DE, test_size_DE])

     # DataLoader erstellen
     batch_size_DE = test_size_DE
     train_loader_DE = DataLoader(train_dataset_DE, batch_size, shuffle=True)
     val_loader_DE = DataLoader(val_dataset_DE, batch_size)
     test_loader_DE = DataLoader(test_dataset_DE, batch_size_DE)

     # Hyperparameter definieren
     input_size = all_data_DE.shape[1] 
     output_size = 1
     
     # Modell instanziieren
     model = model_1(input_size, hidden_sizes, output_size, learning_rate)

     # Training des Modells
     trainer = pl.Trainer(max_epochs=max_epochs)
     trainer.fit(model, train_loader_DE, val_loader_DE)

     # Testen des Modells 
     test_model(model, test_loader_DE, "Test_DE") 

Mit diesem Beispiel wird am Ende das Modell trainiert, validiert und getestet. Mit dem `Test_DE`
wird eine Genauigkeit erreicht von:

.. math::

    \text{Genauigkeit} = \frac{\text{Anzahl der korrekten Vorhersagen}}{\text{Gesamtanzahl der Vorhersagen}} \approx 0,985

Dies wird formelhaft wie folgt berechnet:

.. math::

    \text{Genauigkeit} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\{ \hat{y}_i = y_i \}

wobei:

- :math:`N` die Gesamtzahl der Vorhersagen ist.
- :math:`\hat{y}_i` die vorhergesagte Klasse für das :math:`i`-te Beispiel ist.
- :math:`y_i` die tatsächliche Klasse für die :math:`i`-te Datenpunkte ist.
- :math:`\mathbf{1}\{ \hat{y}_i = y_i \}` eine Indikatorfunktion ist, die :math:`1` ist, wenn :math:`\hat{y}_i` gleich :math:`y_i` ist, und :math:`0` ansonsten.

Das unten resultierende Diagramm zeigt die ersten 100 aus 5457 getesteten Punkten als grafische Darstellung.

.. figure:: /_static/Figure4.png
   :alt:
   :align: center

   Abbildung (4): Visualisierung der Testung.
