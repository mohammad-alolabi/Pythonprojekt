import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_features(data, interval_length):
    """
    Berechnet verschiedene Merkmale der Daten in festgelegten Intervallen.

    Diese Funktion berechnet die Wölbung (Kurtosis), die Standardabweichung, das Maximum, 
    die Varianz und den Root Mean Square (RMS) der Daten in jedem Intervall.
    sie teilt die Eingangsdaten in Intervalle auf, berechnet für jedes Intervall die Merkmale 
    und gibt diese als Merkmalsvektoren zurück.
    Um ein zusätzliches Merkmal wie die Spitzenwert zu berechnen, kann die Funktion 
    entsprechend wie die Wölbung oder andere Mekrmale erweitert werden. (Ein genaues Beispiel wird später in nächstem Kapitel "Verwendung" erläutert)


    Args:
        data (array): Eingabedaten.
        interval_length (float): Länge des Intervalls in Sekunden, über das die Merkmale berechnet werden.

    Returns:
        list: Liste der berechneten Merkmale. Jedes Merkmal wird als NumPy-Array zurückgegeben.
    """
    kurtosis_values = []
    std_values = []
    max_values = []
    var_values = []
    rms_values = []

    # Anzahl der Intervalle bestimmen
    num_intervals = int(len(data) / (interval_length * 10000))
    
    for i in range(num_intervals):
        start_index = int(i * interval_length * 10000)
        end_index = int((i + 1) * interval_length * 10000)
        interval_data = data[start_index:end_index]

        # Berechnung der Wölbung
        interval_kurtosis = kurtosis(interval_data)
        kurtosis_values.append(interval_kurtosis)

        # Berechnung der Standardabweichung
        interval_std = np.std(interval_data)
        std_values.append(interval_std)

        # Berechnung des Maximums
        interval_max = np.max(interval_data)
        max_values.append(interval_max)

        # Berechnung der Varianz
        interval_var = np.var(interval_data)
        var_values.append(interval_var)

        # Berechnung der RMS (Root Mean Square)
        interval_rms = np.sqrt(np.mean(interval_data**2))
        rms_values.append(interval_rms)

    # Rückgabe der Merkmale
    features = [
        np.array(kurtosis_values), 
        np.array(std_values).reshape(-1, 1), 
        np.array(max_values).reshape(-1, 1),
        np.array(var_values).reshape(-1, 1),
        np.array(rms_values).reshape(-1, 1)
    ]
    
    return features

def prepare_data(features_dict, labels_dict):
    """
    Führt die Merkmale und die zugehörigen Labels zusammen.
    Args:
        features_dict (dict): Dictionary der Merkmale, wobei die Schlüssel die Lagerarten repräsentieren.
        labels_dict (dict): Dictionary der Labels, wobei die Schlüssel die Lagerarten repräsentieren.

    Returns:
        tuple: Tuple bestehend aus:
            - all_data (array): Array aller Merkmalsvektoren.
            - all_labels (array): Array aller zugehörigen Labels.
    """
    all_data = []
    all_labels = []
    
    for key in features_dict:
        base_key = key.split('_')[0]  # Extrahieren des Basis-Keys (z.B. 'IR', 'B', etc.)
        num_features = len(features_dict[key])  # Anzahl der Merkmale bestimmen
        for i in range(len(features_dict[key][0])):
            feature_vector = [features_dict[key][f][i] for f in range(num_features)]
            all_data.append(feature_vector)
            all_labels.append(labels_dict[base_key])  # Verwenden des Basis-Keys für das Label
    
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    return all_data, all_labels

class model_2(pl.LightningModule):
    """
    Ein Feedforward-Neuronales Netzwerk zur Klassifikation von Lagerfehlern.

    Diese Klasse definiert die Struktur und das Training eines neuronalen Netzwerks zur Klassifikation
    von Merkmalen, um verschiedene Arten von Lagerfehlern zu erkennen. Sie verwendet PyTorch Lightning
    für das Training und die Evaluierung. Für Mehrklassenklassifikation wird eine Softmax-Funktion in der 
    Ausgangsschicht verwendet.


    Diese Funktion kombiniert die berechneten Merkmale und erstellt die zugehörigen Labels 
    für die spätere Modellierung. Die Labels werden in dem Skript definiert.

    Args:
        input_size (int): Die Größe des Eingangsvektors. Typischerweise genau die Anzahl der Merkmale.
        hidden_sizes (list): Liste der Größen und Anzahl der versteckten Schichten.
                             Beispiel: ``hidden_sizes = [10, 5]`` bedeutet, dass es zwei versteckte Schichten gibt;
                             die erste Schicht hat 10 Neuronen und die zweite hat 5 Neuronen.
        output_size (int): Die Größe des Ausgangsvektors, typischerweise die Anzahl der Klassen für Mehrklassenklassifikation.
        learning_rate (float): Die Lernrate für den Optimierer (kann angepasst werden).

    """
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate):
        super(model_2, self).__init__()
        self.layers = nn.ModuleList()
        
        # Eingabeschicht
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        # Verborgene Schichten
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
        
        # Ausgabeschicht
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers.append(nn.Softmax(dim=1))

        self.learning_rate = learning_rate
        
    def forward(self, x):
        """
        Führt einen Vorwärtsdurchlauf durch das Netzwerk aus.

        Args:
            x (torch.Tensor): Eingabedaten.

        Returns:
            torch.Tensor: Ausgabe des Netzwerks.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Führt einen Schritt im Trainingsprozess durch.

        Args:
            batch (tuple): Batch von Daten und Labels.
            batch_idx (int): Index des Batches.

        Returns:
            torch.Tensor: Verlustwert des Trainings.

        """
        data, labels = batch
        outputs = self(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Führt einen Schritt im Validierungsprozess durch.

        Args:
            batch (tuple): Batch von Daten und Labels.
            batch_idx (int): Index des Batches.

        Returns:
            torch.Tensor: Verlustwert der Validierung.
        """
        data, labels = batch
        outputs = self(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Führt einen Schritt im Testprozess durch.

        Args:
            batch (tuple): Batch von Daten und Labels.
            batch_idx (int): Index des Batches.

        Returns:
            dict: Dictionary der Testergebnisse, einschließlich Verlust, Genauigkeit, Präzision, 
                  Rückruf und F1-Score.
        """
        data, labels = batch
        outputs = self(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('test_loss', loss)
        
        # Zusätzliche Metriken
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        precision = precision_score(labels.cpu(), preds.cpu(), average='weighted')
        recall = recall_score(labels.cpu(), preds.cpu(), average='weighted')
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        
        self.log('test_acc', acc)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
        
        return {'test_loss': loss, 'test_acc': acc, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1}

    def configure_optimizers(self):
        """
        Konfiguriert die Optimizer für das Training. 

        Returns:
            torch.optim.Optimizer: Der konfigurierte Optimierer.
 
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def test_model(model, test_loader, title):
    """
    Testet das Modell und visualisiert die Vorhersagen gegenüber den tatsächlichen Labels.

    Diese Funktion führt einen Testdurchlauf mit dem Modell durch, berechnet die Genauigkeit und 
    plottet die Vorhersagen des Modells gegenüber den tatsächlichen Labels. Weil die Anzahl der 
    Testdaten groß ist, sodass die Punkte der Vorhersagen des Modells gegenüber den tatsächlichen 
    Labels schlecht abzulesen sind, wird ein bestimmtes Intervall in der Funktion implementiert, 
    und zwar Intervall_A und Intervall_E. Ein Beispiel: (Intervall_A, Intervall_E = [0, len(test_labels)])
    für die gesamten Punkte zu plotten oder (Intervall_A, Intervall_E = [0, 50]) für die erste 
    50 Punkt zu plotten.

    Args:
        model (pl.LightningModule): Das zu testende Modell.
        test_loader (DataLoader): DataLoader für die Testdaten.
        title (str): Titel für den Plot.

    Returns:
        None
    """
    test_data, test_labels = next(iter(test_loader))
    with torch.no_grad():
        outputs = model(test_data)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == test_labels).float().mean()
        print(f'{title} Accuracy: {accuracy.item():.4f}')
        
        # Plot der Vorhersagen gegenüber den tatsächlichen Labels
        Intervall_A, Intervall_E = [0, 50]
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(test_labels))[Intervall_A:Intervall_E], test_labels.numpy()[Intervall_A:Intervall_E], color='blue', label='Actual Labels')
        plt.scatter(range(len(predictions))[Intervall_A:Intervall_E], predictions.numpy()[Intervall_A:Intervall_E], color='red', marker='x', label='Predicted Labels')
        plt.xlabel('Datenpunkt')
        plt.ylabel('Label')
        plt.title(f'Vorhersagen des Modells gegenüber den tatsächlichen Labels ({title})')
        plt.legend()
        plt.show()

