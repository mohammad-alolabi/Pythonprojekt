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
   Berechnet Merkmale wie Wölbung (Kurtosis) und Standardabweichung aus den Daten in festgelegten Intervallen.

   Diese Funktion teilt die Eingangsdaten in Intervalle auf, berechnet für jedes Intervall die Wölbung
   und die Standardabweichung und gibt diese als Merkmalsvektoren zurück.
   Um ein zusätzliches Merkmal wie die Spitzenwert zu berechnen, kann die Funktion 
   entsprechend wie die Wölbung und die Standardabweichung erweitert werden. (Ein genaues Beispiel wird später in nächstem Kapitel "Verwendung" erläutert)

   Args:
       data (np.ndarray): Die Eingabedaten, typischerweise Schwingungsmessungen.
       interval_length (float): Die Länge der Intervalle in Sekunden, die zur Berechnung der Merkmale verwendet werden.

   Returns:
       list: Eine Liste, die Arrays enthält. Die Arrays enthalten die berechneten Merkmale über den festgelegten Intervallen.

   """
    kurtosis_values = []
    std_values = []
    
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

    features = [
        np.array(kurtosis_values),
        np.array(std_values).reshape(-1, 1)
        ]
    return features

def prepare_data(features_NL, features_FL):
    """
    Bereitet die Merkmale und Labels für normale und fehlerhafte Lager für das Training vor.

    Diese Funktion kombiniert die Merkmale normaler und fehlerhafter Lager zu einem vollständigen Datensatz
    und generiert die zugehörigen Labels. Die Labels sind 0 für normale und 1 für fehlerhafte Lager.

    Args:
        features_NL (dict): Ein Dictionary mit Merkmalen der normalen Lager.
        features_FL (dict): Ein Dictionary mit Merkmalen der fehlerhaften Lager.

    Returns:
        tuple: Ein Tupel bestehend aus:
            - `all_data` (np.ndarray): Der kombinierte Merkmalsdatensatz.
            - `all_labels` (np.ndarray): Die zugehörigen Labels.
    """
    all_data = []
    all_labels = []

    for key in features_NL:
        num_features = len(features_NL[key])  # Anzahl der Merkmale bestimmen
        for i in range(len(features_NL[key][0])):
            feature_vector = [features_NL[key][f][i] for f in range(num_features)]
            all_data.append(feature_vector)
        all_labels.extend([0] * len(features_NL[key][0]))

    for key in features_FL:
        num_features = len(features_FL[key])  # Anzahl der Merkmale bestimmen
        for i in range(len(features_FL[key][0])):
            feature_vector = [features_FL[key][f][i] for f in range(num_features)]
            all_data.append(feature_vector)
        all_labels.extend([1] * len(features_FL[key][0]))

    all_data = np.array(all_data)
    all_labels = np.array(all_labels).reshape(-1, 1)

    return all_data, all_labels

class model_1(pl.LightningModule):
    """
    Ein einfaches Feedforward-Neuronales Netzwerk zur Klassifikation von Lagerdaten.

    Diese Klasse definiert die Struktur und das Training eines neuronalen Netzwerks zur Klassifikation
    der Merkmale von Kugellagern in normale und fehlerhafte Lager. Es verwendet PyTorch Lightning
    für das Training und die Evaluierung. Da das Modell darauf trainiert wird, normale und fehlerhafte 
    Lager zu unterscheiden (also eine binäre Klassifikation), wird eine Sigmoid-Funktion in der Ausgangsschicht verwendet.

    Args:
        input_size (int): Die Größe des Eingangsvektors. (typischerweise genau die Anzahl der Merkmale)
        hidden_sizes (list): Liste der Größen und Anzahl der versteckten Schichten.
                            Beispiel: ``hidden_sizes = [10, 5]`` bedeutet, dass es zwei versteckte Schichten gibt;
                            die erste Schicht hat 10 Neuronen und die zweite hat 5 Neuronen.
        output_size (int): Die Größe des Ausgangsvektors (typischerweise 1 für binäre Klassifikation).
        learning_rate (float): Die Lernrate für den Optimierer (kann angepasst werden).

    """
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate):
        super(model_1, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers.append(nn.Sigmoid())
        
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Definiert den Vorwärtsdurchlauf durch das Netzwerk.

        Args:
            x (torch.Tensor): Der Eingabe-Tensor.

        Returns:
            torch.Tensor: Der Ausgabe-Tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Definiert einen einzelnen Trainingsschritt.

        Args:
            batch (tuple): Ein Tupel bestehend aus den Eingabedaten und den Labels.
            batch_idx (int): Der Index des Batches.

        Returns:
            torch.Tensor: Der Verlustwert für den aktuellen Batch.
        """
        data, labels = batch
        outputs = self(data)
        loss = nn.BCELoss()(outputs, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Definiert einen einzelnen Validierungsschritt.

        Args:
            batch (tuple): Ein Tupel bestehend aus den Eingabedaten und den Labels.
            batch_idx (int): Der Index des Batches.

        Returns:
            torch.Tensor: Der Verlustwert für den aktuellen Batch.
        """
        data, labels = batch
        outputs = self(data)
        loss = nn.BCELoss()(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Definiert einen einzelnen Testschritt und berechnet zusätzliche Metriken.

        Args:
            batch (tuple): Ein Tupel bestehend aus den Eingabedaten und den Labels.
            batch_idx (int): Der Index des Batches.

        Returns:
            dict: Ein Dictionary mit dem Verlust und den berechneten Metriken.
        """
        data, labels = batch
        outputs = self(data)
        loss = nn.BCELoss()(outputs, labels)
        self.log('test_loss', loss)
        
        # Additional metrics
        preds = (outputs > 0.5).float()
        acc = (preds == labels).float().mean()
        precision = precision_score(labels.cpu(), preds.cpu(), average='binary')
        recall = recall_score(labels.cpu(), preds.cpu(), average='binary')
        f1 = f1_score(labels.cpu(), preds.cpu(), average='binary')
        
        self.log('test_acc', acc)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
        
        return {'test_loss': loss, 'test_acc': acc, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1}

    def configure_optimizers(self):
        """
        Konfiguriert die Optimierer für das Training.

        Returns:
            torch.optim.Optimizer: Der konfigurierte Optimierer.
    
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)


# Funktion zum Testen des Modells
def test_model(model, test_loader, title):
    """
    Testet das Modell und plottet die Vorhersagen gegenüber den tatsächlichen Labels.

    Diese Funktion führt einen Testdurchlauf mit dem Modell durch, berechnet die Genauigkeit und 
    plottet die Vorhersagen des Modells gegenüber den tatsächlichen Labels. Weil die Anzahl der 
    Testdaten groß ist, sodass die Punkte der Vorhersagen des Modells gegenüber den tatsächlichen 
    Labels schlecht abzulesen sind, wird ein bestimmtes Intervall in der Funktion implementiert, 
    und zwar Intervall_A und Intervall_E. Ein Beispiel: (Intervall_A, Intervall_E = [0, len(test_labels)])
    für die gesamten Punkte zu plotten oder (Intervall_A, Intervall_E = [0, 50]) für die erste 
    50 Punkt zu plotten.

    Args:
        model (pl.LightningModule): Das zu testende Modell.
        test_loader (DataLoader): Der DataLoader für die Testdaten.
        title (str): Ein Titel für den Plot und die Ausgabe.

    Returns:
        None
    """
    test_data, test_labels = next(iter(test_loader))
    with torch.no_grad():
        predictions = (model(test_data) > 0.5).float()
        accuracy = (predictions == test_labels).float().mean()
        print(f'{title} Accuracy: {accuracy.item():.4f}')
        
        # Plot der Vorhersagen gegenüber den tatsächlichen Labels
        Intervall_A, Intervall_E = [0, len(test_labels)]
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(test_labels))[Intervall_A:Intervall_E], test_labels.numpy()[Intervall_A:Intervall_E], color='blue', label='Actual Labels')
        plt.scatter(range(len(predictions))[Intervall_A:Intervall_E], predictions.numpy()[Intervall_A:Intervall_E], color='red', marker='x', label='Predicted Labels')
        plt.xlabel('Datenpunkt')
        plt.ylabel('Label')
        plt.title(f'Vorhersagen des Modells gegenüber den tatsächlichen Labels ({title})')
        plt.legend()
        plt.show()
