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
    Funktion zur Berechnung von Wölbung und Standardabweichung.

    Args:
        data (array): Eingabedaten.
        interval_length (int): 

    Returns:
        list: Liste der berechneten Merkmale (Wölbung, Standardabweichung).
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
    Daten zusammenführen.

    Args:
        features_NL (dict): Merkmale der normalen Lager.
        features_FL (dict): Merkmale der fehlerhaften Lager.

    Returns:
        tuple: Tuple aus allen Daten und den zugehörigen Labels.
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
    Definition des neuronalen Netzes.

    Args:
        input_size (int): Größe des Eingangsvektors.
        hidden_sizes (list): Liste der Größen der versteckten Schichten.
        output_size (int): Größe des Ausgangsvektors.
        learning_rate (float): Lernrate.
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
        Vorwärtsdurchlauf durch das Netzwerk.

        Args:
            x (Tensor): Eingabedaten.

        Returns:
            Tensor: Ausgabe des Netzwerks.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Trainingsschritt.

        Args:
            batch (tuple): Batch von Daten und Labels.
            batch_idx (int): Index des Batches.

        Returns:
            Tensor: Verlustwert.
        """
        data, labels = batch
        outputs = self(data)
        loss = nn.BCELoss()(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validierungsschritt.

        Args:
            batch (tuple): Batch von Daten und Labels.
            batch_idx (int): Index des Batches.

        Returns:
            Tensor: Verlustwert.
        """
        data, labels = batch
        outputs = self(data)
        loss = nn.BCELoss()(outputs, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Tests Schritt.

        Args:
            batch (tuple): Batch von Daten und Labels.
            batch_idx (int): Index des Batches.

        Returns:
            dict: Dictionary der Metriken.
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
        Konfiguriert die Optimizer.

        Returns:
            Optimizer: Optimizer für das Training.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# Funktion zum Testen des Modells
def test_model(model, test_loader, title):
    """
    Testet das Modell und plottet die Vorhersagen gegenüber den tatsächlichen Labels.

    Args:
        model (pl.LightningModule): Das zu testende Modell.
        test_loader (DataLoader): DataLoader für die Testdaten.
        title (str): Titel für den Plot.

    Returns:
        None
    """
    test_data, test_labels = next(iter(test_loader))
    with torch.no_grad():
        predictions = (model(test_data) > 0.5).float()
        accuracy = (predictions == test_labels).float().mean()
        print(f'{title} Accuracy: {accuracy.item():.4f}')
        
        # Plot der Vorhersagen gegenüber den tatsächlichen Labels
        Intervall_A, Intervall_E = [0, 100]
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(test_labels))[Intervall_A:Intervall_E], test_labels.numpy()[Intervall_A:Intervall_E], color='blue', label='Actual Labels')
        plt.scatter(range(len(predictions))[Intervall_A:Intervall_E], predictions.numpy()[Intervall_A:Intervall_E], color='red', marker='x', label='Predicted Labels')
        plt.xlabel('Datenpunkt')
        plt.ylabel('Label')
        plt.title(f'Vorhersagen des Modells gegenüber den tatsächlichen Labels ({title})')
        plt.legend()
        plt.show()