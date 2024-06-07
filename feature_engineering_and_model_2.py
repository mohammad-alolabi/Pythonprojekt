import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.metrics import precision_score, recall_score, f1_score

# Funktion zur Berechnung von Wölbung, Standardabweichung, Maximum, Varianz und Root Mean Square
def compute_features(data, interval_length):
    """
    Berechnet verschiedene Merkmale der Daten.

    Args:
        data (array): Eingabedaten.

    Returns:
        list: Liste der berechneten Merkmale.
    """
    kurtosis_values = []
    std_values = []
    max_values = []
    var_values = []
    rms_values = []
    
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
        interval_rms = np.sqrt(np.mean(data**2))
        rms_values.append(interval_rms)
        
        
        
        features = [
            np.array(kurtosis_values), 
            np.array(std_values).reshape(-1, 1), 
            np.array(max_values).reshape(-1, 1),
            np.array(var_values).reshape(-1, 1),
            np.array(rms_values).reshape(-1, 1)
                    ]
        
    return features

# Daten zusammenführen
def prepare_data(features_dict, labels_dict):
    """
    Führt die Daten und die zugehörigen Labels zusammen.

    Args:
        features_dict (dict): Dictionary der Merkmale.
        labels_dict (dict): Dictionary der Labels.

    Returns:
        tuple: Tuple aus allen Daten und den zugehörigen Labels.
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

class model_2(pl.LightningModule):
    """
    Definition des neuronalen Netzes.

    Args:
        input_size (int): Größe des Eingangsvektors.
        hidden_sizes (list): Liste der Größen der versteckten Schichten.
        output_size (int): Größe des Ausgangsvektors.
        learning_rate (float): Lernrate.
    """
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate):
        super(model_2, self).__init__()
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
        self.layers.append(nn.Softmax(dim=1))

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
        loss = nn.CrossEntropyLoss()(outputs, labels)
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
        loss = nn.CrossEntropyLoss()(outputs, labels)
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
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('test_loss', loss)
        
        # Additional metrics
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
        Konfiguriert die Optimizer.

        Returns:
            Optimizer: Optimizer für das Training.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)
