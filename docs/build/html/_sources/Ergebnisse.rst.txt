Ergebnisse, Zusammenfassung und Schlussfolgerung
=====================================================

In diesem Kapitel werden die Ergebnisse der bereitgestellten Skripte `main_modell_1.py` und 
`main_modell_2.py` präsentiert und die Auswirkungen verschiedener Hyperparameteranpassungen diskutiert. 
Nachdem die MAT-Daten von der Webseite in die entsprechenden Ordner heruntergeladen wurden, wird das 
erste Modell zur Klissifikastion der normalen und fehlerarten Lager mit den folgenden konfigurierbaren Parametern verwendet:

 .. code-block:: python

     # Konfigurierbare Parameter
     Umlaufzeit = 0.033
     train_size, val_size = 0.6, 0.2
     batch_size = 230
     max_epochs = 50
     learning_rate = 0.01
     hidden_sizes = [50, 25]  


**Ergebnisse**
----------------------

Nach der Ausführung des Skriptes werden die berechneten Merkmale Wölbung und Standardabweichung grafisch 
dargestellt. Aus den beiden untenstehenden Diagrammen kann eine klare Aussage getroffen werden: Die 
Wahl der Merkmale spielt eine entscheidende Rolle für die Robustheit und Genauigkeit des Modells.

In der Abbildung (5) ist offensichtlich, dass es eine erhebliche Überlappung zwischen dem Merkmal der 
normalen Lager und den Merkmalen der in Ball fehlerhaften Lager gibt. Diese Überlappung ist besonders 
deutlich bei den Merkmalsklassen Wölbung und aber nicht bei der Standardabweichung in der Abbildung (6).

.. figure:: /_static/Figure5.png
   :alt:
   :align: center

   Abbildung (5):  Merkmal (Wölbung) über den Datenpunkten in verschiedenen Zuständen.

.. figure:: /_static/Figure6.png
   :alt:
   :align: center

   Abbildung (6):  Merkmal (Standardabweichung) über den Datenpunkten in verschiedenen Zuständen.

Es ist wichtig, die Auswirkungen einzelner Merkmale auf das Modell zu verstehen. Zu diesem Zweck 
wurde das Modell einmal nur mit der Wölbung und einmal nur mit der Standardabweichung trainiert und 
getestet. Die Genauigkeiten dieser beiden Merkmale sind in Tabelle 1 aufgeführt.

.. csv-table:: Genauigkeit des Modells bei Verwendung einzelner Merkmale
   :header: "Wölbung", "Standardabweichung", "Genauigkeit"
   :align: center

   "x", "", "83.0 %"
   "", "x", "98.4 %"

Die Ergebnisse zeigen, dass die Verwendung der Standardabweichung als Merkmal eine deutlich höhere 
Genauigkeit erzielt als die Wölbung. Dies deutet darauf hin, dass die Standardabweichung ein stabileres 
und aussagekräftigeres Merkmal zur Unterscheidung zwischen normalen und fehlerhaften Lagern darstellt.

Die in Abbildung (7) gezeigte Trainings- und Validierungsverlustfunktion verdeutlicht, dass das Verhalten der 
Verlustfunktion für das Merkmal (Standardabweichung) stärker als das Merkmal (Wölbung) exponentiell 
abnimmt. Die Ergebnisse legen nahe, dass die Wahl der Merkmale einen erheblichen Einfluss auf die 
Modellleistung hat.

.. figure:: /_static/Figure7.png
   :alt:
   :align: center

   Abbildung (7): Trainings- und Validierungsverlustfunktion bei Verwendung einzelner Merkmale.

Wie bereits erwähnt, wurde in den Daten auch ein anderer Beschleunigungssensor verwendet, der in der 
12-Uhr-Position am Gehäuse der Ventilatorseite (FE) angebracht ist. Dies wirft die Frage auf, ob das 
Modell auch in der Lage ist, anhand der Informationen dieses Sensors zwischen normalen und 
fehlerhaften Zuständen zu unterscheiden. Hierzu wurden die Schritte zur Datenauswertung, 
Merkmalsberechnung und -zusammenführung, die für den Beschleunigungssensor (DE) erläutert wurden, auch 
für den Sensor (FE) durchgeführt, jedoch ohne das Modell mit diesen Daten zu trainieren. Das Modell 
erreichte eine Genauigkeit von 90 % bei der Verwendung der Merkmale (Standardabweichung) vom 
Beschleunigungssensor (DE). Dies zeigt, dass die Erkennung des Zustands der Kugellager unabhängig vom 
Ort des Beschleunigungssensors möglich ist.

Eine weitere Untersuchung bestand darin, die konfigurierbaren Parameter zu ändern, insbesondere die 
Größe und Anzahl der verdeckten Schichten, z.B. `hidden_sizes = [5]` und `hidden_sizes = [50]`. Dabei 
zeigte sich, dass die Genauigkeit des Modells stabil bleibt, der Unterschied jedoch in der 
Trainingsverlustfunktion liegt. Die Trainingsverlustfunktion zeigt die Entwicklung des 
Trainingsverlusts im Verlauf der Trainingsiterationen und gibt Einblicke in das Verhalten des Modells. 
Das folgende Diagramm zeigt, wie sich verschiedene Netzwerkkonfigurationen auf die Lernkurve auswirken. 
Jede Kurve repräsentiert eine spezifische Kombination von versteckten Schichten und Neuronenanzahlen 
und zeigt, wie der Trainingsverlust im Verlauf der Iterationen abnimmt.

.. figure:: /_static/Figure8.png
   :alt: Trainingsverlustfunktion bei Verwendung verschiedener verdeckter Schichten.
   :align: center

   Abbildung (8): Trainingsverlustfunktion bei Verwendung verschiedener verdeckter Schichten.

Durch den Vergleich der Kurven lässt sich feststellen, dass komplexere Modelle (mehr Neuronen und 
Schichten) in der Regel schneller konvergieren, was auf eine bessere Modellleistung hinweist. Diese 
Analyse zeigt die Bedeutung der Auswahl geeigneter Netzwerkkonfigurationen basierend auf den 
Anforderungen und der Komplexität der Daten, um optimale Ergebnisse zu erzielen. Anhand der Abbildung 
lässt sich erkennen, welchen Einfluss die maximale Anzahl an Iterationen (`max_epochs`) hat: Wenn das 
Modell mit `hidden_sizes = [5]` nur mit 10 Iterationen trainiert würde, hätte es nicht geschafft, nahe 
Null zu konvergieren. Somit ist es wichtig, eine ausreichende Anzahl an Iterationen zu wählen, um eine 
gute Konvergenz der Trainingsverlustfunktion sicherzustellen.

Der Parameter `batch_size` gibt die Anzahl der Trainingsbeispiele an, die verwendet werden, um die 
Gewichte des Modells in einer einzigen Iteration zu aktualisieren. Kleinere Batchgrößen z.B 
`batch_size = 100` führen zu 
häufigeren Aktualisierungen der Gewichte, was zu einer besseren Generalisierung führen kann, jedoch 
mit noisier Lernkurven und längeren Trainingszeiten verbunden ist. Größere Batchgrößen z.B 
`batch_size = 1000` führen zu 
selteneren Aktualisierungen, was zu einem stabileren und schnelleren Training führt, aber die 
Generalisierungsfähigkeit des Modells reduzieren kann und mehr Speicher benötigt. Die Wahl der 
Batchgröße beeinflusst somit direkt die Trainingsstabilität und die Fähigkeit des Modells, zu 
generalisierten Lösungen zu gelangen.

Jetzt können wir das Skript `main_modell_2.py`, in dem die entsprechenden Funktionen und das Modell, 
das die Fehlerarten klassifiziert, ausgeführt werden können, in Betracht ziehen. Zunächst werden die 
konfigurierbaren Parameter verwendet, die am Anfang definiert wurden. Zusätzlich werden Merkmale wie 
Wölbung, Standardabweichung, Maximum, Varianz und Root Mean Square (RMS) berechnet.

Die Tabelle unten zeigt die Genauigkeit des Modells unter Verwendung verschiedener Kombinationen der 
konfigurierbaren Parameter. Diese Parameter umfassen die Anzahl der maximalen Epochen (`max_epochs`), 
die Architektur der verdeckten Schichten (`hidden_sizes`), die Batch-Größe (`batch_size`), 
die Lernrate (`learning_rate`), die Umlaufzeit (`Umlaufzeit`) und die erzielte Genauigkeit 
(`Genauigkeit`). Die Trainingsverlustkurve für jede Kombination wird ebenfalls grafisch in der Abbildung (9) dargestellt.

.. csv-table:: Genauigkeit des Modells bei Verwendung verschiedener Parameter
   :header: "Nr.", "max_epochs", "hidden_sizes", "batch_size", "learning_rate", "Umlaufzeit", "Genauigkeit"
   :align: center

   1, 50, "[50, 25]", 230, 0.01, 0.033, "68 %"
   2, 50, "[100, 50, 25]", 50, 0.01, 0.033, "53 %"
   3, 50, "[200, 100, 50, 25]", 50, 0.0001, 0.033, "69 %"
   4, 50, "[200, 100, 50, 25]", 50, 0.0001, "0.033 * 3", "80 %"
   5, 100, "[200, 200, 200, 200]", 40, 0.0001, "0.033 * 3", "85 %"

.. figure:: /_static/Figure9.png
   :alt: Trainingsverlustfunktion bei Verwendung verschiedener verdeckter Schichten.
   :align: center

   Abbildung (9): Trainingsverlustfunktion bei Verwendung verschiedener Parameter anhand der Tabelle.

Die Tabelle veranschaulicht, wie sich die Veränderung der Parameter auf die Modellgenauigkeit auswirkt. 
Hier sind einige wichtige Beobachtungen:

1. **Nr. 1:** Mit dieser vorherige Anpasung erreichte das Modell eine Genauigkeit von 68 %. Diese 
Konfiguration zeigt, dass mit diesen Parametern eine schlechte Leistung erzielt wurde.

2. **Nr. 2:** Durch Erhöhung der Anzahl der versteckten Schichten und Reduzierung der Batch-Größe 
sank die Genauigkeit auf 53 %. Aus der dazugehörigen Trainingsverlustkurve lässt sich erkennen, dass 
das Modell wahrscheinlich Oszillation in einem (lokalen oder globalen) Minimum hat. Dies deutet darauf 
hin, dass die Lernrate möglicherweise nicht optimal ist und angepasst werden sollte, um bessere Ergebnisse zu erzielen.

3. **Nr. 3:** Eine sehr niedrige Lernrate ermöglichten es dem Modell, eine Genauigkeit von 69 % zu 
erreichen. Dies zeigt, dass eine niedrige Lernrate das Problem in der Nr.2 gelost wurde, Jedoch bleibt
die Modellleistung immer noch niedrig wie in der Nr.1. 

4. **Nr. 4:** Bei gleicher Architektur wie Nr. 3, jedoch einer verlängerten Umlaufzeit, stieg die 
Genauigkeit auf 80 %. Aus der dazugehörigen Trainingsverlustkurve wird deutlich, dass die Anzahl der 
maximalen Epochen möglicherweise angepasst werden muss, um eine bessere Konvergenz zu erreichen. 
Die Verlängerung der Umlaufzeit bedeutet, dass das Modell mit längeren Intervallen arbeitet, was dazu 
beiträgt, die Datenmenge pro Merkmal zu reduzieren. Dies kann Überanpassung (Overfitting) vermeiden 
und die Robustheit des Modells erhöhen, da es weniger anfällig für zufällige Schwankungen in den Daten wird.

5. **Nr. 5:** Mit `max_epochs = 100`, `hidden_sizes = [200, 200, 200, 200]`, `batch_size = 40`, 
und einer sehr niedrigen Lernrate (`learning_rate = 0.0001`) erreichte das Modell eine Genauigkeit 
von 85 %. Diese Konfiguration, die eine höhere Komplexität und mehr Epochen umfasst, liefert die besten 
Ergebnisse. Dies verdeutlicht, dass komplexere Modelle, kombiniert mit einer ausreichend langen 
Trainingszeit und einer niedrigen Lernrate, zu besseren Leistungen führen können.

Die Ergebnisse zeigen, dass die Anzahl der Epochen (`max_epochs`) eine entscheidende Rolle für die 
Konvergenz des Modells spielt. Die Modelle mit `hidden_sizes = [50, 25]` und 
`hidden_sizes = [100, 50, 25]` haben mit 50 Epochen nicht optimal konvergiert, wie in den Kurven für 
Nr. 1 und Nr. 2 zu sehen ist. Die Komplexität des Modells sollte zur Anzahl der Epochen passen, um 
sicherzustellen, dass das Modell ausreichend trainiert wird. Die Wahl der Lernrate ist ebenfalls 
entscheidend, um zu vermeiden, dass das Modell in lokalen Minima stecken bleibt. Insgesamt 
unterstreichen diese Ergebnisse die Wichtigkeit der Hyperparameter-Tuning und der Analyse der 
Trainingskurven, um das Verhalten des Modells besser zu verstehen und die beste Leistung zu erzielen.
Es ist jedoch bemerkenswert, dass das Modell nicht in der Lage ist, eine gute Genauigkeit zu erreichen, 
wenn es mit den Daten des anderen Sensors (FE) getestet wird. Die Genauigkeit liegt in diesem Fall nur 
bei etwa 20 %. In der Abbildung (10) werden die ersten 50 aus 881 getesteten Punkten als grafisch dargestellt, 
wobei die Labels sind wie folgt:

 .. code-block:: python

     # Labels für fehlerhafte Lager
     labels_FL = {
        'IR': 0,
        'B': 1,
        'OR3': 2,
        'OR6': 3,
        'OR12': 4
     }

.. figure:: /_static/Figure10.png
   :alt: Trainingsverlustfunktion bei Verwendung verschiedener verdeckter Schichten.
   :align: center

   Abbildung (10): Vorhersagen des Modells gegenüber den tatsächlichen Labels.


**Zusammenfassung**
----------------------

Die Experimente zeigen, dass die Wahl der Merkmale und Hyperparameter entscheidend für die 
Modellleistung ist. Die Standardabweichung erwies sich als stabileres Merkmal zur Unterscheidung 
zwischen normalen und fehlerhaften Lagern im Vergleich zur Wölbung. Das Training mit einer verlängerten 
Umlaufzeit führte zu einer höheren Genauigkeit, da es die Datenmenge pro Merkmal reduzierte und damit 
Überanpassung minimierte. Die Anpassung der Lernrate und Batch-Größe beeinflusst die Konvergenz und 
Generalisierung des Modells erheblich. Es wurde festgestellt, dass das Modell Schwierigkeiten hat, die 
Genauigkeit bei der Klsifizierung der Fehlerarten aufrechtzuerhalten, wenn es mit den Daten des anderen 
Sensors getestet wird. Insgesamt betont diese Arbeit die Bedeutung der Feinabstimmung der 
Netzwerkkonfigurationen und Hyperparameter für optimale Ergebnisse.

**Schlussfolgerung**
----------------------

Diese Untersuchungen zeigen, dass eine sorgfältige Auswahl und Anpassung von Merkmalen und Hyperparametern 
wesentlich zur Verbesserung der Modellleistung beiträgt. Für zukünftige Forschungen könnte die Genauigkeit 
des Modells zur Klassifikation von Fehlerarten durch den Einsatz von Merkmalen aus dem Frequenzbereich 
erhöht werden. Alternativ könnten andere Verlustfunktionen getestet oder komplexere Netzarchitekturen, 
wie rekurrente neuronale Netze (RNNs) oder lang-kurzzeitgedächtnis-Netze (LSTMs), implementiert werden, 
um die Genauigkeit und Generalisierung weiter zu verbessern.






