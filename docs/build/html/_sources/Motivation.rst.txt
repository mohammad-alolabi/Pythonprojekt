   
Motivation
==========

Es ist wichtig, Maschinenkomponenten zu überwachen und Probleme frühzeitig zu erkennen, um 
sicherzustellen, dass industrielle Anlagen sicher und effizient arbeiten. Lager sind besonders 
wichtige Teile von Maschinen, weil sie die Hauptkomponente in einer rotierenden Maschine 
(Motor, Wellen, usw.) darstellen. Sie fallen jedoch oft aus und verursachen Betriebsstörungen. 
Mit der Verfügbarkeit von Sensordaten und dem Fortschritt im Bereich des Machine Learnings bietet 
der Einsatz von neuronalen Netzwerken die Möglichkeit, Probleme in Maschinenteilen frühzeitig zu 
erkennen.

In dieser Arbeit wird ein einfaches Neuronales Netz (Feedforward Netze) mit PyTorch bzw. 
Pytorch Lightning programmiert. Das Projekt konzentriert sich auf die Analyse von Kugellager-Testdaten, 
um zwischen normalen und fehlerhaften Lagern zu unterscheiden. Die Daten umfassen verschiedene Arten 
von Fehlern, darunter Defekte Innenring, Außenring und Wälzkörper. Die Testdaten können von 
der Webseite der [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) 
heruntergeladen werden. Die Herausforderung besteht darin, mit dem Neuronalen Netz die Klassen 
(Normal, Fehlerhaft) zu identifizieren. Eine weitere große Herausforderung liegt darin, die Art der 
fehlerhaften Lager zu klassifizieren. Dies erfordert nicht nur eine Vorverarbeitung der Daten, 
sondern auch die Anpassung der Hyperparameter des Netzwerks, um die verschiedenen Klassen zu erkennen.
