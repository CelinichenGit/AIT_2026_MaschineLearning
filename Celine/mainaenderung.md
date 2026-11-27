Wie es jetzt läuft
Die Frontkamera am Auto (ESP32) schickt ihre Bilder per UDP. Der vorhandene Prozess receive_frontcam_image_thread legt sie im Shared Memory "front" ab. Die Bilder sind 320×240 groß, also genauso groß wie das Kamerabild in der Simulation.
Der Steuerprozess des Autos (controlVehicleThread) lädt jetzt einmal das Modell (CNNDriver("cnn_driver.keras")) und holt sich bei jedem Durchlauf (40 Hz) das neueste Bild.
cnn.predict(bild) berechnet daraus Geschwindigkeit und Lenkwinkel. Das ist genau dieselbe Vorverarbeitung wie beim Training: oben 110 Zeilen abschneiden, auf 200×66 verkleinern, RGB.
Danach läuft alles wie bisher: Lenkwinkel begrenzen, bei Bedarf umdrehen, in der ersten Sekunde langsam anfahren und mit sendControlsToHardware() ans Auto schicken.
Was ich geändert habe
Neue Einstellungen oben in der Datei (main.py:54-58):
USE_CNN_CONTROLLER = True schaltet das CNN ein. Mit False fährt wieder der alte Lidar-Controller.
CNN_MODEL_PATH gibt an, welche Modelldatei geladen wird.
CNN_STEERING_SMOOTHING stellt ein, wie stark die Lenkung geglättet wird.
controlVehicleThread: Der Prozess bekommt den Zeiger auf das Frontkamera-Bild und den Modellpfad übergeben. Er nutzt das CNN statt compute_next_command().
TensorFlow wird erst in diesem Prozess geladen, nicht im Hauptprozess.
Solange noch kein Kamerabild angekommen ist, bleibt das Auto stehen (Geschwindigkeit 0).
Frontkamera: Wenn das CNN eingeschaltet ist, wird die Frontkamera automatisch benutzt. Die Frage danach am Anfang entfällt.
Worauf du achten musst
# Die Deckenkamera wird trotzdem gebraucht. 
Sie erkennt das Auto und misst seine Geschwindigkeit, und der Motor-Regler braucht diese gemessene Geschwindigkeit.
Das Modell kennt bisher nur Simulationsbilder. Echte Bilder sehen anders aus (Licht, Farben, Unschärfe, Hintergrund). Rechne damit, dass es am Anfang schlecht fährt. **Starte zuerst mit niedriger Geschwindigkeit: MAX_SPEED_MPS** in cnn_controller.py gilt beim Fahren genauso.
Eine Lösung dafür wären Trainingsdaten von der echten Strecke. Du lässt in main.py den Lidar-Controller fahren und speicherst dabei jedes Frontkamera-Bild zusammen mit den Befehlen, die er sendet. Das ist dasselbe Prinzip wie --mode collect. Mit diesen Daten trainierst du das Modell dann weiter.
**Nur ein Auto kann das CNN nutzen**. Es gibt nur einen Frontkamera-Stream, und das Modell ist auf die Lenkwerte vom gelben Auto eingestellt (MAX_STEER_RAD = 27°).
**Kamera-Ausschnitt prüfen:** Schau nach, ob der Ausschnitt CROP_TOP_PX = 110 auch bei der echten Kamera passt, also ob danach wirklich nur noch die Strecke zu sehen ist.