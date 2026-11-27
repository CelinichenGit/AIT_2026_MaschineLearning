"""
#VORSAGE: Kopie/Erweiterung von cnn_controller.py für "Ansatz 2": das CNN lernt die Verzögerung selbst.

Problem: Zwischen Kamerabild und Ausführung des Befehls vergehen ca. 0,12 s. Ein Befehl, den das Netz
für das Bild zum Zeitpunkt t berechnet, wirkt erst bei t + Verzögerung.

Lösung (zwei Teile):
  1. Beschriftung verschieben: Zum Bild bei t lernt das Netz den Befehl, den der Lehrer bei t + n Schritten
     geben würde (n = Verzögerung in Simulationsschritten). Das Netz lernt also "vorauszuschauen".
  2. Noch unterwegs befindliche Befehle als Zusatz-Input: Das Netz bekommt die n Befehle, die nach dem Bild
     noch ausgeführt werden, bevor sein eigener Befehl wirkt. So weiß es, wo das Auto bis dahin sein wird.

Netz-Input "cmds" (Länge 2n): [speed_0 .. speed_{n-1}, steer_0 .. steer_{n-1}], normiert wie die Ausgaben.
Index 0 ist der Befehl, der als nächstes ausgeführt wird, Index n-1 der zuletzt gesendete.
"""
from collections import deque

import numpy as np

# #VORSAGE: unveränderte Teile aus dem Original wiederverwenden (gleiche Vorverarbeitung wie beim alten Modell)
from cnn_controller import (preprocess, set_skybox_seed, IMG_H, IMG_W, CROP_TOP_PX,  # noqa: F401
                            MAX_STEER_RAD, MAX_SPEED_MPS)

CMD_DT = 1.0 / 30.0       # #VORSAGE: Zeitabstand der Befehle im "cmds"-Input (= ein Simulationsschritt bei 30 fps)
DEFAULT_DELAY_STEPS = 4   # #VORSAGE: 4 * 33 ms ≈ 0,13 s ≈ delay_ms (120 ms) des gelben Fahrzeugs


def pending_command_features(sent_speed_mps, sent_steer_rad, rec_delay_steps: int, n: int) -> np.ndarray:
    """#VORSAGE: baut für eine aufgezeichnete Episode den "cmds"-Input für jedes Bild (für das Training).

    sent_*          : pro Frame der gesendete Befehl (vor der Verzögerung)
    rec_delay_steps : Verzögerung (in Schritten), mit der die Episode aufgenommen wurde
    n               : Verzögerung, für die das Modell trainiert wird

    Der Befehl, der bei Schritt t tatsächlich ausgeführt wurde, ist sent[t - rec_delay_steps].
    Für Bild t sind die "noch kommenden" Befehle die, die bei t .. t+n-1 ausgeführt werden.
    Das funktioniert für Aufnahmen mit UND ohne Verzögerung (bei 0 sind es die nächsten n Befehle).
    Vor dem Episodenstart war der Befehl (0, 0) - genau wie die Warteschlange in tools/simulation.py.
    """
    T = len(sent_speed_mps)
    sp = np.clip(np.asarray(sent_speed_mps, np.float32) / MAX_SPEED_MPS, 0.0, 1.0)
    st = np.clip(np.asarray(sent_steer_rad, np.float32) / MAX_STEER_RAD, -1.0, 1.0)
    idx = np.arange(T)[:, None] - rec_delay_steps + np.arange(n)[None, :]   # (T, n)
    valid = (idx >= 0) & (idx < T)
    idx = np.clip(idx, 0, T - 1)
    return np.concatenate([np.where(valid, sp[idx], 0.0),
                           np.where(valid, st[idx], 0.0)], axis=1).astype(np.float32)


def build_model(n_delay_steps: int = DEFAULT_DELAY_STEPS):
    """#VORSAGE: gleiche PilotNet-Architektur wie im Original, aber mit zweitem Input "cmds"."""
    from tensorflow import keras

    img_in = keras.Input((IMG_H, IMG_W, 3), dtype="uint8", name="image")
    cmd_in = keras.Input((2 * n_delay_steps,), dtype="float32", name="cmds")  # #VORSAGE: neuer Input

    x = keras.layers.Rescaling(1.0 / 255.0)(img_in)
    x = keras.layers.Conv2D(24, 5, strides=2, activation="relu")(x)
    x = keras.layers.Conv2D(36, 5, strides=2, activation="relu")(x)
    x = keras.layers.Conv2D(48, 5, strides=2, activation="relu")(x)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = keras.layers.Flatten()(x)

    c = keras.layers.Dense(32, activation="relu")(cmd_in)  # #VORSAGE: Befehle in kleinen Merkmalsvektor umwandeln
    x = keras.layers.Concatenate()([x, c])                   # #VORSAGE: Bildmerkmale + Befehlsmerkmale zusammenführen

    x = keras.layers.Dense(100, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(50, activation="relu")(x)
    x = keras.layers.Dense(10, activation="relu")(x)
    steer = keras.layers.Dense(1, activation="tanh", name="steer")(x)
    speed = keras.layers.Dense(1, activation="sigmoid", name="speed")(x)

    model = keras.Model({"image": img_in, "cmds": cmd_in}, {"steer": steer, "speed": speed},
                        name="pilotnet_delay")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss={"steer": "mse", "speed": "mse"},
                  loss_weights={"steer": 1.0, "speed": 0.5})
    return model


class CNNDriverVorhersage:
    """#VORSAGE: wie CNNDriver, merkt sich zusätzlich die gesendeten Befehle mit Zeitstempel.

    Ablauf pro Schritt:
        speed, steer = cnn.predict(bild, now)           # now = Sim-Zeit bzw. time.time()
        ... Befehl evtl. ändern/begrenzen ...
        cnn.register_sent(speed, steer, now)            # den TATSÄCHLICH gesendeten Befehl melden
    Über die Zeitstempel funktioniert das bei 30 Hz (Simulation) und bei 40 Hz (main.py) gleich.
    """

    def __init__(self, model_path: str, steering_smoothing: float = 0.5):
        import tensorflow as tf
        from tensorflow import keras

        self._tf = tf
        self.model = keras.models.load_model(model_path)
        # #VORSAGE: Verzögerung n aus der Form des "cmds"-Inputs ablesen (Länge 2n)
        cmd_inputs = [t for t in self.model.inputs if len(t.shape) == 2]
        if not cmd_inputs:
            raise ValueError(f"{model_path} hat keinen 'cmds'-Input - mit train_cnn_vorhersage.py trainiert?")
        self.n = int(cmd_inputs[0].shape[-1]) // 2
        self.delay_s = self.n * CMD_DT
        self.alpha = steering_smoothing
        self.last_steer = 0.0
        self.history = deque(maxlen=500)  # #VORSAGE: (zeit, speed_mps, steer_rad) der gesendeten Befehle
        self._forward = tf.function(lambda img, cmd: self.model({"image": img, "cmds": cmd}, training=False))

    def reset(self):
        self.last_steer = 0.0
        self.history.clear()

    def register_sent(self, speed_mps: float, steer_rad: float, now: float):
        """#VORSAGE: den Befehl melden, der wirklich ans Fahrzeug ging (ohne Lenk-Invertierung)."""
        self.history.append((now, float(speed_mps), float(steer_rad)))

    def pending_commands(self, now: float) -> np.ndarray:
        """#VORSAGE: die n Befehle, die nach "jetzt" noch ausgeführt werden, bevor der neue Befehl wirkt.

        Befehl k wird bei now + k*CMD_DT ausgeführt, wurde also bei now - (n-k)*CMD_DT gesendet.
        Genommen wird jeweils der letzte Befehl, der bis zu diesem Zeitpunkt gesendet wurde.
        """
        speeds, steers = np.zeros(self.n, np.float32), np.zeros(self.n, np.float32)
        for k in range(self.n):
            t_k = now - (self.n - k) * CMD_DT + 0.5 * CMD_DT  # halber Schritt Toleranz gegen Rundungsfehler
            for ts, s, a in reversed(self.history):
                if ts <= t_k:
                    speeds[k] = np.clip(s / MAX_SPEED_MPS, 0.0, 1.0)
                    steers[k] = np.clip(a / MAX_STEER_RAD, -1.0, 1.0)
                    break
        return np.concatenate([speeds, steers])

    def predict(self, bgr: np.ndarray, now: float):
        """Returns (target_speed_mps, steering_angle_rad)."""
        img = preprocess(bgr)[None]
        cmds = self.pending_commands(now)[None]  # #VORSAGE: zweiter Input
        out = self._forward(self._tf.constant(img), self._tf.constant(cmds))
        steer = float(out["steer"][0, 0]) * MAX_STEER_RAD
        speed = float(out["speed"][0, 0]) * MAX_SPEED_MPS

        steer = self.alpha * self.last_steer + (1.0 - self.alpha) * steer
        self.last_steer = steer
        return speed, steer
