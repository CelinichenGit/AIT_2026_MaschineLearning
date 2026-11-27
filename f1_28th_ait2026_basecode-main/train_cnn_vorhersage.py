"""
#VORSAGE: Kopie von train_cnn.py für "Ansatz 2" (das CNN lernt die Verzögerung).
  - Beschriftung verschoben: Bild t bekommt den Lehrer-Befehl von Schritt t + n.
  - Zusatz-Input "cmds": die n Befehle, die nach Bild t noch ausgeführt werden (siehe cnn_controller_vorhersage.py).

  python train_cnn_vorhersage.py                                        # neues Modell, n = 4 Schritte (~0,13 s)
  python train_cnn_vorhersage.py --delay-steps 4 --data-dir cnn_data_vorhersage cnn_data   # alte Daten mitnehmen
  python train_cnn_vorhersage.py --init cnn_driver_vorhersage.keras     # weitertrainieren (z.B. nach DAgger)
"""
import argparse
import glob
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from cnn_controller_vorhersage import (build_model, pending_command_features,  # #VORSAGE
                                      DEFAULT_DELAY_STEPS, MAX_SPEED_MPS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", nargs="+", default=["cnn_data_vorhersage"],  # #VORSAGE: mehrere Ordner möglich
                   help="ein oder mehrere Ordner mit .npz-Episoden")
    p.add_argument("--out", default="cnn_driver_vorhersage.keras")  # #VORSAGE: eigenes Modell
    p.add_argument("--delay-steps", type=int, default=DEFAULT_DELAY_STEPS,  # #VORSAGE
                   help="Verzögerung n in Simulationsschritten (1/30 s), für die das Modell lernt")
    p.add_argument("--init", default=None, help="existing model to continue training from")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.15, help="fraction of EPISODES used for validation")
    return p.parse_args()


def load_files(files, n):
    """#VORSAGE: lädt die Episoden und baut daraus (Bild t, cmds t) -> Lehrer-Befehl bei t + n."""
    xs, cmds, steers, speeds = [], [], [], []
    for f in files:
        d = np.load(f)
        T = len(d["x"])
        # #VORSAGE: echte Aufnahmen (main_collect_vorhersage.py) werden in Stücken gespeichert. Die ersten skip_first
        #           Frames eines Stücks sind nur Vorgeschichte für den cmds-Input und werden nicht als Beispiel benutzt.
        skip = int(d["skip_first"]) if "skip_first" in d else 0
        if T - n <= skip:  # #VORSAGE: Episode zu kurz, um die Beschriftung um n zu verschieben
            continue

        # #VORSAGE: gesendete Befehle der Episode. Alte Aufnahmen (simulation_cnn.py) haben kein sent_speed_mps;
        #           dort fuhr der Lehrer selbst, die gesendete Geschwindigkeit = seine Beschriftung.
        sent_speed = d["sent_speed_mps"] if "sent_speed_mps" in d else d["speed"] * MAX_SPEED_MPS
        sent_steer = d["executed_steer_rad"]
        rec_delay = int(d["delay_steps"]) if "delay_steps" in d else 0  # #VORSAGE: alte Aufnahmen: ohne Verzögerung

        c = pending_command_features(sent_speed, sent_steer, rec_delay, n)

        # #VORSAGE: Beschriftung verschieben. Bild t (nur t < T-n) bekommt den Lehrer-Befehl von t + n.
        xs.append(d["x"][skip:T - n]); cmds.append(c[skip:T - n])          # #VORSAGE: ab skip
        steers.append(d["steer"][skip + n:]); speeds.append(d["speed"][skip + n:])
    if not xs:
        raise SystemExit("Keine Episode lang genug.")
    return np.concatenate(xs), np.concatenate(cmds), np.concatenate(steers), np.concatenate(speeds)


def augment(inp, y):
    x, cmds = inp["image"], inp["cmds"]  # #VORSAGE: zwei Inputs
    # random horizontal flip: mirrored image -> mirrored steering angle (speed stays the same)
    flip = tf.random.uniform(()) < 0.5
    x = tf.cond(flip, lambda: tf.image.flip_left_right(x), lambda: x)
    steer = tf.where(flip, -y["steer"], y["steer"])
    # #VORSAGE: beim Spiegeln auch die Lenkwinkel in cmds spiegeln (zweite Hälfte), Geschwindigkeiten bleiben
    half = tf.shape(cmds)[0] // 2
    sign = tf.concat([tf.ones([half]), tf.where(flip, -1.0, 1.0) * tf.ones([half])], axis=0)
    cmds = cmds * sign

    # brightness / contrast jitter (robustness, later also helpful for the real camera)
    xf = tf.cast(x, tf.float32)
    xf = xf * tf.random.uniform((), 0.7, 1.3) + tf.random.uniform((), -25.0, 25.0)
    x = tf.cast(tf.clip_by_value(xf, 0.0, 255.0), tf.uint8)
    return {"image": x, "cmds": cmds}, {"steer": steer, "speed": y["speed"]}


def make_dataset(x, cmds, steer, speed, batch_size, train):
    ds = tf.data.Dataset.from_tensor_slices(({"image": x, "cmds": cmds},  # #VORSAGE: zwei Inputs
                                             {"steer": steer[:, None], "speed": speed[:, None]}))
    if train:
        ds = ds.shuffle(min(len(x), 20000)).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    args = parse_args()
    files = sorted(f for d in args.data_dir for f in glob.glob(os.path.join(d, "*.npz")))  # #VORSAGE: mehrere Ordner
    if not files:
        raise SystemExit(f"No data found in {args.data_dir} - run simulation_cnn_vorhersage.py --mode collect first.")
    n = args.delay_steps  # #VORSAGE
    print(f"Verzögerung: {n} Schritte = {n / 30 * 1000:.0f} ms")  # #VORSAGE

    # split by episode (consecutive frames are almost identical -> a per-frame split would overestimate quality)
    random.seed(0)
    random.shuffle(files)
    n_val = max(1, int(len(files) * args.val_frac)) if len(files) > 1 else 0
    val_files, train_files = files[:n_val], files[n_val:]

    x_tr, c_tr, s_tr, v_tr = load_files(train_files, n)  # #VORSAGE
    print(f"train: {len(train_files)} episodes, {len(x_tr)} frames | "
          f"steer mean {s_tr.mean():+.3f} std {s_tr.std():.3f} | speed mean {v_tr.mean():.3f}")
    ds_train = make_dataset(x_tr, c_tr, s_tr, v_tr, args.batch_size, train=True)  # #VORSAGE

    ds_val = None
    if val_files:
        x_va, c_va, s_va, v_va = load_files(val_files, n)  # #VORSAGE
        print(f"val:   {len(val_files)} episodes, {len(x_va)} frames")
        ds_val = make_dataset(x_va, c_va, s_va, v_va, args.batch_size, train=False)  # #VORSAGE

    model = keras.models.load_model(args.init) if args.init else build_model(n)  # #VORSAGE: n übergeben
    if args.init:  # #VORSAGE: weitertrainiertes Modell muss für dieselbe Verzögerung gebaut sein
        n_model = [t for t in model.inputs if len(t.shape) == 2][0].shape[-1] // 2
        if n_model != n:
            raise SystemExit(f"{args.init} wurde für {n_model} Schritte gebaut, nicht für --delay-steps {n}.")
    model.summary()

    monitor = "val_loss" if ds_val is not None else "loss"
    callbacks = [
        keras.callbacks.ModelCheckpoint(args.out, monitor=monitor, save_best_only=True),
        keras.callbacks.EarlyStopping(monitor=monitor, patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3),
    ]
    model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, callbacks=callbacks)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
