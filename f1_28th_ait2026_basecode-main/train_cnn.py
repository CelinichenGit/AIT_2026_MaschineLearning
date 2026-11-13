"""
Train the CNN controller on the data recorded by simulation_cnn.py (behavior cloning / DAgger).

  python train_cnn.py                               # new model from scratch
  python train_cnn.py --init cnn_driver.keras       # continue training an existing model (e.g. after DAgger)
"""
import argparse
import glob
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from cnn_controller import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="cnn_data")
    p.add_argument("--out", default="cnn_driver.keras")
    p.add_argument("--init", default=None, help="existing model to continue training from")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.15, help="fraction of EPISODES used for validation")
    return p.parse_args()


def load_files(files):
    xs, steers, speeds = [], [], []
    for f in files:
        d = np.load(f)
        xs.append(d["x"]); steers.append(d["steer"]); speeds.append(d["speed"])
    return np.concatenate(xs), np.concatenate(steers), np.concatenate(speeds)


def augment(x, y):
    # random horizontal flip: mirrored image -> mirrored steering angle (speed stays the same)
    flip = tf.random.uniform(()) < 0.5
    x = tf.cond(flip, lambda: tf.image.flip_left_right(x), lambda: x)
    steer = tf.where(flip, -y["steer"], y["steer"])

    # brightness / contrast jitter (robustness, later also helpful for the real camera)
    xf = tf.cast(x, tf.float32)
    xf = xf * tf.random.uniform((), 0.7, 1.3) + tf.random.uniform((), -25.0, 25.0)
    x = tf.cast(tf.clip_by_value(xf, 0.0, 255.0), tf.uint8)
    return x, {"steer": steer, "speed": y["speed"]}


def make_dataset(x, steer, speed, batch_size, train):
    ds = tf.data.Dataset.from_tensor_slices((x, {"steer": steer[:, None], "speed": speed[:, None]}))
    if train:
        ds = ds.shuffle(min(len(x), 20000)).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not files:
        raise SystemExit(f"No data found in {args.data_dir}/ - run simulation_cnn.py --mode collect first.")

    # split by episode (consecutive frames are almost identical -> a per-frame split would overestimate quality)
    random.seed(0)
    random.shuffle(files)
    n_val = max(1, int(len(files) * args.val_frac)) if len(files) > 1 else 0
    val_files, train_files = files[:n_val], files[n_val:]

    x_tr, s_tr, v_tr = load_files(train_files)
    print(f"train: {len(train_files)} episodes, {len(x_tr)} frames | "
          f"steer mean {s_tr.mean():+.3f} std {s_tr.std():.3f} | speed mean {v_tr.mean():.3f}")
    ds_train = make_dataset(x_tr, s_tr, v_tr, args.batch_size, train=True)

    ds_val = None
    if val_files:
        x_va, s_va, v_va = load_files(val_files)
        print(f"val:   {len(val_files)} episodes, {len(x_va)} frames")
        ds_val = make_dataset(x_va, s_va, v_va, args.batch_size, train=False)

    model = keras.models.load_model(args.init) if args.init else build_model()
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
