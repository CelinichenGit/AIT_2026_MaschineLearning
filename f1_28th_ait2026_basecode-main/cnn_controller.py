"""
Simple CNN controller: ego camera image -> (steering angle, target speed).

Used by simulation_cnn.py (data collection / driving / DAgger) and train_cnn.py (training).
Architecture is based on NVIDIA's "PilotNet" (End to End Learning for Self-Driving Cars, 2016).
"""
import math

import numpy as np
import cv2 as cv

# NOTE: tensorflow is imported lazily inside build_model() / CNNDriver, so that data collection
# (simulation_cnn.py --mode collect) and dataset_tool.py also work without tensorflow and start faster.

from tools.renderer3d import _generate_skybox_pattern_texture

# ---------------------------------------------------------------------------
# Input / output normalization. Training and inference MUST use the same values!
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 66, 200          # network input size (after crop + resize)
CROP_TOP_PX = 110              # rows cut off at the top of the 320x240 ego image (sky / random background).
                                # Check with: python simulation_cnn.py --mode collect --show-input
MAX_STEER_RAD = math.radians(27.0)  # vehicle_config["yellow"]["max_steering_angle"]
MAX_SPEED_MPS = 2.0             # network speed output 0..1 is scaled to 0..MAX_SPEED_MPS


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR ego camera image -> uint8 RGB image of size (IMG_H, IMG_W, 3).

    Kept as uint8 to save memory when storing datasets; scaling to 0..1 happens inside the model.
    """
    img = bgr[CROP_TOP_PX:, :]
    img = cv.resize(img, (IMG_W, IMG_H), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def build_model():
    from tensorflow import keras

    inp =keras.Input((IMG_H, IMG_W, 3), dtype="uint8", name="image")
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    x = keras.layers.Conv2D(24, 5, strides=2, activation="relu")(x)
    x = keras.layers.Conv2D(36, 5, strides=2, activation="relu")(x)
    x = keras.layers.Conv2D(48, 5, strides=2, activation="relu")(x)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(100, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(50, activation="relu")(x)
    x = keras.layers.Dense(10, activation="relu")(x)
    steer = keras.layers.Dense(1, activation="tanh", name="steer")(x)     # -1..1  -> * MAX_STEER_RAD
    speed = keras.layers.Dense(1, activation="sigmoid", name="speed")(x)  #  0..1  -> * MAX_SPEED_MPS

    model = keras.Model(inp, {"steer": steer, "speed": speed}, name="pilotnet_small")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss={"steer": "mse", "speed": "mse"},
                  loss_weights={"steer": 1.0, "speed": 0.5})
    return model


class CNNDriver:
    """Wraps a trained model for use inside the simulation loop."""

    def __init__(self, model_path: str, steering_smoothing: float = 0.5):
        """steering_smoothing: 0 = no filter, towards 1 = strong low-pass filter (reduces jittering)."""
        import tensorflow as tf
        from tensorflow import keras

        self._tf = tf
        self.model = keras.models.load_model(model_path)
        self.alpha = steering_smoothing
        self.last_steer = 0.0
        # compiled forward pass - much faster than model.predict() for single frames
        self._forward = tf.function(lambda x: self.model(x, training=False))

    def reset(self):
        self.last_steer = 0.0

    def predict(self, bgr: np.ndarray):
        """Returns (target_speed_mps, steering_angle_rad)."""
        x = preprocess(bgr)[None]  # add batch dimension
        out = self._forward(self._tf.constant(x))
        steer = float(out["steer"][0, 0]) * MAX_STEER_RAD
        speed = float(out["speed"][0, 0]) * MAX_SPEED_MPS

        steer = self.alpha * self.last_steer + (1.0 - self.alpha) * steer
        self.last_steer = steer
        return speed, steer


def set_skybox_seed(sim, seed: int):
    """Replace the random background of the ego camera at runtime (domain randomization).

    The renderer only creates the skybox texture once in its constructor; this swaps the texture
    without having to recreate the whole OpenGL context.
    """
    import glfw
    from OpenGL import GL as gl

    r = sim.renderer
    glfw.make_context_current(r._window)
    if r._tex_sky is not None:
        gl.glDeleteTextures([r._tex_sky])
    r.skybox_seed = int(seed)
    r._tex_sky = r._create_texture_from_rgb(_generate_skybox_pattern_texture(512, r.skybox_seed))
    