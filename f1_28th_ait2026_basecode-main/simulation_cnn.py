"""
Simulation with a CNN controller. Copy of simulation_singlethreaded.py with three modes:

  collect : the existing lidar based controller (compute_next_command) drives and acts as "teacher".
            Ego camera images + teacher commands are saved as training data (behavior cloning).
  drive   : the trained CNN drives using only the ego camera image.
  dagger  : the CNN drives (mixed with the teacher, see --beta), the teacher labels every frame.
            Collects exactly the situations the CNN gets itself into (fixes "distribution shift").

Typical workflow:
  python simulation_cnn.py --mode collect --episodes 20 --headless --noise-deg 3 --random-start 0.7
  python train_cnn.py
  python simulation_cnn.py --mode drive
  python simulation_cnn.py --mode dagger --episodes 10 --headless --random-start 0.7
  python train_cnn.py --init cnn_driver.keras        # retrain on all data, repeat dagger/train

Look at the recorded data with:  python dataset_tool.py view
"""
import os
import sys

# Fix for "Segmentation fault" on Linux/Ubuntu with NVIDIA GPU when OpenGL (GLFW renderer) and tensorflow
# run in the same process (see note in simulation_singlethreaded.py). Forces OpenGL to use the NVIDIA driver,
# the same one CUDA/tensorflow uses. Must be set BEFORE any OpenGL library is loaded, i.e. before the imports below.
# Only applied on Linux with an NVIDIA driver, and only if the variable is not already set by the user.
if sys.platform.startswith("linux") and os.path.exists("/proc/driver/nvidia"):
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import argparse
import math
import random
import time

import numpy as np
import cv2 as cv
import shapely
from shapely.geometry import Point, box

from localization.helper_functions import hueToBGR
from tools.OverHeadView import OverHeadView
from tools.simulation import Simulation
from vehicle_config import vehicles

from simulation_singlethreaded import (checkFinishLine, vertical_resolution_px, horizontal_resolution_px,
                                       frames_per_seconds, ego_cam_opening_angle_horizontal_degrees,
                                       ego_cam_resolution, height_over_ground_camera_m, ego_cam_pitch_degrees,
                                       meters_to_pixels)
from cnn_controller import (preprocess, CNNDriver, set_skybox_seed, MAX_STEER_RAD, MAX_SPEED_MPS)

VEHICLE_COLOR = "yellow"
DEFAULT_START_POSE = (850, 1030, math.radians(0.0))  # x_px, y_px, yaw_rad (same as simulation_singlethreaded.py)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["collect", "drive", "dagger"], default="collect")
    p.add_argument("--episodes", type=int, default=0, help="number of runs, 0 = endless")
    p.add_argument("--model", default="cnn_driver.keras")
    p.add_argument("--data-dir", default="cnn_data")
    p.add_argument("--headless", action="store_true",
                   help="no windows, no real-time waiting -> much faster data collection")
    p.add_argument("--show-input", action="store_true", help="show the preprocessed CNN input image")
    p.add_argument("--random-skybox", action="store_true", default=True,
                   help="new random background each episode (default on)")
    p.add_argument("--fixed-skybox", dest="random_skybox", action="store_false")
    p.add_argument("--noise-deg", type=float, default=0.0,
                   help="collect: std dev of steering noise added to the EXECUTED command (label stays clean). "
                        "Makes the vehicle leave the ideal line, so recovery situations are recorded.")
    p.add_argument("--beta", type=float, default=0.3,
                   help="dagger: probability per frame that the teacher's command is executed instead of the CNN's")
    p.add_argument("--smoothing", type=float, default=0.5, help="CNN steering low-pass filter (0..1)")
    p.add_argument("--time-limit", type=float, default=60.0)

    # random start positions
    p.add_argument("--random-start", type=float, default=0.0,
                   help="probability (0..1) that an episode starts at a random point of the track "
                        "instead of the fixed start position. 1.0 = always random")
    p.add_argument("--start-lateral", type=float, default=0.4,
                   help="random start: max. lateral offset from the center line, as fraction of the "
                        "distance to the track border (0 = always on the center line)")
    p.add_argument("--start-yaw-deg", type=float, default=15.0,
                   help="random start: max. deviation of the heading from the track direction in degrees")
    p.add_argument("--seed", type=int, default=None, help="random seed (reproducible start points/backgrounds)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# random start positions
# ---------------------------------------------------------------------------
def car_fits_on_track(sim, x, y, yaw):
    """Same check as in Simulation.nextStep(): is the whole vehicle rectangle inside the track?"""
    length_px = vehicles[VEHICLE_COLOR]["length_m"] * meters_to_pixels
    width_px = vehicles[VEHICLE_COLOR]["width_m"] * meters_to_pixels
    car = box(x - length_px / 2, y - width_px / 2, x + length_px / 2, y + width_px / 2)
    car = shapely.affinity.rotate(geom=car, angle=yaw, use_radians=True)
    return sim.racetrack.polygon.contains(car)


def driving_direction(sim, start_yaw):
    """+1 if the driving direction follows the index order of the center line samples, else -1."""
    c = sim.progressCalculator.center
    heading = math.atan2(c[1][1] - c[0][1], c[1][0] - c[0][0])
    return 1 if math.cos(heading - start_yaw) > 0 else -1


def sample_start_pose(sim, direction, max_lateral_frac, max_yaw_deg, tries=200):
    """Random pose on the track, roughly pointing in driving direction.

    Picks a random point of the center line (computed by TrackProgressGated), shifts it sideways
    and rotates the heading a bit. Retries until the vehicle fits completely on the track.
    """
    center = sim.progressCalculator.center
    n = len(center)
    outer, inner = sim.racetrack.outerRing, sim.racetrack.innerRing
    for _ in range(tries):
        i = random.randrange(n)
        c = center[i]
        nxt = center[(i + 5 * direction) % n]  # a few samples ahead -> smoother heading
        heading = math.atan2(nxt[1] - c[1], nxt[0] - c[0])

        # lateral offset perpendicular to the heading, limited by the distance to the nearest border
        half_width = min(outer.distance(Point(c)), inner.distance(Point(c)))
        offset = random.uniform(-max_lateral_frac, max_lateral_frac) * half_width
        x = c[0] - math.sin(heading) * offset
        y = c[1] + math.cos(heading) * offset

        yaw = heading + math.radians(random.uniform(-max_yaw_deg, max_yaw_deg))
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        if car_fits_on_track(sim, x, y, yaw):
            return (int(round(x)), int(round(y)), yaw)
    print("  WARNING: no valid random start pose found, using the default start")
    return DEFAULT_START_POSE


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------
def save_episode(data_dir, mode, rec: dict, meta: dict):
    if len(rec["x"]) == 0:
        return
    os.makedirs(data_dir, exist_ok=True)
    fname = os.path.join(data_dir, f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}_{random.randint(0, 9999):04d}.npz")
    arrays = {k: np.asarray(v, dtype=np.uint8 if k == "x" else np.float32) for k, v in rec.items()}
    arrays.update({k: np.asarray(v) for k, v in meta.items()})
    np.savez_compressed(fname, **arrays)
    print(f"  saved {len(rec['x'])} frames -> {fname}")


def new_recording():
    # x, steer, speed      : training input + labels (used by train_cnn.py), normalized like the network outputs
    # all other entries    : extra information per frame for analysis / dataset_tool.py, not used for training
    return {"x": [], "steer": [], "speed": [],
            "actual_speed_mps": [], "executed_steer_rad": [],
            "pos_x": [], "pos_y": [], "yaw": [], "t": []}


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    sim = Simulation(track_border_filename="track_borders_simulation.npy",
                     vehicle_color=VEHICLE_COLOR,
                     initial_pose_x_y_yaw=DEFAULT_START_POSE,
                     meters_to_pixels=meters_to_pixels,
                     vertical_resolution_px=vertical_resolution_px,
                     height_over_ground_camera_m=height_over_ground_camera_m,
                     dt=1.0 / frames_per_seconds,
                     time_limit_s=args.time_limit,
                     track_png_path="realLifeRennstrecke.png",
                     track_size_px=(horizontal_resolution_px, vertical_resolution_px),
                     egocam_render_size=ego_cam_resolution,
                     egocam_fov_y_deg=ego_cam_opening_angle_horizontal_degrees,
                     egocam_pitch_deg=ego_cam_pitch_degrees,
                     egocam_skybox_seed=4)
    direction = driving_direction(sim, DEFAULT_START_POSE[2])

    ohv = None if args.headless else OverHeadView(backgroundImageFilename="realLifeRennstrecke.png",
                                                  track=sim.racetrack)

    cnn = None
    if args.mode in ("drive", "dagger"):
        cnn = CNNDriver(args.model, steering_smoothing=args.smoothing)

    record = args.mode in ("collect", "dagger")
    target_loop_execution_time = 1.0 / frames_per_seconds
    loop_start_time = time.time()
    episode = 0
    rec, meta = new_recording(), {}

    try:
        while args.episodes == 0 or episode < args.episodes:
            episode += 1
            skybox_seed = 4
            if args.random_skybox:
                skybox_seed = random.randint(0, 100000)
                set_skybox_seed(sim, skybox_seed)

            # ---------------- choose the start pose of this episode ----------------
            random_start = random.random() < args.random_start
            start_pose = sample_start_pose(sim, direction, args.start_lateral, args.start_yaw_deg) \
                if random_start else DEFAULT_START_POSE
            sim.initial_pose_x_y_yaw = start_pose  # read by sim.reset() when creating the vehicle

            simdata = sim.reset()

            # progress calculation: start counting at the actual start position
            sim.last_progress = sim.progressCalculator.progress(vehicle_xy=start_pose[0:2], prev_progress=None,
                                                                yaw=start_pose[2])["progress_frac"]
            prev_progress = sim.last_progress
            laps_driven = 0.0

            if cnn is not None:
                cnn.reset()
            rec = new_recording()
            meta = {"skybox_seed": skybox_seed, "start_pose": np.asarray(start_pose, dtype=np.float32),
                    "random_start": random_start}

            while not simdata["simulation_over"] and sim.v is not None:
                obs_bgr = simdata["camera_image_bgr"]

                # ---------------- teacher (lidar based controller) -----------------
                teacher_speed, teacher_steer = None, None
                if record:
                    teacher_speed, teacher_steer, _, _, _ = sim.v.compute_next_command(1.0 / frames_per_seconds, [])
                    teacher_steer = float(np.clip(teacher_steer, -MAX_STEER_RAD, MAX_STEER_RAD))

                # ---------------- choose the command that is actually executed ------
                if args.mode == "collect":
                    desired_speed_mps = teacher_speed
                    steering_angle_rad = teacher_steer + math.radians(np.random.normal(0.0, args.noise_deg)) \
                        if args.noise_deg > 0 else teacher_steer
                else:
                    desired_speed_mps, steering_angle_rad = cnn.predict(obs_bgr)
                    if args.mode == "dagger" and random.random() < args.beta:
                        desired_speed_mps, steering_angle_rad = teacher_speed, teacher_steer

                # ---------------- unchanged part of the original loop ---------------
                steering_angle_rad = np.clip(steering_angle_rad,
                                             math.radians(sim.v.min_steering_angle_deg),
                                             math.radians(sim.v.max_steering_angle_deg))

                # ---------------- record: image + labels + extra info ---------------
                if record:
                    rec["x"].append(preprocess(obs_bgr))
                    rec["steer"].append(teacher_steer / MAX_STEER_RAD)  # label = what the teacher would do
                    rec["speed"].append(np.clip(teacher_speed / MAX_SPEED_MPS, 0.0, 1.0))
                    rec["actual_speed_mps"].append(simdata["vehicle_speed_mps"])
                    rec["executed_steer_rad"].append(float(steering_angle_rad))
                    rec["pos_x"].append(sim.v.x)
                    rec["pos_y"].append(sim.v.y)
                    rec["yaw"].append(sim.v.yaw)
                    rec["t"].append(simdata["current_time"])

                if sim.v.steering_inverted:
                    steering_angle_rad = -steering_angle_rad

                simdata = sim.nextStep(target_velocity_mps=desired_speed_mps,
                                       target_steering_angle_rad=steering_angle_rad)

                # distance driven in laps (progress decreases while driving forward, wraps at 0/1)
                laps_driven += ((prev_progress - sim.last_progress + 0.5) % 1.0) - 0.5
                prev_progress = sim.last_progress

                if not args.headless:
                    overhead_view = ohv.getImage(sim.v)
                    checkFinishLine(sim.v, overhead_view, top_left=(1000, 950), bottom_right=(1050, 1120))
                    cv.putText(overhead_view,
                               f"[{args.mode}] ep {episode}  lap: {(time.time() - sim.v.time_start_of_lap):.2f}s "
                               f"({sim.v.last_laptime:.2f}s)  driven {laps_driven:.2f} laps",
                               (500, 50), cv.FONT_HERSHEY_SIMPLEX, 1, hueToBGR(120 / 2), 2, cv.LINE_AA)
                    cv.imshow("Ego-camera OpenGL", simdata["camera_image_bgr"])
                    cv.imshow("Overhead View", cv.resize(overhead_view, (0, 0), fx=0.5, fy=0.5))
                    if args.show_input:
                        cv.imshow("CNN input", cv.cvtColor(cv.resize(preprocess(simdata["camera_image_bgr"]),
                                                                     (0, 0), fx=2, fy=2), cv.COLOR_RGB2BGR))
                    cv.waitKey(1)

                    actual_loop_execution_time = time.time() - loop_start_time
                    loop_start_time = time.time()
                    if actual_loop_execution_time < target_loop_execution_time:
                        time.sleep(target_loop_execution_time - actual_loop_execution_time)

            reason = "time limit" if simdata["over_time_limit"] else "off track"
            start_txt = f"random start ({start_pose[0]},{start_pose[1]}, {math.degrees(start_pose[2]):.0f} deg)" \
                if random_start else "default start"
            print(f"Episode {episode}: {start_txt}, ended ({reason}) after {simdata['current_time']:.1f}s sim time, "
                  f"driven {laps_driven:.2f} laps")
            if record:
                save_episode(args.data_dir, args.mode, rec, meta)
    except KeyboardInterrupt:
        print("Stopped by user.")
        if record:
            save_episode(args.data_dir, args.mode, rec, meta)


if __name__ == "__main__":
    main()
