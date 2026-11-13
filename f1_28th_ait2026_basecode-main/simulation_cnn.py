"""
Simulation with a CNN controller. Copy of simulation_singlethreaded.py with three modes:

  collect : the existing lidar based controller (compute_next_command) drives and acts as "teacher".
            Ego camera images + teacher commands are saved as training data (behavior cloning).
  drive   : the trained CNN drives using only the ego camera image.
  dagger  : the CNN drives (mixed with the teacher, see --beta), the teacher labels every frame.
            Collects exactly the situations the CNN gets itself into (fixes "distribution shift").

Typical workflow:
  python simulation_cnn.py --mode collect --episodes 20 --headless --noise-deg 3
  python train_cnn.py
  python simulation_cnn.py --mode drive
  python simulation_cnn.py --mode dagger --episodes 10 --headless
  python train_cnn.py --init cnn_driver.keras        # retrain on all data, repeat dagger/train
"""
import argparse
import math
import os
import random
import time

import numpy as np
import cv2 as cv

from localization.helper_functions import hueToBGR
from tools.OverHeadView import OverHeadView
from tools.simulation import Simulation

from simulation_singlethreaded import (checkFinishLine, vertical_resolution_px, horizontal_resolution_px,
                                       frames_per_seconds, ego_cam_opening_angle_horizontal_degrees,
                                       ego_cam_resolution, height_over_ground_camera_m, ego_cam_pitch_degrees,
                                       meters_to_pixels)
from cnn_controller import (preprocess, CNNDriver, set_skybox_seed, MAX_STEER_RAD, MAX_SPEED_MPS)


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
    return p.parse_args()


def save_episode(data_dir, mode, images, steers, speeds):
    if len(images) == 0:
        return
    os.makedirs(data_dir, exist_ok=True)
    fname = os.path.join(data_dir, f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}_{random.randint(0, 9999):04d}.npz")
    np.savez_compressed(fname,
                        x=np.asarray(images, dtype=np.uint8),
                        steer=np.asarray(steers, dtype=np.float32),
                        speed=np.asarray(speeds, dtype=np.float32))
    print(f"  saved {len(images)} frames -> {fname}")


def main():
    args = parse_args()
    vehicle_starting_position = (850, 1030, math.radians(0.0))  # x_px,y_px,yaw_rad

    sim = Simulation(track_border_filename="track_borders_simulation.npy",
                     vehicle_color="yellow",
                     initial_pose_x_y_yaw=vehicle_starting_position,
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

    ohv = None if args.headless else OverHeadView(backgroundImageFilename="realLifeRennstrecke.png",
                                                  track=sim.racetrack)

    cnn = None
    if args.mode in ("drive", "dagger"):
        cnn = CNNDriver(args.model, steering_smoothing=args.smoothing)

    record = args.mode in ("collect", "dagger")
    target_loop_execution_time = 1.0 / frames_per_seconds
    loop_start_time = time.time()
    episode = 0

    try:
        while args.episodes == 0 or episode < args.episodes:
            episode += 1
            if args.random_skybox:
                set_skybox_seed(sim, random.randint(0, 100000))

            simdata = sim.reset()
            if cnn is not None:
                cnn.reset()
            images, steers, speeds = [], [], []

            while not simdata["simulation_over"] and sim.v is not None:
                obs_bgr = simdata["camera_image_bgr"]

                # ---------------- teacher (lidar based controller) -----------------
                teacher_speed, teacher_steer = None, None
                if record:
                    teacher_speed, teacher_steer, _, _, _ = sim.v.compute_next_command(1.0 / frames_per_seconds, [])
                    teacher_steer = float(np.clip(teacher_steer, -MAX_STEER_RAD, MAX_STEER_RAD))

                    # label = what the teacher would do in this situation
                    images.append(preprocess(obs_bgr))
                    steers.append(teacher_steer / MAX_STEER_RAD)
                    speeds.append(np.clip(teacher_speed / MAX_SPEED_MPS, 0.0, 1.0))

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
                if sim.v.steering_inverted:
                    steering_angle_rad = -steering_angle_rad

                simdata = sim.nextStep(target_velocity_mps=desired_speed_mps,
                                       target_steering_angle_rad=steering_angle_rad)

                if not args.headless:
                    overhead_view = ohv.getImage(sim.v)
                    checkFinishLine(sim.v, overhead_view, top_left=(1000, 950), bottom_right=(1050, 1120))
                    cv.putText(overhead_view,
                               f"[{args.mode}] ep {episode}  lap: {(time.time() - sim.v.time_start_of_lap):.2f}s "
                               f"({sim.v.last_laptime:.2f}s)  progress {simdata['progress'] * 100:.0f}%",
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
            print(f"Episode {episode}: ended ({reason}) after {simdata['current_time']:.1f}s sim time, "
                  f"progress {simdata['progress'] * 100:.1f}%")
            if record:
                save_episode(args.data_dir, args.mode, images, steers, speeds)
    except KeyboardInterrupt:
        print("Stopped by user.")
        if record:
            save_episode(args.data_dir, args.mode, images, steers, speeds)


if __name__ == "__main__":
    main()
