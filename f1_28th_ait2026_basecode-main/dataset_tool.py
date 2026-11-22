"""
Look at / export the data recorded by simulation_cnn.py.

Each cnn_data/*.npz file is one episode. Per frame it contains:
  x                  : CNN input image (66x200 RGB, uint8)
  steer              : LABEL steering angle, normalized -1..1  (* MAX_STEER_RAD = rad), negative = left, positive = right
  speed              : LABEL target speed,   normalized  0..1  (* MAX_SPEED_MPS = m/s)
  actual_speed_mps   : speed the simulated vehicle actually had in this frame
  executed_steer_rad : steering command actually sent (incl. noise / CNN output)
  pos_x, pos_y, yaw  : vehicle pose (pixels / rad)
  t                  : simulation time in s
(older files only contain x, steer, speed)

Usage:
  python dataset_tool.py stats                  # overview of all episodes
  python dataset_tool.py view                   # browse images with their labels
  python dataset_tool.py view --file cnn_data/collect_....npz
  python dataset_tool.py export --out cnn_export --every 5   # PNG images + labels.csv
"""
import argparse
import csv
import glob
import math
import os

import numpy as np
import cv2 as cv

from cnn_controller import MAX_STEER_RAD, MAX_SPEED_MPS

EXTRA_KEYS = ["actual_speed_mps", "executed_steer_rad", "pos_x", "pos_y", "yaw", "t"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["stats", "view", "export"])
    p.add_argument("--data-dir", default="cnn_data")
    p.add_argument("--file", default=None, help="single .npz file instead of the whole data dir")
    p.add_argument("--out", default="cnn_export", help="export: output directory")
    p.add_argument("--every", type=int, default=1, help="export: only every n-th frame")
    p.add_argument("--scale", type=int, default=3, help="view: image magnification")
    return p.parse_args()


def list_files(args):
    if args.file:
        return [args.file]
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not files:
        raise SystemExit(f"No .npz files found in {args.data_dir}/")
    return files


def load(fname):
    d = np.load(fname)
    return {k: d[k] for k in d.files}


def frame_labels(d, i):
    """Labels of frame i in physical units."""
    out = {"steer_deg": float(d["steer"][i]) * math.degrees(MAX_STEER_RAD),
           "speed_mps": float(d["speed"][i]) * MAX_SPEED_MPS}
    for k in EXTRA_KEYS:
        if k in d:
            out[k] = float(d[k][i])
    return out


# ---------------------------------------------------------------------------
def cmd_stats(files):
    total = 0
    print(f"{'file':55s} {'frames':>7s} {'steer mean/std [deg]':>22s} {'speed mean [m/s]':>17s} start")
    for f in files:
        d = load(f)
        n = len(d["x"])
        total += n
        steer_deg = d["steer"] * math.degrees(MAX_STEER_RAD)
        start = "random" if bool(d.get("random_start", False)) else "default"
        print(f"{os.path.basename(f):55s} {n:7d} {steer_deg.mean():+9.1f} / {steer_deg.std():6.1f}      "
              f"{(d['speed'] * MAX_SPEED_MPS).mean():12.2f}      {start}")
    print(f"\n{len(files)} episodes, {total} frames total")


# ---------------------------------------------------------------------------
def draw_frame(d, i, fname, scale):
    img = cv.cvtColor(d["x"][i], cv.COLOR_RGB2BGR)
    img = cv.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
    h, w = img.shape[:2]
    lab = frame_labels(d, i)

    panel = np.full((140, w, 3), 30, np.uint8)

    # steering bar: center = straight, label (green) and executed command (orange)
    bar_y, bar_x0, bar_x1 = 25, 20, w - 20
    cx = (bar_x0 + bar_x1) // 2
    cv.line(panel, (bar_x0, bar_y), (bar_x1, bar_y), (120, 120, 120), 2)
    cv.line(panel, (cx, bar_y - 10), (cx, bar_y + 10), (200, 200, 200), 1)
    max_deg = math.degrees(MAX_STEER_RAD)

    def to_px(deg):
        return int(cx + np.clip(deg / max_deg, -1, 1) * (bar_x1 - cx))

    if "executed_steer_rad" in lab:
        cv.circle(panel, (to_px(math.degrees(lab["executed_steer_rad"])), bar_y), 7, (0, 140, 255), -1)
    cv.circle(panel, (to_px(lab["steer_deg"]), bar_y), 9, (0, 220, 0), 2)
    # sign convention in the simulation (command before steering inversion): negative = left, positive = right
    cv.putText(panel, f"-{max_deg:.0f} (left)", (bar_x0, bar_y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.4,
               (180, 180, 180), 1)
    cv.putText(panel, f"+{max_deg:.0f} deg (right)", (bar_x1 - 105, bar_y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.4,
               (180, 180, 180), 1)

    lines = [f"{os.path.basename(fname)}   frame {i + 1}/{len(d['x'])}",
             f"LABEL  steer {lab['steer_deg']:+6.1f} deg   speed {lab['speed_mps']:.2f} m/s   (green)"]
    if "executed_steer_rad" in lab:
        lines.append(f"actual steer {math.degrees(lab['executed_steer_rad']):+6.1f} deg   "
                     f"speed {lab['actual_speed_mps']:.2f} m/s   (orange)   t={lab['t']:.2f}s")
    lines.append("a/d: frame  w/s: episode  space: play  q: quit")
    for k, txt in enumerate(lines):
        cv.putText(panel, txt, (10, 62 + k * 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv.LINE_AA)
    return np.vstack([img, panel])


def cmd_view(files, scale):
    fi, i, playing = 0, 0, False
    d = load(files[fi])
    while True:
        cv.imshow("dataset", draw_frame(d, i, files[fi], scale))
        key = cv.waitKeyEx(33 if playing else 0)
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            playing = not playing
        elif key in (ord("d"), 2555904):            # right arrow
            i = min(i + 1, len(d["x"]) - 1)
        elif key in (ord("a"), 2424832):            # left arrow
            i = max(i - 1, 0)
        elif key in (ord("w"), ord("s")):
            fi = (fi + (1 if key == ord("w") else -1)) % len(files)
            d, i = load(files[fi]), 0
        elif key == -1 and playing:                 # timeout while playing -> next frame
            i += 1
            if i >= len(d["x"]):
                i, playing = len(d["x"]) - 1, False
    cv.destroyAllWindows()


# ---------------------------------------------------------------------------
def cmd_export(files, out_dir, every):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "labels.csv")
    n = 0
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv, delimiter=";")
        writer.writerow(["image", "episode", "frame", "steer_deg", "speed_mps"] + EXTRA_KEYS)
        for f in files:
            d = load(f)
            episode = os.path.splitext(os.path.basename(f))[0]
            os.makedirs(os.path.join(out_dir, episode), exist_ok=True)
            for i in range(0, len(d["x"]), every):
                rel = f"{episode}/frame_{i:05d}.png"
                cv.imwrite(os.path.join(out_dir, rel), cv.cvtColor(d["x"][i], cv.COLOR_RGB2BGR))
                lab = frame_labels(d, i)
                writer.writerow([rel, episode, i, f"{lab['steer_deg']:.3f}", f"{lab['speed_mps']:.3f}"] +
                                [f"{lab[k]:.4f}" if k in lab else "" for k in EXTRA_KEYS])
                n += 1
    print(f"exported {n} images -> {out_dir}/, labels in {csv_path}")


def main():
    args = parse_args()
    files = list_files(args)
    if args.command == "stats":
        cmd_stats(files)
    elif args.command == "view":
        cmd_view(files, args.scale)
    else:
        cmd_export(files, args.out, max(1, args.every))


if __name__ == "__main__":
    main()
