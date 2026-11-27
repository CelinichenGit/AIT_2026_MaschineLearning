"""
#CLEANER: Geht automatisch durch alle DAgger-Aufnahmen (real_dagger*.npz) in cnn_data_real/ und cnn_data_real_safe/,
erkennt Crashes nachträglich und entfernt die betroffenen Frames.

Schritt 1 - Duplikate: Wurde dasselbe Auto gleichzeitig von mehreren Steuerprozessen aufgenommen (alter Bug nach
  Tracking-Verlust), stimmen die Positionsverläufe zur selben Zeit überein. Pro Gruppe bleibt nur die längste Datei.
Schritt 2 - Crashes:

Es gelten dieselben Regeln wie in main_collect_vorhersage_safe.py (Werte unten einstellbar):
  Crash-Anzeichen
    - mehr als CRASH_OUTSIDE_FRACTION der Fahrzeugfläche liegt außerhalb der Trackboundaries (track_borders.npy)
    - "steckt fest": gesendet > CRASH_STUCK_MIN_SENT_MPS, gemessen < CRASH_STUCK_MAX_ACTUAL_MPS, länger als CRASH_STUCK_TIME_S
    - Positionssprung > CRASH_MAX_JUMP_M zwischen zwei Frames
    - Zeitlücke > TRACKING_LOSS_TOLERANCE_S (Tracking lange verloren / Neustart)
  Entfernt werden
    - die Crash-Frames selbst
    - DISCARD_BEFORE_CRASH_FRAMES Frames davor
    - CRASH_RECOVERY_S danach (erst nach so langer "ok"-Fahrt wird wieder behalten)
  Zeitlücken > GAP_SPLIT_S (aber <= TRACKING_LOSS_TOLERANCE_S) sind kein Crash: dort wird die Datei nur geteilt,
  weil train_cnn_vorhersage.py gleichmäßige Zeitabstände erwartet.
  Übrig bleibende Stücke mit weniger als --min-frames Frames werden verworfen.

Ergebnis: jede veränderte Datei wird durch ihre sauberen Stücke ersetzt (<name>_s0.npz, <name>_s1.npz, ...).
Unveränderte Dateien bleiben wie sie sind. Die Originale werden NICHT gelöscht, sondern nach
<ordner>/_vor_crashfilter/ verschoben (außer mit --delete).

Usage:
  python dataset_crash_cleaner.py                 # Probelauf: nur anzeigen, was passieren würde (ändert nichts)
  python dataset_crash_cleaner.py --apply         # wirklich ausführen (Originale -> _vor_crashfilter/)
  python dataset_crash_cleaner.py --apply --delete   # Originale endgültig löschen statt verschieben
  python dataset_crash_cleaner.py --verbose       # pro Datei die Crash-Gründe anzeigen
"""
import argparse
import glob
import os
import re
import shutil
import time

import numpy as np
import shapely

from control.track import Track
from vehicle_config import vehicles

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIRS = [os.path.join(HERE, "cnn_data_real"), os.path.join(HERE, "cnn_data_real_safe")]
FILE_PATTERN = "real_dagger*.npz"
TRACK_FILE = os.path.join(HERE, "track_borders.npy")
BACKUP_SUBDIR = "_vor_crashfilter"

meters_to_pixels = 683            # wie in main_collect_vorhersage_safe.py
CMD_DT = 1.0 / 30.0               # Abstand zweier Frames (30 Hz)

# --- gleiche Regeln/Werte wie in main_collect_vorhersage_safe.py ---------------------------------------------------
DISCARD_BEFORE_CRASH_FRAMES = 30
CRASH_RECOVERY_S = 1.0
CRASH_OUTSIDE_FRACTION = 0.3
CRASH_WALL_MARGIN_M = 0.0
CRASH_STUCK_MIN_SENT_MPS = 0.3
CRASH_STUCK_MAX_ACTUAL_MPS = 0.1
CRASH_STUCK_TIME_S = 1.0
CRASH_MAX_JUMP_M = 0.3
TRACKING_LOSS_TOLERANCE_S = 0.5
# --- nur hier ---------------------------------------------------------------------------------------------------
GAP_SPLIT_S = 0.15                # Zeitlücke, ab der die Datei geteilt wird (normale Schleifen-Überläufe ~0.05 s)
DEFAULT_MIN_FRAMES = 30           # kürzere Stücke werden verworfen (das Training braucht > delay_steps Frames)


def parse_args():
    p = argparse.ArgumentParser(description="Crashes in DAgger-Aufnahmen erkennen und entfernen")
    p.add_argument("--apply", action="store_true", help="wirklich ändern (ohne: nur Probelauf)")
    p.add_argument("--delete", action="store_true", help="Originale löschen statt nach _vor_crashfilter/ verschieben")
    p.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES, help="kürzere Stücke verwerfen")
    p.add_argument("--color", default="yellow", help="Fahrzeug aus vehicle_config.py (Maße für das Rechteck)")
    p.add_argument("--data-dir", action="append", default=None, help="anderer Ordner (mehrfach möglich)")
    p.add_argument("--verbose", action="store_true", help="Crash-Gründe pro Datei anzeigen")
    return p.parse_args()


# ---------------------------------------------------------------------------------------------------------------
def outside_fractions(d, track: Track, veh):
    """Anteil der Fahrzeug-Rechteckfläche außerhalb der Strecke, für jeden Frame (wie Vehicle.getBoundingBox)."""
    length = veh["length_m"] * meters_to_pixels
    half_w = veh["width_m"] * meters_to_pixels / 2
    rear = veh["rear_axle_offset_m"] * meters_to_pixels
    local = np.array([[-rear, -half_w], [-rear, half_w], [-rear + length, half_w], [-rear + length, -half_w]])
    yaw = d["yaw"].astype(np.float64)
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    xs = d["pos_x"][:, None] + c * local[:, 0] - s * local[:, 1]
    ys = d["pos_y"][:, None] + s * local[:, 0] + c * local[:, 1]
    boxes = shapely.polygons(np.stack([xs, ys], axis=-1))
    if CRASH_WALL_MARGIN_M > 0:
        boxes = shapely.buffer(boxes, CRASH_WALL_MARGIN_M * meters_to_pixels)
    area = shapely.area(boxes)
    inside = shapely.area(shapely.intersection(boxes, track.polygon))
    return np.where(area > 0, 1.0 - inside / np.maximum(area, 1e-9), 0.0)


def time_usable(d):
    """Alte Aufnahmen speichern t als float32-Unixzeit: dann sind alle Werte gleich bzw. springen um 128 s."""
    t = d["t"]
    return not (t.dtype == np.float32 and np.abs(t).max() > 1e6)


def analyse(d, track: Track, veh):
    """Gibt (crash[bool T], split_before[bool T], reasons{name: anzahl}) zurück."""
    T = len(d["x"])
    crash = np.zeros(T, bool)
    split_before = np.zeros(T, bool)     # True: zwischen Frame i-1 und i fehlt Zeit -> neues Stück ab i
    reasons = {}

    def mark(idx, name):
        idx = np.asarray(idx, int)
        if len(idx):
            crash[idx] = True
            reasons[name] = reasons.get(name, 0) + len(idx)

    # 1) außerhalb der Strecke
    frac = outside_fractions(d, track, veh)
    mark(np.flatnonzero(frac > CRASH_OUTSIDE_FRACTION), "außerhalb der Strecke")

    # 2) Zeitlücken und Positionssprünge
    t = d["t"].astype(np.float64)
    if not time_usable(d):
        # t wurde als float32-Unixzeit gespeichert (Auflösung 128 s) -> Zeit aus der Frame-Nummer (30 Hz).
        # Zeitlücken sind dann nicht erkennbar, nur Positionssprünge.
        t = np.arange(T) * CMD_DT
    dt = np.diff(t)
    jump_m = np.hypot(np.diff(d["pos_x"]), np.diff(d["pos_y"])) / meters_to_pixels
    for i in range(1, T):
        gap = dt[i - 1]
        if gap > TRACKING_LOSS_TOLERANCE_S:
            mark([i - 1], "Tracking lange verloren / Zeitlücke")   # wie Crash: Frames davor verwerfen
            split_before[i] = True
        elif gap > GAP_SPLIT_S:
            split_before[i] = True                                  # kurze Lücke: nur teilen
        elif jump_m[i - 1] > CRASH_MAX_JUMP_M:
            mark([i], "Positionssprung")

    # 3) steckt fest
    stuck = (d["sent_speed_mps"] > CRASH_STUCK_MIN_SENT_MPS) & (np.abs(d["actual_speed_mps"]) < CRASH_STUCK_MAX_ACTUAL_MPS)
    since = None
    for i in range(T):
        if not stuck[i] or split_before[i]:
            since = None
        if stuck[i]:
            if since is None:
                since = t[i]
            elif t[i] - since > CRASH_STUCK_TIME_S:
                mark([i], "steckt fest")
    return crash, split_before, reasons


def keep_mask(crash, split_before):
    """Crash-Frames + DISCARD_BEFORE_CRASH_FRAMES davor + CRASH_RECOVERY_S danach entfernen."""
    T = len(crash)
    keep = ~crash
    recovery = int(round(CRASH_RECOVERY_S / CMD_DT))
    for i in np.flatnonzero(crash):
        keep[max(0, i - DISCARD_BEFORE_CRASH_FRAMES):i] = False
        # Erholung: ab dem Crash zählt die Zeit neu (Dauer-Crash setzt sie immer wieder zurück, wie online)
        keep[i + 1:min(T, i + 1 + recovery)] = False
    return keep


def segments(keep, split_before):
    """Zusammenhängende Bereiche [a, b) mit keep=True, an Zeitlücken getrennt."""
    out, start = [], None
    for i in range(len(keep) + 1):
        boundary = i == len(keep) or not keep[i] or split_before[i]
        if boundary and start is not None:
            out.append((start, i))
            start = None
        if i < len(keep) and keep[i] and start is None:
            start = i
    return out


def write_segment(d, a, b, fname):
    T = len(d["x"])
    arrays = {}
    for k, v in d.items():
        if k == "skip_first":
            continue
        arrays[k] = v[a:b] if (v.ndim >= 1 and len(v) == T) else v
    skip = int(d["skip_first"]) if "skip_first" in d else 0
    # Vorgeschichte gilt nur, wenn das Stück am Dateianfang beginnt und noch echte Frames danach hat
    arrays["skip_first"] = np.asarray(max(0, skip - a) if a < skip else 0)
    np.savez_compressed(fname, **arrays)


# ---------------------------------------------------------------------------------------------------------------
# Duplikate: mehrere Steuerprozesse haben gleichzeitig dasselbe Auto aufgenommen (alter Bug in main_collect_*).
# Zwei Dateien sind Duplikate, wenn ihre Positionsverläufe zur GLEICHEN Zeit Frame für Frame übereinstimmen.
# Die Zeit kommt aus dem Dateinamen (Speicherzeitpunkt = letzter Frame, 1 s Auflösung), weil t in alten Dateien
# unbrauchbar ist. Nur so werden spätere Runden auf derselben Ideallinie NICHT fälschlich als Duplikat erkannt.
DUP_WINDOW_S = 35.0          # Dateien, deren Speicherzeit weiter auseinander liegt, werden nicht verglichen
DUP_OFFSET_TOL_FRAMES = 45   # erlaubte Abweichung von der aus den Speicherzeiten erwarteten Verschiebung
DUP_MAX_MEAN_DIST_M = 0.02   # mittlerer Positionsabstand, unter dem zwei Verläufe als gleich gelten
DUP_MIN_OVERLAP = 0.5        # mindestens dieser Anteil der kürzeren Datei muss überlappen

_NAME_RE = re.compile(r"_(\d{8})_(\d{6})_\d{4}(?:_s\d+)?\.npz$")


def file_time(f):
    m = _NAME_RE.search(os.path.basename(f))
    if not m:
        return None
    return time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))


def original_stem(f):
    return re.sub(r"_s\d+\.npz$", ".npz", os.path.basename(f))


def is_duplicate(pa, ta, pb, tb):
    """pa, pb: Positionen (n, 2) in Pixeln; ta, tb: Speicherzeit (= Zeit des letzten Frames)."""
    na, nb = len(pa), len(pb)
    # Frame j von B entspricht zeitlich ungefähr Frame j + o_exp von A
    o_exp = (na - 1) - (nb - 1) - (ta - tb) / CMD_DT
    need = max(3, int(DUP_MIN_OVERLAP * min(na, nb)))
    lim = DUP_MAX_MEAN_DIST_M * meters_to_pixels
    for o in range(int(o_exp) - DUP_OFFSET_TOL_FRAMES, int(o_exp) + DUP_OFFSET_TOL_FRAMES + 1):
        j0, j1 = max(0, -o), min(nb, na - o)
        if j1 - j0 < need:
            continue
        dist = np.hypot(*(pa[j0 + o:j1 + o] - pb[j0:j1]).T)
        if dist.mean() < lim:
            return True
    return False


def find_duplicates(files):
    """Gibt {duplikat: behaltene Datei} zurück. Pro Gruppe bleibt die längste Datei."""
    info = []
    for f in files:
        with np.load(f) as z:
            if "pos_x" not in z.files:
                continue
            info.append((f, file_time(f), np.stack([z["pos_x"], z["pos_y"]], 1).astype(np.float64)))
    parent = {f: f for f, _, _ in info}

    def root(f):
        while parent[f] != f:
            f = parent[f]
        return f

    for i, (fa, ta, pa) in enumerate(info):
        for fb, tb, pb in info[i + 1:]:
            if (ta is None or tb is None or abs(ta - tb) > DUP_WINDOW_S or len(pa) == 0 or len(pb) == 0
                    or os.path.dirname(fa) != os.path.dirname(fb) or original_stem(fa) == original_stem(fb)):
                continue
            if is_duplicate(pa, ta, pb, tb):
                parent[root(fb)] = root(fa)

    groups = {}
    length = {f: len(p) for f, _, p in info}
    for f in parent:
        groups.setdefault(root(f), []).append(f)
    dups = {}
    for members in groups.values():
        if len(members) > 1:
            keep = max(members, key=lambda f: length[f])
            for f in members:
                if f != keep:
                    dups[f] = keep
    return dups


def remove_file(f, args):
    if args.delete:
        os.remove(f)
    else:
        backup = os.path.join(os.path.dirname(f), BACKUP_SUBDIR)
        os.makedirs(backup, exist_ok=True)
        shutil.move(f, os.path.join(backup, os.path.basename(f)))


# ---------------------------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    data_dirs = args.data_dir or DATA_DIRS
    veh = vehicles[args.color]
    track = Track()
    track.loadFromFile(TRACK_FILE)

    files = sorted(f for dd in data_dirs for f in glob.glob(os.path.join(dd, FILE_PATTERN)))
    if not files:
        raise SystemExit(f"Keine {FILE_PATTERN} in {data_dirs} gefunden.")

    mode = "AUSFÜHREN" if args.apply else "PROBELAUF (nichts wird geändert, mit --apply ausführen)"
    print(f"{mode}: {len(files)} Dateien, Regeln: >{CRASH_OUTSIDE_FRACTION:.0%} außerhalb, "
          f"{DISCARD_BEFORE_CRASH_FRAMES} Frames davor, {CRASH_RECOVERY_S:.1f} s danach, min. {args.min_frames} Frames\n")

    tot_in = tot_out = n_unchanged = n_changed = n_removed = n_new = n_bad_time = 0
    tot_reasons = {}

    # Schritt 1: Duplikate (mehrere Prozesse haben dasselbe Auto gleichzeitig aufgenommen) -> nur die längste behalten
    dups = find_duplicates(files)
    dup_frames = 0
    for f, keep in sorted(dups.items()):
        with np.load(f) as z:
            n = len(z["x"])
        dup_frames += n
        tot_in += n
        if args.verbose:
            print(f"  Duplikat: {os.path.relpath(f, HERE)} ({n} Frames) = {os.path.basename(keep)}")
        if args.apply:
            remove_file(f, args)
    print(f"Duplikate: {len(dups)} Dateien ({dup_frames} Frames) werden entfernt.\n")
    files = [f for f in files if f not in dups]

    # Schritt 2: Crashes
    for f in files:
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        T = len(d["x"])
        tot_in += T
        name = os.path.relpath(f, HERE)

        if T == 0 or not all(k in d for k in ("pos_x", "pos_y", "yaw", "t", "sent_speed_mps", "actual_speed_mps")):
            print(f"  übersprungen (alte Datei ohne Positionsdaten): {name}")
            tot_out += T
            n_unchanged += 1
            continue

        n_bad_time += not time_usable(d)
        crash, split_before, reasons = analyse(d, track, veh)
        for k, v in reasons.items():
            tot_reasons[k] = tot_reasons.get(k, 0) + v
        segs = [(a, b) for a, b in segments(keep_mask(crash, split_before), split_before) if b - a >= args.min_frames]

        if segs == [(0, T)]:
            n_unchanged += 1
            tot_out += T
            continue

        kept = sum(b - a for a, b in segs)
        tot_out += kept
        if segs:
            n_changed += 1
            n_new += len(segs)
        else:
            n_removed += 1
        info = ", ".join(f"{k} {v}" for k, v in reasons.items()) or "nur zu kurz / Zeitlücke"
        print(f"  {name}: {T} -> {kept} Frames in {len(segs)} Stück(en)" + (f"   [{info}]" if args.verbose else ""))

        if args.apply:
            stem = os.path.splitext(f)[0]
            for k, (a, b) in enumerate(segs):
                write_segment(d, a, b, f"{stem}_s{k}.npz")
            remove_file(f, args)

    print(f"\n{len(dups)} Duplikate entfernt. Übrige {len(files)} Dateien: {n_unchanged} unverändert, {n_changed} bereinigt (-> {n_new} neue Dateien), "
          f"{n_removed} komplett entfernt")
    print(f"Frames: {tot_in} -> {tot_out} ({tot_in - tot_out} entfernt, {100 * (tot_in - tot_out) / max(tot_in, 1):.1f} %)")
    if n_bad_time:
        print(f"Hinweis: {n_bad_time} Dateien haben keinen brauchbaren Zeitstempel (t als float32) -> "
              f"Zeit aus Frame-Nummer, Zeitlücken dort nicht erkennbar.")
    if tot_reasons:
        print("Crash-Frames nach Grund: " + ", ".join(f"{k} {v}" for k, v in tot_reasons.items()))
    if not args.apply:
        print("\nProbelauf - nichts geändert. Zum Ausführen: python dataset_crash_cleaner.py --apply")
    elif not args.delete:
        print(f"Originale liegen in <ordner>/{BACKUP_SUBDIR}/ (werden vom Training nicht geladen).")


if __name__ == "__main__":
    main()
