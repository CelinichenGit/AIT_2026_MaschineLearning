import socket

from localization.camera import Camera
from localization.vehicle import Vehicle
from localization.helper_functions import hueToBGR, createPlot
from localization.pid_controller import PIDController
from control.track import Track
from control.disparity_extender import DisparityExtender
from control.lidarSensor import LidarSensor
from shapely import LineString, Polygon  # #SAFE: Polygon für die Wandkontakt-Erkennung
import numpy as np
import math

from multiprocessing import Process, Manager, Value
from multiprocessing.shared_memory import SharedMemory
import cv2 as cv
import time
import copy
import collections
import os      # #VORSAGE: Ordner für die Aufnahmen
import random  # #VORSAGE: Dateinamen, DAgger-Mischung


#ip addresses and parameters of vehicles
from vehicle_config import vehicles

#camera parameters
vertical_resolution_px = 1200
horizontal_resolution_px = 1920
frames_per_seconds = 90
opening_angle_vertical_degrees = 88.0
opening_angle_horizontal_degrees = 126.0

#physical parameters of camera and vehicle features
meters_to_pixels = 683     #how many pixels are in one meter?1 m entspricht 683 Pixeln im Bild.
max_speed_vehicle_mps = 5.0        #max speed of a car in meters per second
minimum_brightness = 1.0 #2.7   #used to brighten the image of the webcam Werte >1 machen das Bild heller.
threshold_brightness_of_black = 150       #rgb from 0-255
threshold_brightness_of_white = 200        #rgb from 0-255
circle_diameter_meters = 0.025  #diameter of black and white dots (2cm)
size_between_black_and_white_center_meters = 0.08  #8cm from center to center
height_over_ground_black_meters = 0.055    #how high is the black dot on the vehicle measured from the ground
height_over_ground_white_meters = 0.035    #how high is the white dot on the vehicle measured from the ground
camera_exposure_time_absolute = 50 # exposure of the camera, known working configuration from v4l2-ctl: exposure_time_absolute 0x009a0902 (int)    : min=1 max=10000 step=1 default=156 value=50
                                   # DO NOT INCREASE EXPOSURE ABOVE 50 AS THE IMAGE WILL GET BLURRED!!!!
                                   #Belichtungszeit der Kamera.Kleinere Werte → dunkler, aber schärfer.Größere Werte → heller, aber Bewegungen verwischen.

#camera correction parameters
cameraMatrix = np.array([[1.19164513e+03, 0.00000000e+00, 9.32255365e+02],
                         [0.00000000e+00, 1.19269246e+03, 5.44789222e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distortionCoefficients = np.array([[ 0.02473071, -0.39668063,  0.00151336,  0.00085757,  0.25759047]])

frontcam_image_shape = (240, 320, 3) # small resolution to reduce bandwidth and latency.Höhe #Frontkamera   = 240 Pixel Breite = 320 Pixel Kanäle = 3

#SAFE: main_collect_vorhersage_safe.py = Kopie von main_collect_vorhersage.py, die NUR "saubere" Fahrdaten speichert.
#       Alle Frames laufen zuerst durch einen Puffer von DISCARD_BEFORE_CRASH_FRAMES Frames. Erst wenn sie älter als dieser
#       Puffer sind und in der Zwischenzeit kein Crash erkannt wurde, werden sie an den Recorder übergeben.
#       Bei einem Crash wird der Puffer (= die letzten 30 Frames vor dem Crash) verworfen, die Datei-Zeitreihe beendet und
#       erst wieder aufgenommen, wenn das Fahrzeug CRASH_RECOVERY_S Sekunden ohne Crash-Anzeichen gefahren ist.
#       Crash-Anzeichen: Fahrzeug-Rechteck berührt/überschreitet eine Streckenbegrenzung, Fahrzeug steckt fest
#       (Gas gegeben, aber keine Bewegung), Positionssprung (Auto wurde umgesetzt) oder Tracking verloren.
#VORSAGE: main_collect_vorhersage.py = Kopie von main_vorhersage.py zum DATEN SAMMELN auf der echten Strecke.
#         Gespeichert wird im selben Format wie simulation_cnn_vorhersage.py -> direkt nutzbar mit train_cnn_vorhersage.py.
#         Lehrer ist der Lidar-Controller (Position aus der Deckenkamera), er beschriftet jedes Frontkamera-Bild.
COLLECT_MODE = "drive"          # #VORSAGE: "collect" = Lidar-Lehrer fährt | "dagger" = Vorhersage-CNN fährt, Lehrer beschriftet
                                  # #SAFE:    "drive"   = NUR das CNN fährt (kein Lehrer-Eingriff), es wird NICHTS gespeichert.
                                  #           Crash-Erkennung läuft trotzdem und meldet Crashes in der Konsole.
assert COLLECT_MODE in ("collect", "dagger", "drive"), f"unbekannter COLLECT_MODE '{COLLECT_MODE}'"
SAVE_DATA = COLLECT_MODE != "drive"  # #SAFE: im drive-Modus wird keine Datei geschrieben
CNN_DRIVES = COLLECT_MODE in ("dagger", "drive")  # #SAFE: Modus, in dem das CNN fährt (Modell wird geladen)
COLLECT_VEHICLE_COLOR = "yellow"  # #VORSAGE: nur dieses Fahrzeug hat die Frontkamera und wird aufgenommen
# #VORSAGE: Ordner für die echten Daten. Immer neben diesem Skript, egal aus welchem Ordner es gestartet wird.
# #SAFE: eigener Ordner, damit gefilterte und ungefilterte Daten getrennt bleiben
#        (train_cnn_vorhersage.py kann mehrere Ordner gleichzeitig laden)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_data_real_safe")
NOISE_DEG = 3.0                   # #VORSAGE: collect: Lenkrauschen (Std.-Abw. in Grad) auf den gefahrenen Befehl, 0 = aus
DAGGER_BETA = 0.3                 # #VORSAGE: dagger: Wahrscheinlichkeit pro Frame, dass der Lehrer statt des CNN fährt
REAL_DELAY_S = None               # #VORSAGE: Verzögerung Frontkamera-Bild -> Ausführung in s. None = delay_ms aus vehicle_config.py
LABEL_DELTA_T = 0.0               # #VORSAGE: Lehrer-Beschriftung für die aktuelle Position (keine eigene Vorhersage)
SAVE_EVERY_FRAMES = 900           # #VORSAGE: alle 900 Frames (= 30 s) eine Datei schreiben

# #SAFE: Crash-Filter
DISCARD_BEFORE_CRASH_FRAMES = 30  # #SAFE: so viele Frames VOR einem Crash werden verworfen (30 Frames = 1 s bei 30 Hz)
CRASH_RECOVERY_S = 1.0            # #SAFE: nach einem Crash erst wieder aufnehmen, wenn so lange alles ok war
DISCARD_PENDING_AT_END = True     # #SAFE: beim Beenden (Ctrl+C, meist nach einem Crash) die letzten 30 Frames verwerfen
DISCARD_ON_TRACKING_LOSS = True   # #SAFE: Tracking verloren (Auto hochgehoben/umgekippt) wie einen Crash behandeln
TRACKING_LOSS_TOLERANCE_S = 0.5   # #SAFE: ... aber erst, wenn das Tracking LÄNGER als diese Zeit weg ist.
                                  #        Kürzere Aussetzer: kein Crash, nichts wird verworfen, nur die Datei wird
                                  #        an der Lücke geteilt (die Zeitreihe hat dort fehlende Frames).
CRASH_OUTSIDE_FRACTION = 0.3      # #SAFE: Crash erst, wenn mehr als dieser Anteil (0.3 = 30 %) der Fahrzeugfläche
                                  #        außerhalb der Trackboundaries liegt. 0.0 = altes Verhalten (jede Berührung)
CRASH_WALL_MARGIN_M = 0.0         # #SAFE: vergrößert das Fahrzeug-Rechteck um diesen Rand (m), bevor der Anteil berechnet wird
CRASH_STUCK_MIN_SENT_MPS = 0.3    # #SAFE: "steckt fest": gesendete Geschwindigkeit mindestens so hoch ...
CRASH_STUCK_MAX_ACTUAL_MPS = 0.1  # #SAFE: ... aber gemessene Geschwindigkeit darunter ...
CRASH_STUCK_TIME_S = 1.0          # #SAFE: ... und das länger als diese Zeit
CRASH_STARTUP_IGNORE_S = 2.0      # #SAFE: in den ersten Sekunden nach dem Start kein "steckt fest" (Anfahren)
CRASH_MAX_JUMP_M = 0.3            # #SAFE: Positionssprung pro Frame (1/30 s), ab dem das Auto als umgesetzt gilt
# Geschwindigkeit: "speed_factor" in vehicle_config.py (wirkt auf den Lehrer -> auch auf die Beschriftung)

USE_CNN_CONTROLLER = True          # #VORSAGE: sorgt nur noch dafür, dass die Frontkamera immer eingeschaltet wird
CNN_MODEL_PATH = "cnn_driver_vorhersage.keras"  # #VORSAGE: Modell für den dagger-Modus
CNN_STEERING_SMOOTHING = 0.5   # 0 = no filter, towards 1 = strong low-pass filter


# this code is running in another process. 
# Data is transfered from the main thread using the d parameter
def showImageThread(d: dict, track: Track, shm_front_pointer: Value):

    print("In show Image Thread.")

    history_plot_motor_values = dict()
    history_plot_delta_speed  = dict()
    history_plot_lateral_speed = dict()
    history_plot_lateral_acceleration = dict()

    for color in Camera.ColorMap.keys():
        history_plot_motor_values[color] = collections.deque(maxlen=100)
        history_plot_delta_speed[color] = collections.deque(maxlen=100)
        history_plot_lateral_speed[color] = collections.deque(maxlen=100)
        history_plot_lateral_acceleration[color] = collections.deque(maxlen=100)

    #connect to shared memory
    shm = SharedMemory(name="frame")

    while d["showVisualization"]:
        vehicles :list[Vehicle]  = d["vehicles"]

        frame = np.ndarray((vertical_resolution_px, horizontal_resolution_px, 3), dtype=np.uint8, buffer=shm.buf).copy()

        if frame is not None:
            #draw control loop frequency:
            i = 1
            for vehicle in vehicles:
                if vehicle.color is not None:
                    if "control_loop_frequency_" + vehicle.color in d.keys():
                        cv.putText(frame, f"Control frequency: {d['control_loop_frequency_' + vehicle.color]:.1f} Hz", (50, 50+i*20), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                            hueToBGR(Camera.ColorMap[vehicle.color]), 1, cv.LINE_AA)
                        i += 1
            
            #draw laptimes:
            i = 0
            for v in vehicles:
                cv.putText(frame, f"Lap time: {(time.time()-v.time_start_of_lap):.2f}s ({v.last_laptime:.2f}s)", (850, 50+i*40), cv.FONT_HERSHEY_SIMPLEX, 1,
                        hueToBGR(Camera.ColorMap[v.color]), 2, cv.LINE_AA)
                i += 1

            #draw racetrack:
            for idx in range(-1, len(track.innerBounds)-1):
                cv.line(frame, track.innerBounds[idx], track.innerBounds[idx+1], (255,255,255), 2)

            for idx in range(-1, len(track.outerBounds)-1):
                cv.line(frame, track.outerBounds[idx], track.outerBounds[idx+1], (255,255,255), 2)

            #draw vehicle bounding boxes:
            for vehicle in vehicles:
                boundingbox = vehicle.getBoundingBox(time.time()) #TODO: think about which time to use here!
                color = (0, 255, 0)
                if vehicle.ttl < 15:
                    color = (0, 0, 255)
                Camera.drawBoundingBox(frame, boundingbox, color=color)
                cv.putText(frame, f"Speed: {vehicle.getSpeed():.2f} m/s", (int(boundingbox[0][0]), int(boundingbox[0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,255,0), 1, cv.LINE_AA)
                
            #draw lidar rays and setpoint:
            for color in Camera.ColorMap.keys():
                if "lidar_rays_"+color in d:
                    rays : list[LineString] = d["lidar_rays_"+color] #rays:      the rays sent out by the lidar -> list of shapely.LineString objects
                    if rays is not None:
                        for ray in rays:
                            xy = ray.coords.xy
                            cv.line(frame, [int(xy[0][0]), int(xy[1][0])], [int(xy[0][1]), int(xy[1][1])], hueToBGR(Camera.ColorMap[color]), 1)

                
                if "setpoints_"+color in d:
                    sp = d["setpoints_"+color]
                    if sp is not None:    
                        c = [int(sp[0]), int(sp[1])]
                        cv.circle(frame, c, radius=5, color=hueToBGR(Camera.ColorMap[color]), thickness=5, lineType=1)
                

            #plot target speed over demanded motor voltage
            i = -1
            for vehicle in vehicles:
                color = vehicle.color
                if "target_velocity_delta_mps_"+color in d and "motor_voltage_"+color in d:
                    i += 1
                    dy = 100    #height of y axis.
                    plot_x, plot_y = 20, 400 +i*dy+20
                    
                    delta_speed_m = d["target_velocity_delta_mps_" +  color]
                    motor_value = d["motor_voltage_" +  color]
                    history_plot_motor_values[color].append((motor_value / 255)*dy)
                    history_plot_delta_speed[color].append((delta_speed_m/3)   *dy)

                    bgr = hueToBGR(Camera.ColorMap[color])

                    createPlot(frame, 
                               plot_x, 
                               plot_y, 
                               plot_height=dy, 
                               title="Velocity delta in black and applied motor voltage in color",
                               colors=[bgr, [0,0,0]],
                               values=[list(history_plot_motor_values[color]), list(history_plot_delta_speed[color])])


            #show frontcam image if enabled#FrontKamera
            if shm_front_pointer is not None:
                current_buffer_index = shm_front_pointer.value
                next_buffer_index = 1 - current_buffer_index  # switch to the other buffer

                # Create a NumPy array pointing directly to the shared memory buffer
                shm_front = SharedMemory(name="front")
                shm_buffers = [np.ndarray(frontcam_image_shape, dtype=np.uint8, buffer=shm_front.buf, offset=0),
                                np.ndarray(frontcam_image_shape, dtype=np.uint8, buffer=shm_front.buf, offset=math.prod(frontcam_image_shape) * np.dtype(np.uint8).itemsize)]

                front_frame = shm_buffers[current_buffer_index].copy()


                #copy front_frame to the top left corner of the main frame (1600, 50)
                frame[50:50+frontcam_image_shape[0], 1600:1600+frontcam_image_shape[1]] = front_frame


            #show top-view camera image
            cv.imshow('frame', frame)

            if cv.waitKey(1) == ord('q'):
                break

# this code is running in another process.
# Data is transfered from the main thread using the d parameter
class EpisodeRecorder:
    """#VORSAGE: sammelt die Frames einer Fahrt und speichert sie regelmäßig als .npz (Format wie simulation_cnn_vorhersage.py).

    Damit bei einem Absturz nicht alles verloren ist, wird alle SAVE_EVERY_FRAMES Frames eine Datei geschrieben.
    Jede Folgedatei beginnt mit den letzten delay_steps Frames der vorherigen Datei ("skip_first"): diese dienen
    nur als Vorgeschichte für den cmds-Input (Befehle, die noch unterwegs waren) und werden nicht als Beispiel trainiert.
    """

    def __init__(self, data_dir, mode, color, delay_steps, save_every):
        self.data_dir, self.mode, self.color = data_dir, mode, color
        self.delay_steps, self.save_every = delay_steps, save_every
        self.context = None   # letzte delay_steps Frames der vorherigen Datei derselben Fahrt
        self.rec = self._new()
        self.total = 0        # #VORSAGE: Anzahl aller aufgenommenen Frames (für die Fortschrittsanzeige)

    @staticmethod
    def _new():
        return {"x": [], "steer": [], "speed": [], "executed_steer_rad": [], "sent_speed_mps": [],
                "actual_speed_mps": [], "pos_x": [], "pos_y": [], "yaw": [], "t": [], "new_image": []}

    def add(self, **frame):
        for k, val in frame.items():
            self.rec[k].append(val)
        self.total += 1
        if self.total % 150 == 1:  # #VORSAGE: alle ~5 s ein Lebenszeichen
            print(f"  Aufnahme läuft: {self.total} Frames, nächste Datei nach {self.save_every} Frames -> {self.data_dir}")
        if len(self.rec["x"]) >= self.save_every:
            self.flush(end_of_segment=False)

    def flush(self, end_of_segment: bool):
        """Datei schreiben. end_of_segment=True: Fahrt unterbrochen (Tracking verloren / Ende) -> keine Vorgeschichte."""
        if len(self.rec["x"]) == 0:
            if end_of_segment:
                self.context = None
            return
        skip = 0
        data = self.rec
        if self.context is not None:
            skip = len(self.context["x"])
            data = {k: self.context[k] + self.rec[k] for k in self.rec}

        if not SAVE_DATA:  # #SAFE: drive-Modus -> nichts schreiben, nur Puffer leeren
            self.context = None
            self.rec = self._new()
            return
        os.makedirs(self.data_dir, exist_ok=True)
        fname = os.path.join(self.data_dir, f"real_{self.mode}_{self.color}_{time.strftime('%Y%m%d_%H%M%S')}"
                                            f"_{random.randint(0, 9999):04d}.npz")
        # #SAFE: t als float64 - in float32 hat die Unixzeit nur 128 s Auflösung (alle t einer Datei wären gleich)
        arrays = {k: np.asarray(v, dtype=np.uint8 if k == "x" else np.float64 if k == "t" else np.float32)
                  for k, v in data.items()}
        np.savez_compressed(fname, **arrays,
                            delay_steps=np.asarray(self.delay_steps),   # Verzögerung dieser Aufnahme (in 1/30 s)
                            skip_first=np.asarray(skip),                # erste Frames nur Vorgeschichte
                            real=np.asarray(True))
        print(f"  saved {len(self.rec['x'])} frames (+{skip} Vorgeschichte) -> {fname}")

        self.context = None if (end_of_segment or self.delay_steps == 0) else \
            {k: data[k][-self.delay_steps:] for k in data}
        self.rec = self._new()


class CrashDetector:
    """#SAFE: erkennt Crash-Anzeichen anhand der Deckenkamera-Position. check() gibt None oder einen Grund (str) zurück."""

    def __init__(self, track: Track):
        self.track = track
        self.stuck_since = None
        self.last_pos = None

    def reset(self):
        self.stuck_since = None
        self.last_pos = None

    def check(self, v: Vehicle, sent_speed_mps: float, t: float, startup: bool):
        reason = None

        # 1) Wandkontakt: erst wenn mehr als CRASH_OUTSIDE_FRACTION der Fahrzeug-Rechteckfläche außerhalb der Strecke
        #    liegt. Ein leichtes Überschreiten der Begrenzung (Kurven schneiden, Tracking-Rauschen) ist kein Crash.
        #    (Die frühere Mittelpunkt-Prüfung ist hier mit drin: Mittelpunkt draußen ~ 50 % der Fläche draußen.)
        try:
            bbox = Polygon(v.getBoundingBox())
            if CRASH_WALL_MARGIN_M > 0:
                bbox = bbox.buffer(CRASH_WALL_MARGIN_M * meters_to_pixels)
            if bbox.area > 0:
                outside_fraction = 1.0 - bbox.intersection(self.track.polygon).area / bbox.area
                if outside_fraction > CRASH_OUTSIDE_FRACTION:
                    reason = f"außerhalb der Strecke ({outside_fraction * 100:.0f} % des Fahrzeugs)"
        except Exception as e:  # Geometrie-Fehler sollen die Fahrt nicht abbrechen
            print(f"  CrashDetector: Wandprüfung fehlgeschlagen ({e})")

        # 2) Positionssprung: Auto wurde hochgehoben/umgesetzt oder Tracking springt
        pos = (float(v.x), float(v.y))
        if reason is None and self.last_pos is not None:
            jump_m = math.dist(pos, self.last_pos) / meters_to_pixels
            if jump_m > CRASH_MAX_JUMP_M:
                reason = f"Positionssprung {jump_m:.2f} m"
        self.last_pos = pos

        # 3) Steckt fest: Gas gegeben, aber keine Bewegung (z. B. Front an der Wand)
        if not startup and sent_speed_mps > CRASH_STUCK_MIN_SENT_MPS and abs(v.getSpeed()) < CRASH_STUCK_MAX_ACTUAL_MPS:
            if self.stuck_since is None:
                self.stuck_since = t
            elif reason is None and t - self.stuck_since > CRASH_STUCK_TIME_S:
                reason = "steckt fest (keine Bewegung trotz Gas)"
        else:
            self.stuck_since = None

        return reason


class CrashFilterRecorder:
    """#SAFE: Puffer vor dem EpisodeRecorder. Frames werden erst nach discard_frames Frames weitergegeben.
    Bei einem Crash wird der Puffer verworfen und die Aufnahme für recovery_frames Frames ausgesetzt."""

    def __init__(self, recorder: EpisodeRecorder, discard_frames: int, recovery_frames: int):
        self.recorder = recorder
        self.discard_frames = discard_frames
        self.recovery_frames = recovery_frames
        self.pending = collections.deque()
        self.blocked = 0          # verbleibende Frames bis die Aufnahme nach einem Crash wieder startet
        self.in_crash = False
        self.crashes = 0
        self.kept = 0
        self.discarded = 0

    def add(self, **frame):
        if self.blocked > 0:
            self.blocked -= 1
            self.discarded += 1
            if self.blocked == 0:
                self.in_crash = False
                print(f"  CrashFilter: Fahrzeug wieder ok -> Aufnahme läuft weiter "
                      f"(behalten {self.kept}, verworfen {self.discarded}, Crashes {self.crashes})")
            return
        self.pending.append(frame)
        if len(self.pending) > self.discard_frames:
            self.recorder.add(**self.pending.popleft())
            self.kept += 1

    def crash(self, reason: str):
        if not self.in_crash:
            self.crashes += 1
            print(f"  CRASH erkannt ({reason}): {len(self.pending)} Frames "
                  f"({len(self.pending) * CMD_DT_SAFE:.1f} s) davor werden verworfen.")
        self.in_crash = True
        self.discarded += len(self.pending)
        self.pending.clear()
        self.recorder.flush(end_of_segment=True)  # Zeitreihe ist unterbrochen
        self.blocked = self.recovery_frames        # bei Dauer-Crash (steckt fest) immer wieder neu gesetzt

    def gap(self):
        """#SAFE: kurze Lücke ohne Crash (z. B. kurzer Tracking-Aussetzer): gepufferte Frames BEHALTEN,
        aber die Zeitreihe an der Lücke beenden, weil dort Frames fehlen."""
        for frame in self.pending:
            self.recorder.add(**frame)
            self.kept += 1
        self.pending.clear()
        self.recorder.flush(end_of_segment=True)

    def finish(self):
        if DISCARD_PENDING_AT_END:
            self.discarded += len(self.pending)
        else:
            for frame in self.pending:
                self.recorder.add(**frame)
                self.kept += 1
        self.pending.clear()
        self.recorder.flush(end_of_segment=True)
        print(f"  CrashFilter Ende: behalten {self.kept} Frames, verworfen {self.discarded} Frames, "
              f"{self.crashes} Crashes erkannt.")


CMD_DT_SAFE = 1.0 / 30.0  # #SAFE: nur für Ausgaben, wird in controlVehicleThread durch CMD_DT überschrieben


def controlVehicleThread(d: dict, vehicleColor: str, delta_t: float, shm_front_pointer=None,
                         shm_front_counter=None, cnn_model_path=None, record=False):
    """#VORSAGE: Steuerprozess mit Datenaufnahme.

    record=False          : normales Fahren mit dem Lidar-Controller (für weitere Fahrzeuge ohne Aufnahme)
    record=True, collect  : Lidar-Lehrer fährt (+ Lenkrauschen), Bild + Lehrer-Befehl werden gespeichert
    record=True, dagger   : Vorhersage-CNN fährt (mit Wahrscheinlichkeit DAGGER_BETA der Lehrer), Lehrer beschriftet
    """
    print(f"Starting controlVehicleThread for the {vehicleColor} vehicle (record={record}).")

    # #VORSAGE: Hilfsfunktionen und Konstanten des Vorhersage-Modells (gleiche Vorverarbeitung/Normierung wie beim Training)
    from cnn_controller_vorhersage import preprocess, CMD_DT, MAX_STEER_RAD, MAX_SPEED_MPS
    global CMD_DT_SAFE
    CMD_DT_SAFE = CMD_DT

    cnn = None
    front_buffers = None
    recorder = None
    crash_filter = None    # #SAFE
    crash_detector = None  # #SAFE
    if record:
        if CNN_DRIVES:
            from cnn_controller_vorhersage import CNNDriverVorhersage
            cnn = CNNDriverVorhersage(cnn_model_path, steering_smoothing=CNN_STEERING_SMOOTHING)
            print(f"{vehicleColor}: {COLLECT_MODE} mit {cnn_model_path} ({cnn.n} Schritte Verzögerung).")
        shm_front = SharedMemory(name="front")  # double buffer written by receive_frontcam_image_thread
        image_bytes = math.prod(frontcam_image_shape) * np.dtype(np.uint8).itemsize
        front_buffers = [np.ndarray(frontcam_image_shape, dtype=np.uint8, buffer=shm_front.buf, offset=0),
                         np.ndarray(frontcam_image_shape, dtype=np.uint8, buffer=shm_front.buf, offset=image_bytes)]

        # #VORSAGE: Verzögerung Bild -> Ausführung in Simulationsschritten (1/30 s), wird mit jeder Datei gespeichert
        real_delay_s = REAL_DELAY_S if REAL_DELAY_S is not None else vehicles[vehicleColor]["delay_ms"] / 1000.0
        delay_steps = int(round(real_delay_s / CMD_DT))
        recorder = EpisodeRecorder(DATA_DIR, COLLECT_MODE + "_safe", vehicleColor, delay_steps, SAVE_EVERY_FRAMES)
        # #SAFE: Crash-Filter vor den Recorder schalten
        crash_filter = CrashFilterRecorder(recorder,
                                           discard_frames=DISCARD_BEFORE_CRASH_FRAMES,
                                           recovery_frames=int(round(CRASH_RECOVERY_S / CMD_DT)))
        print(f"{vehicleColor}: Crash-Filter aktiv: {DISCARD_BEFORE_CRASH_FRAMES} Frames vor jedem Crash werden verworfen, "
              f"Neustart der Aufnahme nach {CRASH_RECOVERY_S:.1f} s ohne Crash.")
        if not SAVE_DATA:
            print(f"{vehicleColor}: Modus 'drive' -> das CNN fährt allein, es werden KEINE Daten gespeichert.")
        print(f"{vehicleColor}: Aufnahme '{COLLECT_MODE}' nach {DATA_DIR if SAVE_DATA else '(nichts)'}/, Verzögerung {delay_steps} Schritte "
              f"({delay_steps * CMD_DT * 1000:.0f} ms), Lenkrauschen {NOISE_DEG if COLLECT_MODE == 'collect' else 0} Grad")

    # #VORSAGE: beim Aufnehmen läuft die Schleife mit 30 Hz, damit ein Frame = ein Schritt wie in der Simulation (CMD_DT)
    target_control_frequency = int(round(1.0 / CMD_DT)) if record else 40
    loop_time_s = 1.0 / target_control_frequency
    current_motor_value = 0
    pid_control_motor : PIDController = None  #we need to store the motor pid in the process as our copy of the vehicle it not synchronized back to the other processes.
    pid_control_steer : PIDController = None  #we need to store the motor pid in the process as our copy of the vehicle it not synchronized back to the other processes.
    command_history = None

    #default end-to-end-delay (will be overwritten by vehicle-specific delay later)
    end_to_end_delay_s = 0.15 #0.15

    smooth_start_time = time.time()
    last_image_counter = 0   # #VORSAGE: Nummer des zuletzt gesehenen Frontkamera-Bildes
    overruns = 0             # #VORSAGE: wie oft die Schleife die 30 Hz nicht geschafft hat
    last_no_image_warning = 0.0  # #VORSAGE: Zeitpunkt der letzten "kein Frontkamera-Bild"-Warnung
    tracking_lost_since = None   # #SAFE: Zeitpunkt, seit dem das Tracking weg ist (None = Tracking ok)
    tracking_loss_crashed = False  # #SAFE: wurde der aktuelle Aussetzer schon als Crash gewertet?

    try:  # #VORSAGE: bei Abbruch (Ctrl+C / Fehler) die letzten Frames noch speichern
        while d["raceEnabled"]:
            start_time = time.time()

            #look for the vehicle with our color and our opponents:
            opponents : list[Vehicle] = []
            v         : Vehicle       = None

            for x in d["vehicles"]:
                if x.color == vehicleColor:
                    v = x
                else:
                    opponents.append(x)

            if v is None:
                d["lidar_rays_" + vehicleColor] = None
                d["setpoints_" + vehicleColor]  = None
                # #SAFE: Tracking verloren. Erst nach TRACKING_LOSS_TOLERANCE_S als Crash werten (letzte 30 Frames
                #        verwerfen). Kürzere Aussetzer werden beim Wiederfinden als Lücke behandelt (siehe unten).
                now = time.time()
                if tracking_lost_since is None:
                    tracking_lost_since = now
                if now - tracking_lost_since > TRACKING_LOSS_TOLERANCE_S:
                    if crash_filter is not None and DISCARD_ON_TRACKING_LOSS:
                        crash_filter.crash(f"Tracking länger als {TRACKING_LOSS_TOLERANCE_S:.1f} s verloren")
                    if not tracking_loss_crashed and cnn is not None:
                        cnn.reset()
                    tracking_loss_crashed = True
                time.sleep(0.002)  # #SAFE: nicht mit 100 % CPU auf das Tracking warten
                continue

            if tracking_lost_since is not None:
                # #SAFE: Tracking wieder da. Kurzer Aussetzer (oder Crash-Wertung abgeschaltet) -> Daten behalten,
                #        aber an der Lücke eine neue Datei beginnen. Langer Aussetzer -> crash() hat schon alles erledigt.
                if crash_filter is not None and (not tracking_loss_crashed or not DISCARD_ON_TRACKING_LOSS):
                    crash_filter.gap()
                    print(f"  Tracking-Aussetzer {time.time() - tracking_lost_since:.2f} s -> kein Crash, Daten behalten.")
                if crash_detector is not None:
                    crash_detector.reset()  # sonst würde die Bewegung während der Lücke als Positionssprung gelten
                tracking_lost_since = None
                tracking_loss_crashed = False

            #set end-to-end delay from config (vehicle dependent, since hw revisions may be different)
            end_to_end_delay_s = v.delay_ms / 1000.0

            if record and crash_detector is None:  # #SAFE: Strecke kommt über den Lidar-Sensor des Fahrzeugs
                crash_detector = CrashDetector(v.lidar.track)

            if pid_control_motor is None:
                pid_control_motor = v.motor_pid
                pid_control_steer = v.steering_pid
            else:
                v.motor_pid = pid_control_motor #put in our local copy of the pid controller.
                v.steering_pid = pid_control_steer

            if command_history is None:
                command_history = v.command_history #put in our local copy of the command history
            else:
                v.command_history = command_history

            v.command_frequency_hz = target_control_frequency

            if not record:
                # unverändert: Lidar-Controller fährt
                target_velocity_mps, target_steering_angle_rad, rays, setpoint, emer_brake = v.compute_next_command(delta_t=end_to_end_delay_s, opponents=opponents)
            else:
                # #VORSAGE: aktuelles Frontkamera-Bild holen
                front_frame = front_buffers[shm_front_pointer.value].copy()
                image_counter = shm_front_counter.value
                have_image = image_counter > 0
                new_image = image_counter != last_image_counter
                last_image_counter = image_counter
                if not have_image and start_time - last_no_image_warning > 5.0:
                    # #VORSAGE: ohne Frontkamera-Bild wird NICHTS gespeichert -> deutlich melden
                    print(f"WARNUNG {vehicleColor}: noch kein Frontkamera-Bild empfangen (UDP-Port 9999) "
                          f"-> es wird nichts aufgenommen.")
                    last_no_image_warning = start_time

                # #VORSAGE: LEHRER-BESCHRIFTUNG. Der Lidar-Controller bewertet die AKTUELLE Position (ohne Vorhersage,
                #           LABEL_DELTA_T), genau wie in der Simulation. Das "in die Zukunft schauen" passiert beim Training
                #           durch das Verschieben der Beschriftung um n Schritte. Aufruf auf einer Kopie, weil
                #           compute_next_command interne Filterwerte des Fahrzeugs verändert.
                label_speed, label_steer, rays, setpoint, _ = copy.copy(v).compute_next_command(delta_t=LABEL_DELTA_T, opponents=opponents)
                label_steer = float(np.clip(label_steer, -MAX_STEER_RAD, MAX_STEER_RAD))

                # #VORSAGE: welcher Befehl wird wirklich gefahren?
                if CNN_DRIVES and have_image and (COLLECT_MODE == "drive" or random.random() >= DAGGER_BETA):
                    target_velocity_mps, target_steering_angle_rad = cnn.predict(front_frame, start_time)  # CNN fährt
                elif CNN_DRIVES and not have_image:
                    target_velocity_mps, target_steering_angle_rad = 0.0, 0.0   # noch kein Bild -> stehen bleiben
                else:
                    # Lehrer fährt, MIT Verzögerungsausgleich wie in main.py (sonst fährt er auf der echten Strecke schlechter)
                    target_velocity_mps, target_steering_angle_rad, _, _, _ = copy.copy(v).compute_next_command(delta_t=end_to_end_delay_s, opponents=opponents)
                    if COLLECT_MODE == "collect" and NOISE_DEG > 0:
                        # Lenkrauschen nur auf den AUSGEFÜHRTEN Befehl, die Beschriftung bleibt sauber -> Rückführ-Situationen
                        target_steering_angle_rad += math.radians(np.random.normal(0.0, NOISE_DEG))
                emer_brake = False


            #limit steering angles due to vehicle configuration
            target_steering_angle_rad = max(math.radians(v.min_steering_angle_deg),
                                            min(math.radians(v.max_steering_angle_deg),
                                                             target_steering_angle_rad)
                                        )
            sent_steering_angle_rad = target_steering_angle_rad  # #VORSAGE: Lenkwinkel VOR der Invertierung (wie im Training)

            #reverse steering if the vehicle is configured to have inverted steering
            if v.steering_inverted:
                target_steering_angle_rad *= -1

            #emergency brakeing disabled. #TODO: implement emergency brake in a safe way, so that the vehicle does not crash into the wall.
            emer_brake = False

            #start smoothly as not to stress the gearbox too much. This is only done for the first second after the start of the race.
            if time.time() - smooth_start_time < 1.0:
                target_velocity_mps = 0.6

            if record:
                # #VORSAGE: dem CNN (dagger) den wirklich gesendeten Befehl melden
                if cnn is not None:
                    cnn.register_sent(target_velocity_mps, sent_steering_angle_rad, start_time)

                # #VORSAGE: Frame speichern: Bild + Lehrer-Beschriftung + gesendeter Befehl (+ Infos zur Analyse).
                #           Jede Schleife (= 1/30 s) ein Frame, auch wenn das Bild noch dasselbe ist (new_image = 0),
                #           damit der Zeitabstand konstant bleibt, so wie es train_cnn_vorhersage.py erwartet.
                # #SAFE: Crash prüfen. Bei Crash wird dieser Frame nicht gespeichert und der Puffer verworfen.
                crash_reason = crash_detector.check(v, target_velocity_mps, start_time,
                                                    startup=time.time() - smooth_start_time < CRASH_STARTUP_IGNORE_S)
                if crash_reason is not None:
                    crash_filter.crash(crash_reason)
                elif have_image:
                    crash_filter.add(x=preprocess(front_frame),
                                     steer=label_steer / MAX_STEER_RAD,
                                     speed=float(np.clip(label_speed / MAX_SPEED_MPS, 0.0, 1.0)),
                                     executed_steer_rad=float(sent_steering_angle_rad),
                                     sent_speed_mps=float(target_velocity_mps),
                                     actual_speed_mps=float(v.getSpeed()),
                                     pos_x=float(v.x), pos_y=float(v.y), yaw=float(v.yaw),
                                     t=start_time, new_image=float(new_image))

            #send actions to vehicle
            current_motor_value = v.sendControlsToHardware(target_velocity_mps=target_velocity_mps,
                                                           target_steering_angle_rad=target_steering_angle_rad,
                                                           current_motor_value=current_motor_value,
                                                           emergency_brake=emer_brake)

            #forward lidar information to visualization process through the dict.
            d["lidar_rays_" + vehicleColor]  = rays
            d["setpoints_" +  vehicleColor]  = setpoint
            d["target_velocity_delta_mps_" +  vehicleColor]  = target_velocity_mps - v.getSpeed()
            d["motor_voltage_" +  vehicleColor]  = current_motor_value
            d["target_velocity_mps"] = target_velocity_mps
            d["target_steering_angle_rad"] = target_steering_angle_rad


            end_time = time.time()
            dt = end_time - start_time
            sleep_time = loop_time_s - dt

            if sleep_time > 0:
                time.sleep(sleep_time)
            elif record:
                # #VORSAGE: Schleife zu langsam -> Zeitabstände stimmen nicht mehr ganz. Gelegentlich ist ok.
                overruns += 1
                if overruns % 30 == 1:
                    print(f"WARNUNG {vehicleColor}: Schleife braucht {dt * 1000:.0f} ms > {loop_time_s * 1000:.0f} ms "
                          f"({overruns}x)")

            d["control_loop_frequency_" +  vehicleColor] = 1.0/(dt+max(sleep_time, 0.0))
    finally:
        if crash_filter is not None:
            crash_filter.finish()  # #SAFE: Rest speichern (ohne die letzten 30 Frames, siehe DISCARD_PENDING_AT_END)

def receive_frontcam_image_thread(shm_name: str, shm_front_pointer, shm_front_counter=None):  # #VORSAGE: + Bildzähler
    # Shared memory to store latest frontcam picture
    shm = SharedMemory(name=shm_name)

    # Shared momory is build as a double buffer, so that the visualization process can read from one buffer while this process writes to the other buffer.
    shm_buffers = [np.ndarray(frontcam_image_shape, dtype=np.uint8, buffer=shm.buf, offset=0),
                   np.ndarray(frontcam_image_shape, dtype=np.uint8, buffer=shm.buf, offset=math.prod(frontcam_image_shape) * np.dtype(np.uint8).itemsize)]

    # Create a UDP socket to receive live vieo stream from vehicle. The ESP32-CAM will send the video stream to this socket.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 9999))
    #Increase receive buffer size
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 524288)


    # tell esp32-cam to send stream to this IP. As we don't know the IP of the esp32-cam, we will broadcast a magic string to the local network. 
    # The esp32-cam will listen for this magic string and start sending the video stream to this IP and port.

    # Create a UDP socket
    broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Enable broadcasting mode on the socket
    broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    # Target: Local broadcast IP and port 9999
    broadcast_address = ('255.255.255.255', 9999)
    message = b"startStream"

    # Send the packet
    broadcast_sock.sendto(message, broadcast_address)
    print(f"Broadcasted '{message.decode()}' to {broadcast_address}")

    #the esp32 should now start sending the video stream to this computers IP on port 9999.


    #
    #   receive the video stream from the esp32-cam and write it to shared memory
    #

    

    print("Waiting for data from esp32cam...")

    last_shown_image_id = 0
    frame_buffer = {}

    while True:
        data, addr = sock.recvfrom(2000) #receive next udp packet up to 2000 bytes in size (packets should actually be smaller)
        if len(data) < 5:
            continue

        #print(f"Received packet of size {len(data)} bytes from {addr}")
            
        # parse packet header
        pkt_id = data[0] + (data[1] << 8)
        total_pkts = data[2] + (data[3] << 8)
        image_id = data[4]
        payload = data[5:]

        # we need to make sure we are robust against lost packets and out-of-order packets:
        # once we completyle received an image that we did not display so far, display the image and delete all buffers for frames with lower ids.
        

        #create framebuffer for image_id, if not already happend
        if not image_id in frame_buffer:
            frame_buffer[image_id] = {}

        #put image in framebuffer
        frame_buffer[image_id][pkt_id] = payload
        
        #show image as soon as the framebuffer is full
        if len(frame_buffer[image_id]) == total_pkts:
            buffer = bytearray()
            for i in range(total_pkts):
                try:
                    buffer.extend(frame_buffer[image_id][i])
                except:
                    print(f"Error occurred while extending buffer for image {image_id}, packet {i}")
                    print(list(frame_buffer[image_id].keys()))
                    print("?")

                
            frame_buffer[image_id].clear() #image shown --> clear buffer.
            last_shown_image_id = image_id

            #clear all buffers for old images:
            older_ids = [i % 256 for i in range(last_shown_image_id-125, last_shown_image_id)]
            for id in older_ids:
                if id in frame_buffer.keys():
                    frame_buffer[id].clear()
            
            #decode JPEG:
            np_arr = np.frombuffer(buffer, dtype=np.uint8)
            frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)

            if frame is not None:
                #flip frame vertically, because the camera is mounted upside down
                frame = cv.flip(frame, 0)

                #write to shared memory which is a double buffer. We use a pointer to the current buffer to avoid race conditions.
                current_buffer_index = shm_front_pointer.value
                next_buffer_index = 1 - current_buffer_index  # switch to the other buffer
                
                
                # Create a NumPy array pointing directly to the shared memory buffer
                shm_buffers[next_buffer_index][:] = frame[:]    # store image in shared memory
                shm_front_pointer.value = next_buffer_index
                if shm_front_counter is not None:
                    shm_front_counter.value += 1  # #VORSAGE: neues Bild angekommen



            
def main():

    #setup camera
    cam = Camera(vertical_resolution_px = vertical_resolution_px,
                 horizontal_resolution_px = horizontal_resolution_px,
                 frames_per_seconds = frames_per_seconds,
                 exposure_time_absolute = camera_exposure_time_absolute,
                 opening_angle_vertical_degrees = opening_angle_vertical_degrees,
                 opening_angle_horizontal_degrees = opening_angle_horizontal_degrees,
                 meters_to_pixels = meters_to_pixels,
                 max_speed_vehicle_mps = max_speed_vehicle_mps,
                 minimum_brightness = minimum_brightness,
                 threshold_brightness_of_black = threshold_brightness_of_black,
                 threshold_brightness_of_white = threshold_brightness_of_white,
                 circle_diameter_meters = circle_diameter_meters,
                 size_between_black_and_white_center_meters = size_between_black_and_white_center_meters,
                 height_over_ground_black_meters = height_over_ground_black_meters,
                 height_over_ground_white_meters = height_over_ground_white_meters,
                 cameraMatrix = cameraMatrix,
                 distortionCoefficients = distortionCoefficients)



    #calibrate (physical) setup of the camera by giving visual guidance
    print("Do you want to calibrate the physical setup of the camera (y/n)?")
    key = input()
    if key.upper() == "Y":
        cam.cameraCalibrationWizard()

    #calibrate track
    print("Do you want to re-set the track boundaries (y/n)?")
    key = input()
    racetrack = Track()
    if key.upper() == "Y":
        print("Do you want to edit the old track boundaries (y/n)?")
        key = input()
        if key.upper() == "Y":
            racetrack.loadFromFile("track_borders.npy")
            cam.racetrack = racetrack

        old = cam.minimum_brightness
        cam.minimum_brightness = 2
        cam.increaseExposureTime()
        racetrack.manuallyPickTrackBorders(cam.get_frame(initializeColorCorrection=True))
        cam.resetExposeToDefault()

        cam.minimum_brightness = old
        cam.get_frame(initializeColorCorrection=True)

        racetrack.saveToFile("track_borders.npy")
        print("Track saved to file.")
    else:
        racetrack.loadFromFile("track_borders.npy")
    
    cam.racetrack = racetrack

    print("Do you want to write the camera stream as video to disk for debugging (y/n)?")
    key = input()
    if key.upper() == "Y":
        cam.setup_debug_video()

    # ===== Init Frontcam ====
    enable_frontcam = False
    show_frontcam = False

    if USE_CNN_CONTROLLER:
        print("CNN controller enabled -> frontcam is required and will be used.")
        key = "Y"
    else:
        print("Do you want to use the frontcam (y/n)?")
        key = input()
    if key.upper() == "Y":
        enable_frontcam = True

        print("Do you want to see the frontcam stream (y/n)?")
        key = input()
        if key.upper() == "Y":
            show_frontcam = True


    # create manager object for multiprocessing - used for inter-process communication
    # all processes can read and write to this dict.
    manager = Manager()
    d = manager.dict()
    d["vehicles"] = []
    d["showVisualization"] = True
    d["raceEnabled"] = True

    #create shared memory for the last image of the camera (because its faster than the manager!)
    import os
    os.system("rm -rf /dev/shm/frame > /dev/null")
    os.system("rm -rf /dev/shm/front > /dev/null")
    last_frame = cam.get_frame()
    print(last_frame.shape)
    shm = SharedMemory(name="frame", create=True, size=last_frame.nbytes)
    shared_memory_buffer = np.ndarray(last_frame.shape, dtype=np.uint8, buffer=shm.buf)

    shm_front_pointer = None  # Initialize the shared memory pointer for frontcam
    shm_front_counter = None  # #VORSAGE: Bildzähler (wird mit der Frontkamera angelegt)
    frontcam_process = None  # Initialize the frontcam process variable
    shm_front = None  # Initialize the shared memory for frontcam variable

    # shared memory for last frontcam image
    if enable_frontcam:

        #this shared momory contains TWO images for double buffering!
        shm_front = SharedMemory(
            name="front",
            create=True,
            size=math.prod(frontcam_image_shape) * np.dtype(np.uint8).itemsize * 2
        )
        shm_front_pointer = Value('i', 0)   #points to the current image in the double buffer (0 or 1)
        shm_front_counter = Value('i', 0)   # #VORSAGE: zählt die empfangenen Bilder (erkennt neue Bilder)

        # Start the process which receives the frontcam images and writes them to shared memory for display in the visualization process.
        frontcam_process = Process(
            target=receive_frontcam_image_thread,
            args=("front", shm_front_pointer, shm_front_counter)  # #VORSAGE: + Bildzähler
        )

        frontcam_process.start()


    #spawn one process for visualization:
    process_visualization = Process(target=showImageThread, args=(d, racetrack, shm_front_pointer if show_frontcam else None))
    process_visualization.start()

    #list of processes - one for each vehicle to compute steering commands
    control_processes = {}

    while cam.is_video_stream_active():
        if cam.detectVehicles() > 0:
            print(f"found {len(cam.tracked_vehicles)} vehicles.")

            #setup vehicles
            for v in cam.tracked_vehicles:
                if v.color in vehicles.keys():
                    #set physical properties
                    v.setPhysicalProperties(vehicles[v.color]["length_m"], 
                                            vehicles[v.color]["width_m"],
                                            vehicles[v.color]["rear_axle_offset_m"], 
                                            vehicles[v.color]["steering_measurements"], 
                                            vehicles[v.color]["min_motor_value"],
                                            vehicles[v.color]["max_motor_value"],
                                            vehicles[v.color]["speed_factor"],
                                            vehicles[v.color]["min_steering_angle"],
                                            vehicles[v.color]["max_steering_angle"],
                                            vehicles[v.color]["steering_inverted"],
                                            vehicles[v.color]["height_over_ground_colored_marker_m"],
                                            vehicles[v.color]["height_over_ground_white_marker_m"],
                                            vehicles[v.color]["delay_ms"]                                            
                                            )
                    #setup IP communication
                    v.initNetworkConnection(ip_adress=vehicles[v.color]["ip"], port=vehicles[v.color]["port"])

                    #setup motor parameters
                    v.initMotorPID(vehicles[v.color]["motor_pid"][0],
                                   vehicles[v.color]["motor_pid"][1],
                                   vehicles[v.color]["motor_pid"][2],
                                   error_history_length=50)
                    #setup steering parameters
                    v.initSteeringPID(vehicles[v.color]["steering_pid"][0],
                                   vehicles[v.color]["steering_pid"][1],
                                   vehicles[v.color]["steering_pid"][2],
                                   error_history_length=50)
                    
                    #setup controller
                    v.controller = DisparityExtender(car_width=(vehicles[v.color]["width_m"])*meters_to_pixels, 
                                                     disparity_threshold=50, 
                                                     tolerance=20)
                    
                    #setup lidar
                    v.lidar = LidarSensor(track=racetrack, 
                                          field_of_view_deg=vehicles[v.color]["lidar_field_of_view_deg"], 
                                          numRays=vehicles[v.color]["lidar_numRays"], 
                                          rayLength_px=vehicles[v.color]["lidar_rayLength_m"] * meters_to_pixels)
                

                    #create process for controlling the vehice with color v.color:
                    # #SAFE: KEINE Duplikate. Nach einem Tracking-Verlust wird das Fahrzeug hier neu erkannt. Läuft für
                    #        diese Farbe noch ein Steuerprozess, übernimmt DER das Fahrzeug wieder (er liest es aus
                    #        d["vehicles"]). Vorher wurde jedes Mal ein weiterer Prozess gestartet -> mehrere Prozesse
                    #        steuerten dasselbe Auto und schrieben doppelte Dateien.
                    if v.color in control_processes and control_processes[v.color].is_alive():
                        print(f"Fahrzeug '{v.color}' wieder erkannt -> vorhandener Steuerprozess übernimmt (kein Duplikat).")
                        continue
                    # #VORSAGE: nur COLLECT_VEHICLE_COLOR nimmt auf (und bekommt die Frontkamera), andere fahren normal mit Lidar
                    record_this = (v.color == COLLECT_VEHICLE_COLOR) and shm_front_pointer is not None
                    if not record_this:  # #VORSAGE: deutlich melden, wenn dieses Fahrzeug NICHT aufgenommen wird
                        print(f"HINWEIS: Fahrzeug '{v.color}' fährt ohne Aufnahme "
                              f"(aufgenommen wird nur COLLECT_VEHICLE_COLOR = '{COLLECT_VEHICLE_COLOR}').")
                    control_processes[v.color] = Process(target=controlVehicleThread,
                                                         args=(d, v.color, 1.0/frames_per_seconds,
                                                               shm_front_pointer if record_this else None,
                                                               shm_front_counter if record_this else None,
                                                               CNN_MODEL_PATH if CNN_DRIVES else None,
                                                               record_this))
                    control_processes[v.color].start()
                else:
                    print("Could not properly detect color of vehicle!")

            #tracking loop
            delta_t = 1.0/frames_per_seconds
            while len(cam.tracked_vehicles) > 0:
                #update position of each vehicle
                cam.trackVehicles()

                cam.checkFinishLine(top_left=(1000, 950), bottom_right=(1050, 1120))

                #send frame and currently tracked vehicles to other processes
                shared_memory_buffer[:] = cam.get_last_frame()[:]
                d["vehicles"] = [copy.copy(v) for v in cam.tracked_vehicles]


            print("Tracking of Vehicle lost!!!!")
        else:
            print("Could not detect any vehicle - restarting.")
            d["vehicles"] = []

    #wait for all threads to end.
    process_visualization.join()
    for k in control_processes.keys():
        control_processes[k].join()

    if enable_frontcam:
        if frontcam_process is not None:
            frontcam_process.join()

        if shm_front is not None:
            shm_front.close()
            shm_front.unlink()
    
    shm.close()
    shm.unlink()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting...")
    finally:
        # stop all video streams from the vehicles

        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Enable broadcasting mode on the socket
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        # Target: Local broadcast IP and port 9999
        # (255.255.255.255 targets the local network layout)
        broadcast_address = ('255.255.255.255', 9999)
        message = b"stopStream"

        # Send the packet
        sock.sendto(message, broadcast_address)
