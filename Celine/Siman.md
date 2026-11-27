#SIMAN

Fast alle Stellen liegen in tools/simulation.py, in der Funktion nextStep(). Ich habe nichts geändert. Hier die Stellen, nach Wichtigkeit sortiert.

1. Geschwindigkeit springt sofort auf den Wert (keine Massenträgheit)
simulation.py:222


v.vehicle_speed = current_motor_value / 100.0
Der Motorwert wird direkt in eine Geschwindigkeit umgerechnet, im selben Zeitschritt. Ändert sich der Motorwert von 60 auf 110, ist das Auto sofort 0,5 m/s schneller. Im Code steht dort selbst schon ein TODO dazu.

Was man stattdessen machen könnte:

Verzögerung 1. Ordnung (einfach): Der Motorwert bestimmt nur eine Zielgeschwindigkeit, und die echte Geschwindigkeit nähert sich ihr langsam an: v_ziel = f(motor) und v += (v_ziel − v) · dt / τ. τ ist die Zeitkonstante, also wie träge das Auto reagiert, etwa 0,2–0,5 s.
Mit Kräften (genauer): m · a = F_motor(motor, v) − F_roll − F_luft.
F_motor ist die Antriebskraft. Sie wird kleiner, je schneller der Motor dreht (Gegenspannung).
F_roll ist der Rollwiderstand, eine ungefähr konstante Bremskraft.
F_luft ist der Luftwiderstand, wächst mit v² und spielt bei dem kleinen Auto kaum eine Rolle.
Bremsen und Ausrollen: Bei Motorwert 0 bleibt das Auto in der Simulation sofort stehen. In echt rollt es aus.
Anfahrschwelle: In echt bewegt sich das Auto erst ab etwa Motorwert 60 (min_motor_value). Darunter sollte die Geschwindigkeit 0 sein.
Kennlinie f(motor): Sie ist in der Realität nicht linear und hängt vom Akkustand ab. Man kann sie messen, indem man das echte Auto mit festen Motorwerten fahren lässt und die Deckenkamera die Geschwindigkeit misst.
2. Lenkung ist sofort am Zielwinkel
simulation.py:229 verwendet target_steering_angle_rad direkt. Ein echter Servo dreht sich aber nur mit begrenzter Geschwindigkeit, ungefähr 0,1–0,2 s für 60°.

Drehgeschwindigkeit begrenzen: δ += clip(δ_ziel − δ, −ω_max·dt, +ω_max·dt).
Totzone, Spiel und Versatz: Die Messwerte in steering_measurements (vehicle_config.py:15) zeigen, dass bei Servo-Wert 90 (geradeaus) tatsächlich etwa 5° herauskommen. Diesen Versatz und die Nichtlinearität könnte man auch in der Simulation nachbilden.
3. Keine Verzögerung zwischen Bild und Ausführung
simulation.py:195 Hier steht ebenfalls ein TODO. Auf der echten Strecke vergehen beim gelben Auto ungefähr 120 ms zwischen Kamerabild und Reaktion des Autos (delay_ms in der Config). Dazu kommen beim CNN noch die WLAN-Übertragung der Frontkamera und die Rechenzeit des Modells.

Umsetzung: Eine Warteschlange (collections.deque), aus der ein Befehl erst nach delay / dt Schritten ausgeführt wird. Bei 30 fps und 120 ms sind das etwa 4 Frames.
Für das CNN ist das wahrscheinlich der wichtigste Punkt. Ohne Verzögerung lernt das Netz, zu spät zu reagieren, und fängt in echt an zu schlingern.
4. Fahrdynamik: kein Rutschen, kein Unter- oder Übersteuern
simulation.py:229-240 Die Simulation rechnet mit einem kinematischen Einspurmodell: Das Auto fährt immer exakt in die Richtung, in die es lenkt. Im Code steht dazu schon die Notiz „never experience under- or oversteering“.

Einfach: Ab einer bestimmten Querbeschleunigung a_quer = v²·κ (κ = Krümmung der Kurve) wird der Lenkeffekt abgeschwächt. Das bildet Untersteuern nach.
Aufwendiger: Ein dynamisches Einspurmodell mit Schräglaufwinkeln, Reifensteifigkeit und Trägheitsmoment um die Hochachse.
Bezugspunkt der Drehung: Momentan dreht sich das Auto um seine Mitte. Bei einem echten Auto liegt der Drehpunkt an der Hinterachse (rear_axle_offset_m steht schon in der Config).
5. Rundungsfehler bei der Position (versteckter Fehler)
simulation.py:242 setzt v.x = int(x).

int() schneidet die Nachkommastellen ab, und zwar jeden Schritt. Bei 1 m/s und 30 fps bewegt sich das Auto etwa 23 px pro Schritt, dabei gehen jedes Mal bis zu 1 px verloren. Das sind ungefähr 4 % systematischer Fehler.
Der Fehler hängt außerdem von der Fahrtrichtung ab: Nach rechts oder unten fährt das Auto zu kurz, nach links oder oben zu weit.
Besser: Die Position intern als float speichern und nur zum Zeichnen runden.
6. Störungen und Rauschen
In der Simulation läuft alles perfekt. Realistischer wird es, wenn man zufällige Störungen einbaut, zum Beispiel:

schwankende Geschwindigkeit, etwa durch einen leerer werdenden Akku (Motorwert mit einem Faktor um 1 multiplizieren)
leichte, zufällige Abweichung beim Lenken
Bodenunebenheiten
Zusätzlich hilft Rauschen in den Messwerten: Die Geschwindigkeit, die der PID-Regler sieht, kommt auf der echten Strecke aus der Deckenkamera und ist dort verrauscht.

7. Kamerabild (wichtig für das CNN)
Das Bild wird in simulation.py:183 gerendert. Der Renderer selbst liegt in renderer3d.py. Er erzeugt perfekte Bilder. Die echte ESP32-Kamera hat dagegen:

JPEG-Artefakte: In der Simulation lässt sich das mit cv.imencode und anschließendem imdecode bei niedriger Qualität nachbilden.
Bewegungsunschärfe, Bildrauschen, schwankende Helligkeit und Farben.
Linsenverzerrung: Der Renderer kann das schon (egocam_distortion), es ist aber nicht eingeschaltet.
Wackelnde Kamera: Die Kameraneigung (Pitch) und die Höhe schwanken beim Fahren und Bremsen.
Ausfälle: Einzelne Frames gehen über WLAN verloren, oder dasselbe Bild kommt doppelt an.
8. Zeittakt
Die Simulation läuft mit 30 fps (simulation_singlethreaded.py:31).
Der echte Regelkreis in main.py läuft mit 40 Hz, und die Frontkamera liefert ihre Bilder unregelmäßig.
Schwankendes dt: Man könnte in der Simulation die Zeit zwischen den Schritten leicht zufällig variieren lassen, so wie es in der Realität passiert.


Umsetzung:
Kannst Verzögerung und Trägheit von Motor und Lenkung einfügen und an den Stellen wo dur etwas änders, ein #SIMAN als Kommentar setzen und auf deustsch kurz erklären was die Änderung macht
Bash
IN
cd "c:/Users/celin/Documents/Studium/WISE2627/AIT/AIT_2026_MaschineLearning/f1_28th_ait2026_basecode-main" && grep -n "self.vehicle_speed\s*=\|self.yaw_rate\s*=\|def __init__" localization/vehicle.py | head; grep -rn "Simulation(" --include=*.py . | grep -v "\.venv"

OUT
18:    def __init__(self, x, y, radius):
40:    def __init__(self, x, y, yaw, meters_to_pixels, racetrack, front_marker:Circle|None = None, rear_marker:Circle|None = None):
45:        self.yaw_rate = 0.0
101:        self.vehicle_speed = 0 #meters per second; absolute speed
426:        self.vehicle_speed = sum(self.speed_filter_values) / len(self.speed_filter_values) #average
429:        self.yaw_rate = self.yaw_rate if dt == 0.0 else (self.yaw - old_yaw)/dt
./simulation_cnn.py:154:    sim = Simulation(track_border_filename="track_borders_simulation.npy",
./simulation_singlethreaded.py:67:    sim = Simulation(track_border_filename="track_borders_simulation.npy",
Die neuen Parameter werden mit Standardwerten eingefügt, sodass simulation_cnn.py und simulation_singlethreaded.py ohne Änderungen weiterlaufen. Als Nächstes prüfe ich die Simulation.
Edit c:\Users\celin\Documents\Studium\WISE2627\AIT\AIT_2026_MaschineLearning\f1_28th_ait2026_basecode-main\tools\simulation.py
Added 1 line
Edit c:\Users\celin\Documents\Studium\WISE2627\AIT\AIT_2026_MaschineLearning\f1_28th_ait2026_basecode-main\tools\simulation.py
Added 3 lines
Edit c:\Users\celin\Documents\Studium\WISE2627\AIT\AIT_2026_MaschineLearning\f1_28th_ait2026_basecode-main\tools\simulation.py
Added 10 lines
Edit c:\Users\celin\Documents\Studium\WISE2627\AIT\AIT_2026_MaschineLearning\f1_28th_ait2026_basecode-main\tools\simulation.py
Added 5 lines
Edit c:\Users\celin\Documents\Studium\WISE2627\AIT\AIT_2026_MaschineLearning\f1_28th_ait2026_basecode-main\tools\simulation.py
Added 6 lines
Edit c:\Users\celin\Documents\Studium\WISE2627\AIT\AIT_2026_MaschineLearning\f1_28th_ait2026_basecode-main\tools\simulation.py
Added 17 lines
11:20 AM
Jetzt teste ich das mit einer kurzen Fahrt ohne Fenster. Die Daten schreibe ich in den Scratchpad-Ordner, damit cnn_data unverändert bleibt.



Bash
IN
cd "c:/Users/celin/Documents/Studium/WISE2627/AIT/AIT_2026_MaschineLearning/f1_28th_ait2026_basecode-main" && python -m py_compile tools/simulation.py && echo COMPILE_OK && timeout 300 python simulation_cnn.py --mode collect --episodes 2 --headless --time-limit 20 --seed 1 --data-dir "C:/Users/celin/AppData/Local/Temp/claude/46532ef1a24ecd12/c3929237-9927-44d3-8e3e-8c55c3c69963/scratchpad/simtest" 2>&1 | tail -8

OUT
COMPILE_OK
Traceback (most recent call last):
  File "C:\Users\celin\Documents\Studium\WISE2627\AIT\AIT_2026_MaschineLearning\f1_28th_ait2026_basecode-main\simulation_cnn.py", line 26, in <module>
    import cv2 as cv
ModuleNotFoundError: No module named 'cv2'
Bash
IN
cd "c:/Users/celin/Documents/Studium/WISE2627/AIT/AIT_2026_MaschineLearning/f1_28th_ait2026_basecode-main" && PY="c:/Users/celin/Documents/Studium/WISE2627/AIT/f1_28th_ait2026_basecode-main/.venv/Scripts/python.exe"; ls "$PY" && timeout 300 "$PY" simulation_cnn.py --mode collect --episodes 2 --headless --time-limit 20 --seed 1 --data-dir "C:/Users/celin/AppData/Local/Temp/claude/46532ef1a24ecd12/c3929237-9927-44d3-8e3e-8c55c3c69963/scratchpad/simtest" 2>&1 | tail -8

OUT
c:/Users/celin/Documents/Studium/WISE2627/AIT/f1_28th_ait2026_basecode-main/.venv/Scripts/python.exe
Episode 1: default start, ended (off track) after 1.6s sim time, driven 0.22 laps
  saved 49 frames -> C:/Users/celin/AppData/Local/Temp/claude/46532ef1a24ecd12/c3929237-9927-44d3-8e3e-8c55c3c69963/scratchpad/simtest\collect_20261124_112044_1033.npz
Episode 2: default start, ended (off track) after 1.6s sim time, driven 0.22 laps
  saved 49 frames -> C:/Users/celin/AppData/Local/Temp/claude/46532ef1a24ecd12/c3929237-9927-44d3-8e3e-8c55c3c69963/scratchpad/simtest\collect_20261124_112045_7364.npz
11:20 AM
Der Code läuft. Allerdings kommt der Lidar-Controller jetzt schon nach 1,6 s von der Strecke ab. Ich vergleiche, wie er ohne die Änderungen fährt und mit jeder Änderung einzeln.



Bash
IN
cd "c:/Users/celin/Documents/Studium/WISE2627/AIT/AIT_2026_MaschineLearning/f1_28th_ait2026_basecode-main" && cat > "C:/Users/celin/AppData/Local/Temp/claude/46532ef1a24ecd12/c3929237-9927-44d3-8e3e-8c55c3c69963/scratchpad/cmp.py" <<'EOF'
import sys, math
sys.path.insert(0, ".")
from tools.simulation import Simulation
from simulation_singlethreaded import *
from simulation_cnn import DEFAULT_START_POSE
import numpy as np
def run(**kw):
    sim = Simulation(track_border_filename="track_borders_simulation.npy", vehicle_color="yellow",
        initial_pose_x_y_yaw=DEFAULT_START_POSE, meters_to_pixels=meters_to_pixels,
        vertical_resolution_px=vertical_resolution_px, height_over_ground_camera_m=height_over_ground_camera_m,
        dt=1/frames_per_seconds, time_limit_s=30, track_png_path="realLifeRennstrecke.png",
        track_size_px=(horizontal_resolution_px, vertical_resolution_px), egocam_render_size=ego_cam_resolution,
        egocam_fov_y_deg=ego_cam_opening_angle_horizontal_degrees, egocam_pitch_deg=ego_cam_pitch_degrees, **kw)
    s = sim.reset(); speeds=[]
    while not s["simulation_over"]:
        sp, st, *_ = sim.v.compute_next_command(kw.get("_td", 1/frames_per_seconds), [])
        st = np.clip(st, math.radians(sim.v.min_steering_angle_deg), math.radians(sim.v.max_steering_angle_deg))
        if sim.v.steering_inverted: st = -st
        s = sim.nextStep(sp, st); speeds.append(s["vehicle_speed_mps"])
    print(kw, f"-> {s['current_time']:.1f}s, {'time limit' if s['over_time_limit'] else 'off track'}, mean speed {np.mean(speeds):.2f}, max {np.max(speeds):.2f}")
run(actuation_delay_s=0, motor_time_constant_s=0, steering_rate_deg_per_s=None)
run(actuation_delay_s=0, motor_time_constant_s=0.3, steering_rate_deg_per_s=None)
run(actuation_delay_s=0, motor_time_constant_s=0, steering_rate_deg_per_s=300)
run(actuation_delay_s=None, motor_time_constant_s=0, steering_rate_deg_per_s=None)
run()
EOF
"c:/Users/celin/Documents/Studium/WISE2627/AIT/f1_28th_ait2026_basecode-main/.venv/Scripts/python.exe" "C:/Users/celin/AppData/Local/Temp/claude/46532ef1a24ecd12/c3929237-9927-44d3-8e3e-8c55c3c69963/scratchpad/cmp.py" 2>&1 | grep -- "->"

OUT
{'actuation_delay_s': 0, 'motor_time_constant_s': 0, 'steering_rate_deg_per_s': None} -> 30.0s, time limit, mean speed 0.99, max 1.10
{'actuation_delay_s': 0, 'motor_time_constant_s': 0.3, 'steering_rate_deg_per_s': None} -> 30.0s, time limit, mean speed 0.98, max 1.09
{'actuation_delay_s': 0, 'motor_time_constant_s': 0, 'steering_rate_deg_per_s': 300} -> 30.0s, time limit, mean speed 0.99, max 1.10
{'actuation_delay_s': None, 'motor_time_constant_s': 0, 'steering_rate_deg_per_s': None} -> 2.7s, off track, mean speed 0.86, max 1.10
{} -> 1.6s, off track, mean speed 0.77, max 1.06
Die Verzögerung allein bringt den Lidar-Controller schon von der Strecke ab. Der Controller könnte sie ausgleichen, bekommt in den Simulationsskripten aber nur 1/fps, also teste ich als Nächstes, wie er sich mit bekannter Verzögerung verhält.
Bash
IN
cd "c:/Users/celin/Documents/Studium/WISE2627/AIT/AIT_2026_MaschineLearning/f1_28th_ait2026_basecode-main" && F="C:/Users/celin/AppData/Local/Temp/claude/46532ef1a24ecd12/c3929237-9927-44d3-8e3e-8c55c3c69963/scratchpad/cmp.py" && sed -i 's/^def run(\*\*kw):/def run(_td=1\/30, **kw):/; s/kw.get("_td", 1\/frames_per_seconds)/_td/; s/^    print(kw,/    print(_td, kw,/' "$F" && sed -i '/^run(/d' "$F" && cat >> "$F" <<'EOF'
run(_td=0.12)
run(_td=0.12+0.1)
run(_td=0.12, motor_time_constant_s=0.3, steering_rate_deg_per_s=300)
EOF
"c:/Users/celin/Documents/Studium/WISE2627/AIT/f1_28th_ait2026_basecode-main/.venv/Scripts/python.exe" "$F" 2>&1 | grep -- "->"

OUT
0.12 {} -> 1.4s, off track, mean speed 0.73, max 1.05
0.22 {} -> 1.0s, off track, mean speed 0.60, max 1.02
0.12 {'motor_time_constant_s': 0.3, 'steering_rate_deg_per_s': 300} -> 1.4s, off track, mean speed 0.73, max 1.05
Bash
IN
cd "c:/Users/celin/Documents/Studium/WISE2627/AIT/AIT_2026_MaschineLearning/f1_28th_ait2026_basecode-main" && grep -n "def compute_next_command" -A60 localization/vehicle.py | head -90

OUT
184:    def compute_next_command(self, delta_t : float, opponents : list['Vehicle'] = []):
185-        x, y, yaw = self.getPositionEstimateCommandHistory(self.last_update + delta_t, longitudinal_offset_m=0.03) #TODO: find out which delta is required here!
186-
187-        #do not include opponents, if we are currently overtaken
188-        op = opponents
189-        if self.overtaking_followee_counter > 0:
190-            op = []
191-        distances, rays = self.lidar.getReadings(x, y, yaw, opponents = op)
192-
193-        #ask controller what to do
194-        setpoint, target_steering_angle_rad = self.controller.compute_steering(copy.copy(distances), rays)
195-
196-        if setpoint is None:
197-            return 0.0, 0.0, None, None, False
198-
199-        target_steering_angle_rad = target_steering_angle_rad - self.yaw
200-
201-        #filter target angle:
202-        #new_angle = old_angle*0.5+new_angle*0.5
203-        target_steering_angle_rad = get_average_angle([target_steering_angle_rad, self.target_steering_angle_rad])
204-
205-        #unwind target angle
206-        target_steering_angle_rad = unwind_angle(target_steering_angle_rad)
207-
208-
209-        #compute target velocity
210-        target_velocity_mps = 0.8
211-
212-        #find out how fast we can go: TODO: find out more reasonable numbers.
213-        target_steering_angle_deg = abs(math.degrees(target_steering_angle_rad))
214-        if 0 <= target_steering_angle_deg < 2:
215-            target_velocity_mps = 1.6
216-        if 2 <= target_steering_angle_deg < 5:
217-            target_velocity_mps = 1.4
218-        if 5 <= target_steering_angle_deg < 10:
219-            target_velocity_mps = 1.2
220-        if 10 <= target_steering_angle_deg < 20:
221-            target_velocity_mps = 1.0
222-        if 30 <= target_steering_angle_deg:
223-            target_velocity_mps = 0.8
224-
225-
226-        old_target_steering_angle_rad = target_steering_angle_rad
227-
228-        #apply lookahead to smooth out steering commands
229-        
230-        lookahead_m = target_velocity_mps * 0.20
231-        setpoint_distance = math.sqrt( (self.x-setpoint[0])**2 + (self.y-setpoint[1])**2 ) / self.meters_to_pixels
232-        #print(lookahead_m, setpoint_distance)
233-        lookahead_m = min(lookahead_m, max(0.1, setpoint_distance))
234-        steering_angle = math.atan2(2.0 * self.wheel_base_m * math.sin(target_steering_angle_rad), lookahead_m) #pure pursuit formula
235-        target_steering_angle_rad = steering_angle
236-        
237-
238-
239-        #apply speedfactor
240-        target_velocity_mps *= self.speedFactor
241-
242-
243-        #be aware of opponents on the track:
244-        opponent_we_are_following = self.isFollowingAnOpponent(opponents)
11:22 AM
Auch wenn er die Verzögerung kennt, hilft es nicht. Ich teste noch, wie viel Verzögerung der Lidar-Controller in der Simulation verträgt.



Bash
IN
cd "c:/Users/celin/Documents/Studium/WISE2627/AIT/AIT_2026_MaschineLearning/f1_28th_ait2026_basecode-main" && F="C:/Users/celin/AppData/Local/Temp/claude/46532ef1a24ecd12/c3929237-9927-44d3-8e3e-8c55c3c69963/scratchpad/cmp.py" && sed -i '/^run(/d' "$F" && cat >> "$F" <<'EOF'
run(actuation_delay_s=1/30)
run(actuation_delay_s=2/30)
run(actuation_delay_s=3/30)
EOF
"c:/Users/celin/Documents/Studium/WISE2627/AIT/f1_28th_ait2026_basecode-main/.venv/Scripts/python.exe" "$F" 2>&1 | grep -- "->"

OUT
0.03333333333333333 {'actuation_delay_s': 0.03333333333333333} -> 25.9s, off track, mean speed 0.97, max 1.10
0.03333333333333333 {'actuation_delay_s': 0.06666666666666667} -> 5.5s, off track, mean speed 0.89, max 1.07
0.03333333333333333 {'actuation_delay_s': 0.1} -> 2.4s, off track, mean speed 0.81, max 1.05
11:22 AM
Verzögerung und Trägheit von Motor und Lenkung sind jetzt in tools/simulation.py drin, und jede geänderte Stelle hat einen #SIMAN-Kommentar mit kurzer Erklärung auf Deutsch. Aber: Mit der Verzögerung kommt der Lidar-Controller nach etwa 1,6 s von der Strecke ab. Damit liefert --mode collect gerade keine brauchbaren Trainingsdaten mehr.

Was ich eingebaut habe
Drei neue Einstellungen in Simulation.__init__. Jede lässt sich einzeln abschalten:

Einstellung	Standardwert	Was sie macht	Abschalten mit
actuation_delay_s	None, also der delay_ms-Wert des echten Autos: gelb 120 ms, das sind 4 Schritte bei 30 fps	Ein Befehl wird erst später ausgeführt	0
motor_time_constant_s	0,3 s	Das Auto braucht Zeit zum Beschleunigen, Bremsen und Ausrollen	0
steering_rate_deg_per_s	300 °/s	Der Lenkservo dreht nur begrenzt schnell	None
So funktioniert es in nextStep():

Verzögerung: Jeder neue Befehl kommt hinten in eine Warteschlange, ausgeführt wird jeweils der älteste. reset() füllt die Warteschlange zu Beginn mit Stillstand-Befehlen.
Motorträgheit: Aus dem Motorwert wird wie vorher eine Zielgeschwindigkeit (Motorwert / 100). Die tatsächliche Geschwindigkeit nähert sich ihr aber nur langsam an. Nach 0,3 s ist etwa 63 % des Unterschieds aufgeholt.
Lenkträgheit: Der tatsächliche Lenkwinkel ändert sich pro Schritt höchstens um 300 °/s · dt, also um etwa 10°. Die Drehung des Autos wird jetzt aus diesem tatsächlichen Winkel berechnet, nicht mehr aus dem befohlenen.
simulation_cnn.py und simulation_singlethreaded.py musste ich nicht ändern. Sie nutzen automatisch die Standardwerte.

Testergebnis
Ich habe den Lidar-Controller jeweils 30 s fahren lassen, ohne Fenster:

Einstellung	Ergebnis
alles aus (altes Verhalten)	30 s geschafft
nur Motorträgheit	30 s geschafft
nur Lenkträgheit	30 s geschafft
Verzögerung 33 ms (1 Schritt)	nach 25,9 s von der Strecke
Verzögerung 67 ms	nach 5,5 s von der Strecke
Verzögerung 100 ms	nach 2,4 s von der Strecke
nur Verzögerung 120 ms	nach 2,7 s von der Strecke
alles an (Standard)	nach 1,6 s von der Strecke
Die Trägheit macht dem Controller nichts aus, die Verzögerung schon. Dass er die Verzögerung kennt (delta_t = 0.12, so wie in main.py), hilft in der Simulation auch nicht.

Das CNN hat dasselbe Problem. Es wurde ohne Verzögerung trainiert und wird in der neuen Simulation wahrscheinlich auch schlechter fahren.

Wie es weitergehen kann
Daten ohne Verzögerung sammeln, mit Verzögerung testen. Für collect setzt man actuation_delay_s=0 (dafür bräuchte simulation_cnn.py eine Option --delay). drive und dagger laufen dann mit der echten Verzögerung. DAgger ist genau dafür gedacht: Der Lehrer beschriftet die Bilder, auch wenn er selbst nicht fahren könnte.
Lidar-Controller anpassen, sodass er mit der Verzögerung klarkommt, zum Beispiel langsamer fahren oder weiter vorausschauen. Das ist aufwendiger.
Verzögerung beim Sammeln nur klein lassen, etwa 1 Schritt (33 ms).

# Latenz-Kompensation durch Vorhersage #VORSAGE
Für den Lidar-Controller macht der Code das sogar schon: getPositionEstimateCommandHistory() (vehicle.py:453) schätzt aus den zuletzt gesendeten Befehlen, wo das Auto in delta_t Sekunden sein wird, und misst den Lidar von dort aus. Für das Kamerabild gibt es drei Möglichkeiten.

Ansatz 1: Bild geometrisch „vorspulen“ (Image Warping)
Idee:

Man weiß, welche Befehle schon unterwegs, aber noch nicht ausgeführt sind. Das sind genau die in der Warteschlange.
Daraus rechnet man aus, wie weit sich das Auto in 0,12 s bewegt und wie stark es sich dreht.
Dann verzerrt man das Bild so, als stünde die Kamera schon dort.
Warum das klappen kann:

Die Strecke ist flach. Für Punkte auf einer Ebene lässt sich die Verschiebung zwischen zwei Kamerapositionen exakt mit einer Homographie (einer 3×3-Matrix) ausrechnen: cv.warpPerspective.
Alles, was man dafür braucht, ist bekannt: Kamerahöhe, Neigung (Pitch), Öffnungswinkel, Geschwindigkeit, Lenkwinkel.
Der Hintergrund oben ist keine Ebene und wird falsch verzerrt. Dieser Teil wird aber sowieso abgeschnitten (CROP_TOP_PX = 110).
Grenzen:

Vorwärtsfahrt ist harmlos. Bei 1 m/s sind das etwa 12 cm, das Bild wird nur etwas hineingezoomt. Alles, was man braucht, ist schon im Bild, nur etwas unschärfer.
In Kurven fehlt Bildinhalt. Bei voller Lenkung (27°) und 1 m/s dreht sich das Auto in 0,12 s um etwa 33° (Radstand 0,094 m). Die Kamera sieht nur 55° in der Breite, also fehlt dann mehr als die Hälfte des Bildes am Rand. In normalen Kurven (ca. 10° Lenkung) sind es immer noch etwa 13°.
Fehler bei der Vorhersage, zum Beispiel durch Rutschen oder einen falsch angenommenen Lenkwinkel, landen direkt im Bild.

## Ansatz 2: Das CNN lernt die Vorhersage selbst (meist besser) Gewählt
Statt das Bild umzubauen, bringt man dem Netz bei, für die Zukunft zu entscheiden:

Beschriftungen verschieben: Zu dem Bild zum Zeitpunkt t speichert man als richtige Antwort, was der Lehrer bei t + 0,12 s tun würde. Das Netz lernt dann von selbst, vorauszuschauen.
Befehle, die noch unterwegs sind, mitgeben: Neben dem Bild bekommt das Netz die letzten 4 gesendeten Befehle als zusätzlichen Input. Es weiß dann, was in den nächsten 0,12 s ohnehin noch passiert.
Mehrere Bilder hintereinander: Man gibt die letzten 2–3 Bilder gleichzeitig ins Netz. Daraus kann es Geschwindigkeit und Drehung selbst ablesen.
DAgger mit eingeschalteter Verzögerung: Wenn das CNN in der Simulation mit Verzögerung fährt und der Lehrer jedes Bild beschriftet, sieht das Netz genau die Situationen, in die es wegen der Verzögerung gerät. Es lernt dann, früher zu reagieren.
Vorteil: Es muss nichts von Hand modelliert werden, und das Netz lernt auch die Eigenheiten mit, die man schwer ausrechnen kann.

Ansatz 3: Ein Netz, das das zukünftige Bild erzeugt
Ein eigenes Netz bekommt das aktuelle Bild und die Befehle und sagt das Bild in 0,12 s vorher. In der Simulation hätte man dafür sogar perfekte Trainingsdaten, weil man das echte zukünftige Bild einfach rendern kann. Das ist aber viel Aufwand für wenig Gewinn gegenüber Ansatz 2.

Meine Einschätzung
Ansatz 1 ist eine gute Übung für Kamerageometrie und funktioniert auf Geraden und in leichten Kurven gut. In engen Kurven, wo es am meisten darauf ankommt, fehlt aber Bildinhalt.
Ansatz 2 (Beschriftungen verschieben oder DAgger mit Verzögerung) ist der übliche Weg und am wenigsten Arbeit. Ich würde damit anfangen.
Nebenbei gefunden: In vehicle.py:469 steht if not ...pointInsideTrack(...). Laut dem Kommentar darüber soll der Versatz nur innerhalb der Strecke addiert werden. Das not macht genau das Gegenteil. Das könnte ein Grund sein, warum die Verzögerungs-Kompensation des Lidar-Controllers im Test von vorhin nichts gebracht hat.

Was das Netz jetzt anders macht
Verschobene Beschriftung: Zum Bild bei Schritt t lernt das Netz den Befehl, den der Lehrer erst bei t + 4 geben würde. 4 Schritte sind etwa 133 ms, ungefähr die Verzögerung des echten Autos. Das Netz lernt also, vorauszuschauen.
Zusätzlicher Input cmds: Das Netz bekommt neben dem Bild die 4 Befehle, die schon gesendet, aber noch nicht ausgeführt sind. So weiß es, wo das Auto sein wird, wenn sein eigener Befehl ankommt.
Beim Training baut das Skript diese Befehle aus den gespeicherten Daten nach. Beim Fahren merkt sich das CNN die gesendeten Befehle selbst. Ich habe geprüft, dass beide Wege exakt dieselben Werte ergeben (Abweichung ca. 6·10⁻⁸, also nur Rundung).

Neue Dateien (alle in f1_28th_ait2026_basecode-main/)
Datei	Kopie von	Was neu ist
cnn_controller_vorhersage.py	cnn_controller.py	Modell mit zweitem Input cmds; CNNDriverVorhersage merkt sich die gesendeten Befehle mit Zeitstempel
simulation_cnn_vorhersage.py	simulation_cnn.py	Optionen --delay, --motor-tau, --steer-rate; speichert zusätzlich die gesendeten Befehle und die Verzögerung der Aufnahme
train_cnn_vorhersage.py	train_cnn.py	verschiebt die Beschriftung, baut den cmds-Input, Option --delay-steps, mehrere Datenordner möglich
main_vorhersage.py	main.py	nutzt das neue Modell und meldet ihm jeden wirklich gesendeten Befehl
So führst du es aus
Alle Befehle im Ordner f1_28th_ait2026_basecode-main:


# 1. Daten sammeln: der Lidar-Lehrer fährt OHNE Verzögerung (mit schafft er es nicht)
python simulation_cnn_vorhersage.py --mode collect --episodes 20 --headless --noise-deg 3 --random-start 0.7

# 2. Trainieren (Standard: 4 Schritte ≈ 133 ms). Die alten Daten aus cnn_data können mit rein:
python train_cnn_vorhersage.py --data-dir cnn_data_vorhersage cnn_data

# 3. Ansehen, wie das CNN MIT Verzögerung fährt
python simulation_cnn_vorhersage.py --mode drive

# 4. DAgger: das CNN fährt mit Verzögerung, der Lehrer beschriftet jedes Bild
python simulation_cnn_vorhersage.py --mode dagger --episodes 10 --headless --random-start 0.7
python train_cnn_vorhersage.py --data-dir cnn_data_vorhersage cnn_data --init cnn_driver_vorhersage.keras
# Schritt 4 mehrmals wiederholen

# 5. Echte Strecke
python main_vorhersage.py
Standardwerte:

Verzögerung: collect fährt ohne, drive und dagger mit der echten Verzögerung (120 ms, das ergibt 4 Schritte). Mit --delay 0.1 stellst du einen eigenen Wert ein.
Trägheit: Motor 0,3 s, Lenkung 300 °/s. Mit --motor-tau 0 bzw. --steer-rate 0 schaltest du sie ab.
Gut zu wissen
Die Verzögerung im Modell muss zur echten passen. Beim Start gibt main_vorhersage.py aus, für wie viele Millisekunden das Modell trainiert wurde. Auf der echten Strecke kommt zu den 120 ms noch die Übertragung der Frontkamera dazu. Wenn du die gemessen hast, trainierst du mit passendem --delay-steps neu, zum Beispiel 5 für 167 ms.
Alte Daten aus cnn_data funktionieren, wurden aber in der alten Simulation ohne Trägheit aufgenommen. Wenn genug neue Daten da sind, lass sie lieber weg.
DAgger ist hier besonders wichtig. Beim Sammeln fährt der Lehrer ohne Verzögerung, also sieht das Netz dort nie die typischen Fehler, die durch die Verzögerung entstehen. Erst im DAgger-Modus kommen diese Situationen in die Trainingsdaten.
Alte und neue Modelle sind nicht austauschbar. Das alte cnn_driver.keras funktioniert nicht mit den neuen Skripten und umgekehrt. Das Programm meldet das mit einer klaren Fehlermeldung.