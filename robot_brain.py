"""
robot_brain.py  ─  AI Robot Companion for Piano Guru
=====================================================
Drop this file (and dance_moves.py) in the same folder as main.py.
It runs as a background thread and listens to the SAME SerialManager
that main.py already uses.

How it plugs in
---------------
In main.py  PianoGameApp.build()  add two lines:

    from robot_brain import RobotBrain
    self.robot_brain = RobotBrain(self.serial_manager)
    self.robot_brain.start()

That's it.  Nothing else in main.py needs to change.

What this file does
-------------------
1.  Hooks into SerialManager to intercept FSR / key events.
2.  Runs three AI algorithms every 3 seconds while the user plays:
      • DTW  (dtaidistance)   → rhythm analysis
      • IsolationForest       → fatigue detection
      • MLP Neural Network    → engagement state
3.  Picks the highest-priority robot behaviour.
4.  Speaks a sentence via pyttsx3 (laptop speakers → amp → 8Ω speaker).
5.  Publishes a servo command to the robot ESP32 via MQTT (see dance_moves.py).
6.  Sends an OLED command to the PIANO ESP32 via the existing serial_manager.

Servo / dance moves
--------------------
All movement commands are handled by DanceMoveController in dance_moves.py.
Configure MQTT settings (broker, port, topic) in that file.

Dependencies (pip install these)
---------------------------------
    pip install pyttsx3 pyserial scikit-learn dtaidistance numpy paho-mqtt
"""

import threading
import time
import random
import queue
import numpy as np

from dance_moves_robot import DanceMoveController

# ── Optional imports with graceful fallback ──────────────────────────────────
try:
    import pyttsx3
    TTS_AVAILABLE = False
except ImportError:
    TTS_AVAILABLE = False
    print("[ROBOT] pyttsx3 not found – voice disabled. pip install pyttsx3")

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("[ROBOT] dtaidistance not found – rhythm uses fallback. pip install dtaidistance")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False
    print("[ROBOT] scikit-learn not found – ML disabled. pip install scikit-learn")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ─  edit these to match your setup
# ═══════════════════════════════════════════════════════════════════════════════

# How often (seconds) the AI analyses the player and reacts
ANALYSIS_INTERVAL = 3.0

# Minimum gap between spoken sentences (seconds) – avoids spamming the student
MIN_SPEAK_INTERVAL = 8.0

# Rolling window: how many recent note-press timestamps to keep
TIMING_WINDOW = 20

# ═══════════════════════════════════════════════════════════════════════════════
#  SPEECH TEMPLATES
#  Each state maps to a list of templates.
#  {name}, {streak}, {accuracy}, {adjective}, {quality} are filled at runtime.
# ═══════════════════════════════════════════════════════════════════════════════

ADJECTIVES = [
    "incredible", "outstanding", "fantastic", "brilliant",
    "impressive", "excellent", "wonderful", "superb",
    "remarkable", "magnificent", "stellar", "dazzling",
]

QUALITIES = [
    "rhythm", "energy", "focus", "momentum",
    "flow", "intensity", "precision", "control",
]

TEMPLATES = {
    # Student is doing well and engaged
    "engaged": [
        "That was {adjective}! Keep that {quality} going!",
        "{adjective} work! Your {quality} is really showing.",
        "I love your {quality}! Absolutely {adjective}.",
        "You are on fire! {adjective} {quality} right there.",
        "Brilliant! Your {quality} is sounding {adjective}.",
    ],
    # Student is rushing (timing gaps getting shorter)
    "rushing": [
        "Slow down a little. Breathe and feel the beat.",
        "Take your time. There is no race here.",
        "Relax your fingers. Let the notes breathe.",
        "You are rushing. Try counting in your head.",
        "Slower is better right now. Find your {quality}.",
    ],
    # Student is dragging (timing gaps getting longer)
    "dragging": [
        "Pick up the pace just a little.",
        "Keep that {quality} moving forward.",
        "You are slowing down. Trust your fingers.",
        "Stay with the rhythm. You can do this.",
        "A little more energy will help your {quality}.",
    ],
    # Student seems fatigued (force dropping, timing inconsistent)
    "fatigued": [
        "You have been working hard. Take a breath.",
        "Shake out your hands for a moment.",
        "Your fingers might need a short rest.",
        "Great effort today. It is okay to pause.",
        "Rest for a few seconds then come back strong.",
    ],
    # Student seems bored or disengaged (low force, slow)
    "bored": [
        "Let us add some energy! Play with more feeling.",
        "Wake those fingers up! You have got this.",
        "Imagine you are performing on a big stage.",
        "More {quality}! Show me what you can do.",
        "Dig in a little more. Your {quality} is there.",
    ],
    # Student is overwhelmed (lots of errors)
    "overwhelmed": [
        "That part is tricky. Let us go slowly.",
        "Mistakes are how we learn. You are doing great.",
        "Do not worry about errors. Focus on one note at a time.",
        "Take a breath. Everyone struggles with this at first.",
        "You are getting closer every time. Keep going.",
    ],
    # Student just finished a song well
    "celebrate": [
        "Yes! That was absolutely {adjective}!",
        "Outstanding! Your {quality} was perfect there!",
        "What a finish! {adjective} playing all the way through!",
        "I am so proud of that {quality}! {adjective}!",
        "Incredible performance! You nailed every note!",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC TRAINING DATA FOR MLP
#  Because we have no real labelled data yet, we generate rules-based examples.
#  Replace or supplement with real data once you collect it.
#
#  Feature vector order (must match _build_feature_vector):
#    [mean_force, force_std, timing_std, error_rate, streak_norm,
#     dtw_score, fatigue_score]
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_training_data():
    """
    Returns X (n_samples × 7 features) and y (n_samples labels).
    Labels: 0=engaged, 1=rushing, 2=dragging, 3=fatigued, 4=bored, 5=overwhelmed
    """
    rng = np.random.default_rng(42)
    X, y = [], []

    def add(n, label, mean_f, std_f, t_std, err, streak, dtw, fat):
        for _ in range(n):
            X.append([
                rng.normal(mean_f, 0.05),
                rng.normal(std_f,  0.02),
                rng.normal(t_std,  0.02),
                rng.normal(err,    0.03),
                rng.normal(streak, 0.05),
                rng.normal(dtw,    0.05),
                rng.normal(fat,    0.05),
            ])
            y.append(label)

    # engaged:     strong force, consistent timing, low errors, good streak
    add(80, 0, 0.75, 0.10, 0.10, 0.05, 0.70, 0.15, 0.10)
    # rushing:     high force, short timing gaps (low t_std captures inconsistency)
    add(80, 1, 0.80, 0.20, 0.40, 0.15, 0.50, 0.55, 0.15)
    # dragging:    moderate force, long gaps
    add(80, 2, 0.55, 0.10, 0.45, 0.10, 0.45, 0.50, 0.20)
    # fatigued:    dropping force, increasing errors, high fatigue
    add(80, 3, 0.35, 0.25, 0.35, 0.25, 0.30, 0.30, 0.75)
    # bored:       very low force, slow, low errors (not trying hard)
    add(80, 4, 0.25, 0.08, 0.50, 0.08, 0.20, 0.20, 0.30)
    # overwhelmed: high errors, inconsistent timing, moderate force
    add(80, 5, 0.60, 0.30, 0.50, 0.55, 0.15, 0.60, 0.35)

    X = np.clip(np.array(X), 0.0, 1.0)
    return X, np.array(y)


LABEL_NAMES = {
    0: "engaged",
    1: "rushing",
    2: "dragging",
    3: "fatigued",
    4: "bored",
    5: "overwhelmed",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  ROBOT BRAIN  ─  main class
# ═══════════════════════════════════════════════════════════════════════════════

class RobotBrain:
    """
    Plug-and-play robot AI companion.

    Usage (in PianoGameApp.build):
        from robot_brain import RobotBrain
        self.robot_brain = RobotBrain(self.serial_manager)
        self.robot_brain.start()
    """

    def __init__(self, serial_manager, user_name="friend"):
        self.serial_manager   = serial_manager   # existing SerialManager from main.py
        self.user_name        = user_name         # updated from app when name is entered

        # ── Live data buffers (filled by monkey-patched serial handler) ──────
        self.press_timestamps  = []   # millis of each note press (rolling)
        self.press_forces      = []   # FSR analog value 0-4095 → normalised 0-1
        self.error_count       = 0    # total errors this session
        self.correct_count     = 0    # total correct notes this session
        self.current_streak    = 0    # consecutive correct notes
        self.session_active    = False

        # ── Internal state ────────────────────────────────────────────────────
        self._lock             = threading.Lock()
        self._speak_queue      = queue.Queue()
        self._last_speak_time  = 0
        self._last_state       = None
        self._analysis_thread  = None
        self._tts_thread       = None
        self._running          = False
        self._current_genre    = "Neutral"   # updated by notify_new_melody()

        # ── Dance / servo controller (MQTT) ──────────────────────────────────
        self._dance = DanceMoveController()
        self._dance.connect()

        # ── TTS engine ────────────────────────────────────────────────────────
        self._tts_engine = None
        if TTS_AVAILABLE:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty('rate', 155)   # slightly slower = clearer
                self._tts_engine.setProperty('volume', 0.9)
                print("[ROBOT] TTS engine ready")
            except Exception as e:
                print(f"[ROBOT] TTS init failed: {e}")
                self._tts_engine = None

        # ── ML models ─────────────────────────────────────────────────────────
        self._mlp          = None
        self._scaler       = None
        self._iso_forest   = None
        if SK_AVAILABLE:
            self._train_models()

        # ── Patch SerialManager to intercept key-press events ────────────────
        self._patch_serial_manager()

    # ──────────────────────────────────────────────────────────────────────────
    #  MODEL TRAINING
    # ──────────────────────────────────────────────────────────────────────────

    def _train_models(self):
        """Train MLP and IsolationForest on synthetic data at startup."""
        print("[ROBOT] Training MLP on synthetic data...")
        X, y = _generate_synthetic_training_data()

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            max_iter=500,
            random_state=42,
        )
        self._mlp.fit(X_scaled, y)
        print("[ROBOT] MLP trained. Classes:", list(LABEL_NAMES.values()))

        # IsolationForest trained only on "normal" (engaged) samples
        X_normal = X[y == 0]
        self._iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.15,   # expect ~15% anomalous windows
            random_state=42,
        )
        self._iso_forest.fit(X_normal)
        print("[ROBOT] IsolationForest trained for fatigue detection")

    # ──────────────────────────────────────────────────────────────────────────
    #  SERIAL PATCH
    #  We wrap the existing SerialManager.handle_message so we can intercept
    #  "[CASUAL] Key N pressed" and "[RESULT]" lines without touching main.py.
    # ──────────────────────────────────────────────────────────────────────────

    def _patch_serial_manager(self):
        """Monkey-patch SerialManager.handle_message to intercept key events."""
        original_handle = self.serial_manager.handle_message

        def patched_handle(message):
            # Let main.py process it first
            original_handle(message)
            # Then our robot brain also processes it
            self._on_serial_message(message)

        self.serial_manager.handle_message = patched_handle
        print("[ROBOT] Serial manager patched – listening for key events")

    def _on_serial_message(self, message):
        """Called for every line received from the piano ESP32."""
        now = time.time()

        # ── Key press in any mode: "[CASUAL] Key N pressed"
        #    or during USER_PLAYING the ESP32 prints "Note X/Y: Expecting key N"
        #    and "  Pressed: N -> ..."

        if "[FSR]" in message and "Key" in message and "Force" in message:
            try:
                parts = message.strip().split()
                key_idx = int(parts[parts.index("Key") + 1])
                force_raw = int(parts[parts.index("Force") + 1])
                force_norm = max(0, min(force_raw / 4095.0, 1))

                with self._lock:
                    self.press_timestamps.append(now)
                    self.press_forces.append(force_norm)

                    if len(self.press_timestamps) > TIMING_WINDOW:
                        self.press_timestamps.pop(0)
                        self.press_forces.pop(0)

                    self.session_active = True

            except Exception as e:
                print("[ROBOT] FSR parse error:", e)

        # ── Correct note (no "WRONG" message = implied correct after release)
        if "Released. Moving to next note" in message:
            with self._lock:
                self.correct_count += 1
                self.current_streak += 1

        # ── Wrong note
        if "WRONG" in message:
            with self._lock:
                self.error_count += 1
                self.current_streak = 0

        # ── Song completed → trigger celebrate
        if "COMPLETION" in message and "finished" in message.lower():
            self._on_song_complete()

        # ── USERPLAYS started → reset session counters
        if "User playing mode activated" in message:
            with self._lock:
                self.error_count    = 0
                self.correct_count  = 0
                self.current_streak = 0
                self.press_timestamps.clear()
                self.press_forces.clear()
                self.session_active = True


        if "AI playing mode activated" in message or "AI is playing" in message:
            with self._lock:
                self.session_active = False
            self._send_servo_command("celebrate")
            self._update_oled("celebrate")
            return

        # ── STOP or IDLE
        if "Stopped" in message:
            with self._lock:
                self.session_active = False

    # ──────────────────────────────────────────────────────────────────────────
    #  FEATURE VECTOR
    # ──────────────────────────────────────────────────────────────────────────

    def _build_feature_vector(self):
        """
        Returns a 1×7 numpy array:
          [mean_force, force_std, timing_std, error_rate,
           streak_norm, dtw_score, fatigue_score]
        All values normalised to approximately [0, 1].
        """
        with self._lock:
            timestamps = list(self.press_timestamps)
            forces     = list(self.press_forces)
            errors     = self.error_count
            correct    = self.correct_count
            streak     = self.current_streak

        total_notes = errors + correct
        error_rate  = errors / max(total_notes, 1)
        streak_norm = min(streak / 20.0, 1.0)   # cap at 20-note streak = 1.0

        # Force stats
        if forces:
            mean_force = np.mean(forces)
            force_std  = np.std(forces)
        else:
            mean_force = 0.5
            force_std  = 0.1

        # Timing inter-onset intervals
        if len(timestamps) >= 3:
            iois        = np.diff(timestamps)               # seconds between presses
            ioi_norm    = np.clip(iois / 2.0, 0, 1)        # 2s gap = 1.0
            timing_std  = float(np.std(ioi_norm))
        else:
            ioi_norm   = np.array([0.3])
            timing_std = 0.1

        # DTW rhythm score
        dtw_score = self._compute_dtw_score(ioi_norm)

        # Fatigue score
        fatigue_score = self._compute_fatigue_score(forces, timestamps)

        return np.array([[
            float(np.clip(mean_force,    0, 1)),
            float(np.clip(force_std,     0, 1)),
            float(np.clip(timing_std,    0, 1)),
            float(np.clip(error_rate,    0, 1)),
            float(np.clip(streak_norm,   0, 1)),
            float(np.clip(dtw_score,     0, 1)),
            float(np.clip(fatigue_score, 0, 1)),
        ]])

    # ──────────────────────────────────────────────────────────────────────────
    #  ALGORITHM 1 – DTW RHYTHM ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_dtw_score(self, ioi_norm):
        """
        Compare the student's timing pattern to a perfectly steady beat.
        Returns 0.0 (perfect) → 1.0 (very inconsistent).
        """
        if len(ioi_norm) < 2:
            return 0.0

        mean_val  = float(np.mean(ioi_norm))
        reference = np.full(len(ioi_norm), mean_val)

        if DTW_AVAILABLE:
            try:
                distance = dtw.distance_fast(
                    ioi_norm.astype(np.double),
                    reference.astype(np.double),
                )
                score = distance / len(ioi_norm)
                return float(np.clip(score, 0, 1))
            except Exception:
                pass

        # Fallback without dtaidistance: normalised std deviation
        return float(np.clip(np.std(ioi_norm) / 0.5, 0, 1))

    def _classify_rhythm(self, ioi_norm):
        """
        Returns 'rushing', 'dragging', or 'steady' based on trend in IOIs.
        """
        if len(ioi_norm) < 4:
            return "steady"

        x     = np.arange(len(ioi_norm), dtype=float)
        slope = float(np.polyfit(x, ioi_norm, 1)[0])

        if slope < -0.03:
            return "rushing"
        elif slope > 0.03:
            return "dragging"
        return "steady"

    # ──────────────────────────────────────────────────────────────────────────
    #  ALGORITHM 2 – ISOLATION FOREST FATIGUE DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_fatigue_score(self, forces, timestamps):
        """
        Returns 0.0 (fresh) → 1.0 (fatigued).
        Uses IsolationForest if available; falls back to force trend.
        """
        if len(forces) < 5:
            return 0.0

        if SK_AVAILABLE and self._iso_forest is not None:
            try:
                force_arr  = np.array(forces[-10:]).reshape(-1, 1)
                n          = len(force_arr)
                mean_f     = np.full((n, 1), np.mean(force_arr))
                std_f      = np.full((n, 1), np.std(force_arr))
                feat_block = np.hstack([force_arr, mean_f, std_f,
                                        np.zeros((n, 4))])
                preds         = self._iso_forest.predict(feat_block[:, :1])
                anomaly_rate  = float(np.mean(preds == -1))
                return anomaly_rate
            except Exception:
                pass

        # Fallback: if force is trending down, assume fatigue
        if len(forces) >= 6:
            early = np.mean(forces[:3])
            late  = np.mean(forces[-3:])
            drop  = early - late
            return float(np.clip(drop * 2, 0, 1))

        return 0.0

    # ──────────────────────────────────────────────────────────────────────────
    #  ALGORITHM 3 – MLP ENGAGEMENT CLASSIFICATION
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_engagement(self, feature_vector):
        """
        Returns one of: engaged, rushing, dragging, fatigued, bored, overwhelmed
        """
        if SK_AVAILABLE and self._mlp is not None and self._scaler is not None:
            try:
                X_scaled = self._scaler.transform(feature_vector)
                pred     = self._mlp.predict(X_scaled)[0]
                return LABEL_NAMES[int(pred)]
            except Exception as e:
                print(f"[ROBOT] MLP predict error: {e}")

        # Rule-based fallback
        f = feature_vector[0]
        mean_force, force_std, timing_std, error_rate, streak_norm, dtw_score, fatigue = f

        if fatigue > 0.6:
            return "fatigued"
        if error_rate > 0.4:
            return "overwhelmed"
        if timing_std > 0.35 and dtw_score > 0.4:
            return "rushing"
        if mean_force < 0.3:
            return "bored"
        if streak_norm > 0.5 and error_rate < 0.1:
            return "engaged"
        return "engaged"

    # ──────────────────────────────────────────────────────────────────────────
    #  PRIORITY STATE RESOLVER
    #  Fatigue > overwhelmed > rhythm issues > engagement
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_state(self, mlp_state, rhythm_state, fatigue_score):
        """Combines MLP output, rhythm analysis, and fatigue into one state."""
        if fatigue_score > 0.65:
            return "fatigued"
        if mlp_state == "overwhelmed":
            return "overwhelmed"
        if rhythm_state == "rushing":
            return "rushing"
        if rhythm_state == "dragging":
            return "dragging"
        return mlp_state   # engaged / bored / overwhelmed

    # ──────────────────────────────────────────────────────────────────────────
    #  OUTPUT GENERATION
    # ──────────────────────────────────────────────────────────────────────────

    def _pick_sentence(self, state):
        template  = random.choice(TEMPLATES.get(state, TEMPLATES["engaged"]))
        adjective = random.choice(ADJECTIVES)
        quality   = random.choice(QUALITIES)
        sentence  = template.format(
            name      = self.user_name,
            streak    = max(1, self.current_streak),
            accuracy  = max(0, min(100, int(
                self.correct_count / max(self.correct_count + self.error_count, 1) * 100
            ))),
            adjective = adjective,
            quality   = quality,
        )
        return sentence

    def _speak(self, text):
        """Non-blocking speak: pushes to the TTS queue."""
        print(f"[ROBOT VOICE] {text}")
        self._speak_queue.put(text)

    def _send_servo_command(self, state):
        """Pick a random movement for the current state and send it via MQTT."""
        command = self._dance.send_move_for_state(state)
        print(f"[ROBOT SERVO] → {command}")

    def _update_oled(self, state):
        """Send an OLED update to the PIANO ESP32 using the existing serial_manager."""
        oled_messages = {
            "engaged":     ("Great job!", "Keep going!"),
            "rushing":     ("Slow down", "Feel the beat"),
            "dragging":    ("More energy!", "Keep moving"),
            "fatigued":    ("Take a rest", "You did great"),
            "bored":       ("Add energy!", "Play louder"),
            "overwhelmed": ("Stay calm", "One note at a time"),
            "celebrate":   ("AMAZING!", "Perfect play!"),
        }
        line1, line2 = oled_messages.get(state, ("Keep going!", ""))
        msg = f"LCD:{line1}|{line2}" if line2 else f"LCD:{line1}"
        self.serial_manager.send_command(msg)

    # ──────────────────────────────────────────────────────────────────────────
    #  SONG COMPLETE EVENT
    # ──────────────────────────────────────────────────────────────────────────

    def _on_song_complete(self):
        """Triggered when ESP32 reports completion."""
        with self._lock:
            correct = self.correct_count
            errors  = self.error_count
            genre   = self._current_genre

        total = correct + errors
        if total == 0:
            return

        accuracy = correct / total * 100
        state    = "celebrate" if accuracy >= 70 else "overwhelmed"

        sentence = self._pick_sentence(state)
        self._speak(sentence)
        self._update_oled(state)

        # GA-powered dance: genre colours the movement style,
        # accuracy drives the energy level within that style.
        print(f"[ROBOT] Song complete → state={state}, accuracy={accuracy:.1f}%, genre={genre}")
        self._dance.perform_dance(genre, accuracy)

    # ──────────────────────────────────────────────────────────────────────────
    #  ANALYSIS LOOP  (runs every ANALYSIS_INTERVAL seconds)
    # ──────────────────────────────────────────────────────────────────────────

    def _analysis_loop(self):
        """Background thread: runs AI every few seconds while session is active."""
        print("[ROBOT] Analysis loop started")
        while self._running:
            time.sleep(ANALYSIS_INTERVAL)

            with self._lock:
                active     = self.session_active
                timestamps = list(self.press_timestamps)
                forces     = list(self.press_forces)

            if not active or len(timestamps) < 4:
                continue   # not enough data yet

            feat_vec = self._build_feature_vector()

            with self._lock:
                iois = np.diff(timestamps) if len(timestamps) >= 2 else np.array([0.3])
            ioi_norm     = np.clip(iois / 2.0, 0, 1)
            rhythm_state = self._classify_rhythm(ioi_norm)
            fatigue      = feat_vec[0][6]
            mlp_state    = self._classify_engagement(feat_vec)
            final_state  = self._resolve_state(mlp_state, rhythm_state, fatigue)

            print(
                f"[ROBOT AI] MLP={mlp_state} | rhythm={rhythm_state} | "
                f"fatigue={fatigue:.2f} → final={final_state}"
            )

            now = time.time()
            if (final_state == self._last_state and
                    now - self._last_speak_time < MIN_SPEAK_INTERVAL * 1.5):
                continue

            if now - self._last_speak_time < MIN_SPEAK_INTERVAL:
                continue

            sentence = self._pick_sentence(final_state)
            self._speak(sentence)
            self._send_servo_command(final_state)
            self._update_oled(final_state)

            self._last_state      = final_state
            self._last_speak_time = now

    # ──────────────────────────────────────────────────────────────────────────
    #  TTS WORKER THREAD  (pyttsx3 must run on its own thread)
    # ──────────────────────────────────────────────────────────────────────────

    def _tts_worker(self):
        """Dedicated thread for speaking – keeps pyttsx3 off the main thread."""
        if not TTS_AVAILABLE:
            while self._running:
                try:
                    text = self._speak_queue.get(timeout=1)
                    print(f"[ROBOT TTS stub] Would say: {text}")
                except queue.Empty:
                    pass
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty('rate',   155)
            engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"[ROBOT] TTS worker engine error: {e}")
            return

        while self._running:
            try:
                text = self._speak_queue.get(timeout=1)
                engine.say(text)
                engine.runAndWait()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[ROBOT] TTS speak error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def start(self):
        """Start the analysis and TTS threads."""
        self._running = True

        self._analysis_thread = threading.Thread(
            target=self._analysis_loop, daemon=True, name="RobotAnalysis"
        )
        self._analysis_thread.start()

        self._tts_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="RobotTTS"
        )
        self._tts_thread.start()

        print("[ROBOT] Robot brain started ✓")

    def stop(self):
        """Graceful shutdown."""
        self._running = False
        self._dance.disconnect()
        print("[ROBOT] Robot brain stopped")

    def set_user_name(self, name):
        """Call this after the student enters their name."""
        self.user_name = name
        print(f"[ROBOT] User name set to: {name}")


    def notify_new_melody(self, genre: str = "Neutral"):
        """
        Call this from main.py whenever the AI generates a new melody.

        Parameters
        ----------
        genre : str
            The emotion/genre of the generated melody.
            Must be one of: Happy, Sad, Angry, Surprise, Fear, Neutral, Disgust
            (these match the KNNGenreClassifier outputs).
            Defaults to "Neutral" if not supplied.

        Example (in main.py, after melody generation):
            detected_genre = self.knn_classifier.classify(emotion)
            self.robot_brain.notify_new_melody(genre=detected_genre)
        """
        valid_genres = {"Happy", "Sad", "Angry", "Surprise", "Fear", "Neutral", "Disgust"}
        genre = genre.strip().capitalize()
        if genre not in valid_genres:
            print(f"[ROBOT] Unknown genre '{genre}' – defaulting to Neutral")
            genre = "Neutral"

        with self._lock:
            self._current_genre = genre

        print(f"[ROBOT] New AI melody generated – genre={genre}")

        # Attention blink + note display on ESP32 OLED (unchanged behaviour)
        self._dance.send_command("NEW_MELODY")

        # Launch GA choreography tuned to this genre.
        # Score 50.0 = neutral excitement for a fresh melody (not a result yet).
        self._dance.perform_dance(genre, score=50.0)
