"""
dance_moves_robot.py  ─  Robot Servo / Dance Movement Controller
===============================================================
Drop-in replacement for the original file.
Compatible with all existing robot_brain.py calls:
  - robot_brain._dance.send_face("ENERGETIC")
  - robot_brain._dance.send_face_for_accuracy(accuracy)
  - robot_brain._dance.send_move_for_state("engaged")
  - robot_brain._dance.send_command("CELEBRATE")

New public API (called from robot_brain.py):
  - robot_brain._dance.perform_dance(genre, score)
      genre : str  — one of the KNN genres: Happy, Sad, Angry,
                      Surprise, Fear, Neutral, Disgust
      score : float — 0.0 (terrible) → 100.0 (perfect)
      Evolves a choreography using a Genetic Algorithm and plays it.
      Sends IDLE_MODE when done.

How the Genetic Algorithm works
--------------------------------
  Chromosome  : a list of (face_cmd, body_cmd) pairs, length SEQ_LEN
  Gene pool   : sampled from GENRE_MOVE_POOLS[genre]
  Fitness     : measures how well each move matches
                  - the target GENRE_PROFILE (energy, fluidity, sharpness)
                  - the SCORE_TIER (high score → high energy moves rewarded)
                  - variety (penalises back-to-back repeats)
  Selection   : tournament selection (k=3)
  Crossover   : single-point
  Mutation    : random gene replacement at MUTATION_RATE
  Generations : GA_GENERATIONS
  After GA    : best chromosome is published move-by-move via MQTT,
                then IDLE_MODE is sent.

All MQTT command strings match robot_code.ino exactly.
"""

import random
import time
import threading

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[DANCE] paho-mqtt not found – MQTT disabled. pip install paho-mqtt")


# ═════════════════════════════════════════════════════════════════════════════
#  MQTT CONFIG  (must match robot_code.ino)
# ═════════════════════════════════════════════════════════════════════════════
MQTT_BROKER        = "10.146.247.242"
MQTT_PORT          = 1883
MQTT_KEEPALIVE     = 60
MQTT_COMMAND_TOPIC = "robot/servo/command"
MQTT_STATUS_TOPIC  = "robot/servo/status"
MQTT_CLIENT_ID     = "robot_brain"

MQTT_USE_AUTH      = False
MQTT_USERNAME      = ""
MQTT_PASSWORD      = ""


# ═════════════════════════════════════════════════════════════════════════════
#  VALID COMMANDS  (every string here must exist in robot_code.ino)
# ═════════════════════════════════════════════════════════════════════════════

# Face commands — sent as OLED emotion overlays
ALL_FACES = [
    "FACE_HAPPY",
    "FACE_SAD",
    "FACE_ANGRY",
    "FACE_ENERGETIC",
    "FACE_TIRED",
    "FACE_CALM",
    "FACE_MISERABLE",
    "FACE_CONFUSED",
    "FACE_BLUSHING",
    "FACE_EXCITED",
]

# Body/movement commands
ALL_MOVES = [
    "HEAD_NOD",
    "RAISE_HAND",
    "WAVE_FORWARD",
    "CONFUSED",
    "CELEBRATE",
    "SLEEPY",
    "DANCE_HAPPY",
    "SLOW_DOWN",
    "PUMP_UP",
    "GENTLE_NOD",
    "HEAD_SHAKE",
    "BOTH_ARMS_UP",
    "COMFORT",
    "RAISE_LEFT_ARM",
    "LOWER_LEFT_ARM",
    "RAISE_RIGHT_ARM",
    "LOWER_RIGHT_ARM",
    "RAISE_BOTH_ARMS",
    "LOWER_BOTH_ARMS",
    "LOOK_LEFT",
    "LOOK_RIGHT",
    "LOOK_CENTRE",
    "WAIST_LEFT",
    "WAIST_RIGHT",
    "WAIST_CENTRE",
]


# ═════════════════════════════════════════════════════════════════════════════
#  GENRE MOVE POOLS
#  Each genre restricts which moves and faces make sense.
#  The GA will only sample genes from these pools, keeping choreography
#  thematically coherent per genre.
#
#  Format per entry: (face_cmd, body_cmd, energy, fluidity, sharpness)
#    energy    0.0–1.0  (how physically big/active the move is)
#    fluidity  0.0–1.0  (how smooth/slow vs jerky)
#    sharpness 0.0–1.0  (how precise/snappy vs loose)
# ═════════════════════════════════════════════════════════════════════════════

GENE = {
    # (face, move, energy, fluidity, sharpness)
    "happy_dance":      ("FACE_HAPPY",     "DANCE_HAPPY",     0.9, 0.5, 0.7),
    "happy_celebrate":  ("FACE_EXCITED",   "CELEBRATE",       1.0, 0.4, 0.8),
    "happy_wave":       ("FACE_HAPPY",     "WAVE_FORWARD",    0.7, 0.7, 0.5),
    "happy_botharms":   ("FACE_EXCITED",   "BOTH_ARMS_UP",    0.8, 0.5, 0.6),
    "happy_pump":       ("FACE_ENERGETIC", "PUMP_UP",         1.0, 0.3, 0.9),
    "happy_nod":        ("FACE_HAPPY",     "HEAD_NOD",        0.5, 0.7, 0.4),
    "happy_raise_r":    ("FACE_BLUSHING",  "RAISE_RIGHT_ARM", 0.6, 0.6, 0.5),
    "happy_waist_l":    ("FACE_HAPPY",     "WAIST_LEFT",      0.7, 0.6, 0.5),
    "happy_waist_r":    ("FACE_HAPPY",     "WAIST_RIGHT",     0.7, 0.6, 0.5),

    "sad_comfort":      ("FACE_SAD",       "COMFORT",         0.2, 0.9, 0.1),
    "sad_gentle":       ("FACE_MISERABLE", "GENTLE_NOD",      0.2, 0.9, 0.1),
    "sad_sleepy":       ("FACE_TIRED",     "SLEEPY",          0.1, 1.0, 0.0),
    "sad_slow":         ("FACE_CALM",      "SLOW_DOWN",       0.3, 0.9, 0.1),
    "sad_look_l":       ("FACE_SAD",       "LOOK_LEFT",       0.2, 0.8, 0.1),
    "sad_look_r":       ("FACE_MISERABLE", "LOOK_RIGHT",      0.2, 0.8, 0.1),
    "sad_waist_c":      ("FACE_CALM",      "WAIST_CENTRE",    0.1, 1.0, 0.0),
    "sad_lower_both":   ("FACE_TIRED",     "LOWER_BOTH_ARMS", 0.2, 0.9, 0.1),

    "angry_shake":      ("FACE_ANGRY",     "HEAD_SHAKE",      0.9, 0.2, 1.0),
    "angry_pump":       ("FACE_ANGRY",     "PUMP_UP",         1.0, 0.2, 1.0),
    "angry_raise_r":    ("FACE_ANGRY",     "RAISE_RIGHT_ARM", 0.8, 0.3, 0.9),
    "angry_raise_l":    ("FACE_ANGRY",     "RAISE_LEFT_ARM",  0.8, 0.3, 0.9),
    "angry_both":       ("FACE_ANGRY",     "RAISE_BOTH_ARMS", 0.9, 0.2, 1.0),
    "angry_waist_l":    ("FACE_ANGRY",     "WAIST_LEFT",      0.8, 0.3, 0.8),
    "angry_waist_r":    ("FACE_ANGRY",     "WAIST_RIGHT",     0.8, 0.3, 0.8),
    "angry_look_l":     ("FACE_ANGRY",     "LOOK_LEFT",       0.7, 0.3, 0.8),

    "surprise_wave":    ("FACE_EXCITED",   "WAVE_FORWARD",    0.8, 0.5, 0.7),
    "surprise_both":    ("FACE_EXCITED",   "BOTH_ARMS_UP",    0.9, 0.4, 0.8),
    "surprise_nod":     ("FACE_EXCITED",   "HEAD_NOD",        0.6, 0.5, 0.5),
    "surprise_look_l":  ("FACE_EXCITED",   "LOOK_LEFT",       0.5, 0.6, 0.5),
    "surprise_look_r":  ("FACE_EXCITED",   "LOOK_RIGHT",      0.5, 0.6, 0.5),
    "surprise_raise_r": ("FACE_EXCITED",   "RAISE_RIGHT_ARM", 0.7, 0.5, 0.6),
    "surprise_confused":("FACE_CONFUSED",  "CONFUSED",        0.5, 0.4, 0.5),

    "fear_confused":    ("FACE_CONFUSED",  "CONFUSED",        0.4, 0.5, 0.6),
    "fear_shake":       ("FACE_MISERABLE", "HEAD_SHAKE",      0.5, 0.4, 0.7),
    "fear_slow":        ("FACE_TIRED",     "SLOW_DOWN",       0.3, 0.7, 0.2),
    "fear_look_l":      ("FACE_CONFUSED",  "LOOK_LEFT",       0.4, 0.6, 0.4),
    "fear_look_r":      ("FACE_CONFUSED",  "LOOK_RIGHT",      0.4, 0.6, 0.4),
    "fear_lower_l":     ("FACE_MISERABLE", "LOWER_LEFT_ARM",  0.2, 0.8, 0.1),
    "fear_lower_r":     ("FACE_MISERABLE", "LOWER_RIGHT_ARM", 0.2, 0.8, 0.1),

    "neutral_nod":      ("FACE_CALM",      "HEAD_NOD",        0.5, 0.6, 0.4),
    "neutral_wave":     ("FACE_CALM",      "WAVE_FORWARD",    0.5, 0.7, 0.4),
    "neutral_gentle":   ("FACE_CALM",      "GENTLE_NOD",      0.3, 0.9, 0.2),
    "neutral_look_l":   ("FACE_CALM",      "LOOK_LEFT",       0.3, 0.8, 0.3),
    "neutral_look_r":   ("FACE_CALM",      "LOOK_RIGHT",      0.3, 0.8, 0.3),
    "neutral_waist_l":  ("FACE_CALM",      "WAIST_LEFT",      0.4, 0.7, 0.3),
    "neutral_waist_r":  ("FACE_CALM",      "WAIST_RIGHT",     0.4, 0.7, 0.3),
    "neutral_centre":   ("FACE_CALM",      "WAIST_CENTRE",    0.2, 0.9, 0.1),

    "disgust_shake":    ("FACE_ANGRY",     "HEAD_SHAKE",      0.7, 0.3, 0.8),
    "disgust_confused": ("FACE_CONFUSED",  "CONFUSED",        0.5, 0.4, 0.6),
    "disgust_look_l":   ("FACE_ANGRY",     "LOOK_LEFT",       0.6, 0.4, 0.6),
    "disgust_raise":    ("FACE_MISERABLE", "RAISE_HAND",      0.5, 0.5, 0.5),
    "disgust_slow":     ("FACE_TIRED",     "SLOW_DOWN",       0.3, 0.7, 0.2),
}

# Map genre name → list of gene keys valid for that genre
GENRE_POOLS = {
    "Happy":    [k for k in GENE if k.startswith("happy_")],
    "Sad":      [k for k in GENE if k.startswith("sad_")],
    "Angry":    [k for k in GENE if k.startswith("angry_")],
    "Surprise": [k for k in GENE if k.startswith("surprise_")],
    "Fear":     [k for k in GENE if k.startswith("fear_")],
    "Neutral":  [k for k in GENE if k.startswith("neutral_")],
    "Disgust":  [k for k in GENE if k.startswith("disgust_")],
}

# Target movement profile per genre: (energy, fluidity, sharpness)
GENRE_PROFILE = {
    "Happy":    (0.85, 0.55, 0.65),
    "Sad":      (0.20, 0.90, 0.10),
    "Angry":    (0.90, 0.20, 0.95),
    "Surprise": (0.75, 0.50, 0.65),
    "Fear":     (0.40, 0.55, 0.55),
    "Neutral":  (0.40, 0.75, 0.35),
    "Disgust":  (0.60, 0.35, 0.70),
}


# ═════════════════════════════════════════════════════════════════════════════
#  GENETIC ALGORITHM CONFIG
# ═════════════════════════════════════════════════════════════════════════════
GA_POP_SIZE      = 20    # individuals per generation
GA_GENERATIONS   = 25    # number of evolution cycles
GA_MUTATION_RATE = 0.25  # probability a gene mutates
GA_TOURNAMENT_K  = 3     # tournament selection size
SEQ_LEN          = 5     # moves per choreography


# ═════════════════════════════════════════════════════════════════════════════
#  DELAY BETWEEN MOVES (seconds)
#  The ESP32 runs each move synchronously, so we wait before sending the next.
#  Approximate move durations from robot_code.ino timings.
# ═════════════════════════════════════════════════════════════════════════════
MOVE_DELAY = {
    "HEAD_NOD":        1.2,
    "RAISE_HAND":      1.5,
    "WAVE_FORWARD":    1.5,
    "CONFUSED":        1.5,
    "CELEBRATE":       2.5,
    "SLEEPY":          1.8,
    "DANCE_HAPPY":     2.0,
    "SLOW_DOWN":       2.5,
    "PUMP_UP":         1.8,
    "GENTLE_NOD":      1.5,
    "HEAD_SHAKE":      1.2,
    "BOTH_ARMS_UP":    2.0,
    "COMFORT":         2.0,
    "RAISE_LEFT_ARM":  1.2,
    "LOWER_LEFT_ARM":  1.0,
    "RAISE_RIGHT_ARM": 1.2,
    "LOWER_RIGHT_ARM": 1.0,
    "RAISE_BOTH_ARMS": 1.2,
    "LOWER_BOTH_ARMS": 1.0,
    "LOOK_LEFT":       1.0,
    "LOOK_RIGHT":      1.0,
    "LOOK_CENTRE":     0.8,
    "WAIST_LEFT":      1.0,
    "WAIST_RIGHT":     1.0,
    "WAIST_CENTRE":    0.8,
}


# ═════════════════════════════════════════════════════════════════════════════
#  LEGACY STATE → MOVE MAPPING  (kept for send_move_for_state compatibility)
# ═════════════════════════════════════════════════════════════════════════════
STATE_TO_MOVE = {
    "engaged":     ["HEAD_NOD",    "WAVE_FORWARD", "DANCE_HAPPY"],
    "rushing":     ["RAISE_HAND",  "HEAD_SHAKE",   "SLOW_DOWN"],
    "dragging":    ["WAVE_FORWARD","HEAD_NOD",     "PUMP_UP"],
    "fatigued":    ["SLEEPY",      "HEAD_NOD",    "GENTLE_NOD"],
    "bored":       ["PUMP_UP",     "DANCE_HAPPY",  "WAVE_FORWARD"],
    "overwhelmed": ["GENTLE_NOD",  "HEAD_NOD",    "COMFORT"],
    "celebrate":   ["CELEBRATE",   "DANCE_HAPPY",  "BOTH_ARMS_UP"],
}

STATE_TO_FACE = {
    "engaged":     "FACE_HAPPY",
    "rushing":     "FACE_CONFUSED",
    "dragging":    "FACE_ENERGETIC",
    "fatigued":    "FACE_TIRED",
    "bored":       "FACE_EXCITED",
    "overwhelmed": "FACE_CALM",
    "celebrate":   "FACE_HAPPY",
}


# ═════════════════════════════════════════════════════════════════════════════
#  GENETIC ALGORITHM — pure functions (no I/O, easy to test)
# ═════════════════════════════════════════════════════════════════════════════

def _random_chromosome(pool: list, length: int) -> list:
    """Return a random list of gene keys from the genre pool."""
    return [random.choice(pool) for _ in range(length)]


def _fitness(chromosome: list, genre: str, score: float) -> float:
    """
    Score a choreography chromosome.

    Parameters
    ----------
    chromosome : list of gene keys
    genre      : genre string e.g. "Happy"
    score      : player accuracy 0–100

    Returns
    -------
    float  — higher is better
    """
    target_e, target_f, target_s = GENRE_PROFILE.get(genre, (0.5, 0.5, 0.5))

    # Score overrides the profile completely at extremes:
    #
    #   score >= 70  → full celebration: high energy, sharp, not fluid
    #                  genre still colours the move pool but energy maxes out
    #   score 40–69  → genre expresses naturally, no adjustment
    #   score 20–39  → robot looks tired/discouraged: low energy, high fluidity
    #   score <  20  → robot looks defeated: very low energy, very high fluidity
    #
    # This means at 8% accuracy the robot will always look tired/sad
    # regardless of the song genre.
    if score >= 70:
        adjusted_e = min(1.0, target_e + 0.30)   # boost energy hard
        target_f   = max(0.0, target_f - 0.20)   # less fluid = snappier
        target_s   = min(1.0, target_s + 0.20)   # sharper
    elif score >= 40:
        adjusted_e = target_e                     # genre as-is
    elif score >= 20:
        adjusted_e = max(0.0, target_e - 0.45)   # significantly dampen energy
        target_f   = min(1.0, target_f + 0.30)   # more fluid/slow
        target_s   = max(0.0, target_s - 0.30)   # less sharp
    else:
        adjusted_e = max(0.0, target_e - 0.70)   # near-zero energy
        target_f   = 1.0                          # max fluid (slow, droopy)
        target_s   = 0.0                          # no sharpness at all

    fitness = 0.0
    prev_key = None

    for key in chromosome:
        face, move, gene_e, gene_f, gene_s = GENE[key]

        # How close is this gene to the target profile?
        diff = (
            (gene_e - adjusted_e) ** 2 +
            (gene_f - target_f)   ** 2 +
            (gene_s - target_s)   ** 2
        )
        gene_score = 1.0 - (diff / 3.0) ** 0.5   # 0–1, higher is better

        # Variety bonus: penalise consecutive identical moves
        if key == prev_key:
            gene_score -= 0.4

        fitness += gene_score
        prev_key = key

    # Normalise to [0, 1]
    return fitness / len(chromosome)


def _tournament_select(population: list, fitnesses: list, k: int):
    """Return one chromosome via tournament selection."""
    contestants = random.sample(range(len(population)), k)
    best = max(contestants, key=lambda i: fitnesses[i])
    return population[best][:]   # copy


def _crossover(parent_a: list, parent_b: list) -> tuple:
    """Single-point crossover → two children."""
    point = random.randint(1, len(parent_a) - 1)
    child_a = parent_a[:point] + parent_b[point:]
    child_b = parent_b[:point] + parent_a[point:]
    return child_a, child_b


def _mutate(chromosome: list, pool: list, rate: float) -> list:
    """Randomly replace genes with probability `rate`."""
    return [
        random.choice(pool) if random.random() < rate else gene
        for gene in chromosome
    ]


# Score → energy threshold mapping
# Only genes with energy <= threshold are allowed in the pool at that score tier.
# This hard-filters the gene pool BEFORE the GA runs so high-energy genes
# are simply unavailable at low scores — fitness nudging alone isn't enough.
SCORE_ENERGY_CAP = [
    (20,  0.25),   # score <  20 → only very low energy genes (defeated)
    (40,  0.45),   # score <  40 → low energy genes (discouraged)
    (70,  0.75),   # score <  70 → mid energy genes (genre as-is)
    (101, 1.00),   # score >= 70 → all genes available (celebrate)
]

# Score → face override — at low scores ignore the genre face entirely
SCORE_FACE_OVERRIDE = [
    (20,  ["FACE_TIRED", "FACE_MISERABLE"]),   # defeated
    (40,  ["FACE_TIRED", "FACE_SAD", "FACE_CALM"]),  # discouraged
    (70,  None),    # genre decides
    (101, None),    # genre decides (celebrate)
]


def _get_score_pool(genre: str, score: float) -> list:
    """Return gene pool filtered by score-based energy cap."""
    base_pool = GENRE_POOLS.get(genre, GENRE_POOLS["Neutral"])

    energy_cap = 1.0
    for threshold, cap in SCORE_ENERGY_CAP:
        if score < threshold:
            energy_cap = cap
            break

    filtered = [k for k in base_pool if GENE[k][2] <= energy_cap]

    # If filtering removed everything (e.g. genre is Angry and score is 5%),
    # fall back to Sad/Fear pools which are naturally low energy
    if not filtered:
        fallback = GENRE_POOLS["Sad"] + GENRE_POOLS["Fear"]
        filtered = [k for k in fallback if GENE[k][2] <= energy_cap]

    # Final safety — return full base pool rather than empty
    return filtered if filtered else base_pool


def _get_face_override(score: float):
    """Return list of allowed faces for this score, or None to let genre decide."""
    for threshold, faces in SCORE_FACE_OVERRIDE:
        if score < threshold:
            return faces
    return None


def evolve_choreography(genre: str, score: float) -> list:
    """
    Run the genetic algorithm and return the best chromosome
    as a list of (face_cmd, body_cmd) tuples ready to publish.

    Parameters
    ----------
    genre : str   — must be a key in GENRE_POOLS
    score : float — player accuracy 0.0–100.0
    """
    pool = _get_score_pool(genre, score)
    face_override = _get_face_override(score)

    # Initialise population
    population = [_random_chromosome(pool, SEQ_LEN) for _ in range(GA_POP_SIZE)]

    for generation in range(GA_GENERATIONS):
        fitnesses = [_fitness(ch, genre, score) for ch in population]

        next_gen = []

        # Elitism: carry the best individual forward unchanged
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        next_gen.append(population[best_idx][:])

        while len(next_gen) < GA_POP_SIZE:
            p_a = _tournament_select(population, fitnesses, GA_TOURNAMENT_K)
            p_b = _tournament_select(population, fitnesses, GA_TOURNAMENT_K)
            child_a, child_b = _crossover(p_a, p_b)
            child_a = _mutate(child_a, pool, GA_MUTATION_RATE)
            child_b = _mutate(child_b, pool, GA_MUTATION_RATE)
            next_gen.append(child_a)
            if len(next_gen) < GA_POP_SIZE:
                next_gen.append(child_b)

        population = next_gen

    # Final evaluation
    fitnesses = [_fitness(ch, genre, score) for ch in population]
    best = population[max(range(len(population)), key=lambda i: fitnesses[i])]

    print(
        f"[DANCE GA] Genre={genre} Score={score:.1f} → "
        f"Fitness={max(fitnesses):.3f} | "
        f"Sequence: {[GENE[k][1] for k in best]}"
    )

    # Convert gene keys → (face_cmd, body_cmd) pairs
    # Apply face override for low scores so robot looks tired/sad not happy
    result = []
    for k in best:
        face_cmd, body_cmd = GENE[k][0], GENE[k][1]
        if face_override:
            face_cmd = random.choice(face_override)
        result.append((face_cmd, body_cmd))
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  DANCE MOVE CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════

class DanceMoveController:
    """
    Handles MQTT connection and all command dispatch.
    GA choreography runs in a background thread to avoid blocking the
    robot brain's analysis loop.
    """

    def __init__(self):
        self._client      = None
        self._connected   = False
        self._dance_lock  = threading.Lock()   # prevent overlapping dances

    # ─── MQTT lifecycle ──────────────────────────────────────────────────────

    def connect(self):
        if not MQTT_AVAILABLE:
            print("[DANCE] MQTT unavailable – commands will be printed only")
            return

        try:
            self._client = mqtt.Client(client_id=MQTT_CLIENT_ID)

            if MQTT_USE_AUTH:
                self._client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

            self._client.on_connect    = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message    = self._on_message

            self._client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
            self._client.loop_start()
            time.sleep(0.5)

        except Exception as e:
            print(f"[DANCE] MQTT connect error: {e}")
            self._client    = None
            self._connected = False

    def disconnect(self):
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception as e:
                print(f"[DANCE] MQTT disconnect error: {e}")

        self._client    = None
        self._connected = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            print(f"[DANCE] Connected to MQTT broker at {MQTT_BROKER}")
            client.subscribe(MQTT_STATUS_TOPIC)
        else:
            self._connected = False
            print(f"[DANCE] MQTT connection refused – rc {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        print("[DANCE] MQTT disconnected")

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8").strip()
            print(f"[DANCE] ESP32 status: {payload}")
        except Exception:
            pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ─── Low-level publish ───────────────────────────────────────────────────

    def _publish(self, command: str):
        """Publish one command string to the ESP32 via MQTT."""
        command = str(command).strip().upper()
        print(f"[DANCE] → {command}")

        if self._client and self._connected:
            try:
                self._client.publish(MQTT_COMMAND_TOPIC, payload=command, qos=1)
            except Exception as e:
                print(f"[DANCE] Publish error: {e}")
        else:
            print("[DANCE] Not connected to broker – command printed only")

    # ─── Public command API (kept for robot_brain.py compatibility) ──────────

    def send_command(self, command: str):
        """Send any raw command string directly."""
        self._publish(command)

    def send_face(self, face: str):
        """
        Send a face expression command.
        Accepts both "ENERGETIC" and "FACE_ENERGETIC" forms.
        """
        if not face:
            face = "HAPPY"
        face = str(face).strip().upper()
        if not face.startswith("FACE_"):
            face = "FACE_" + face
        self._publish(face)

    def send_face_for_accuracy(self, accuracy):
        """Map a numeric accuracy score to an appropriate face."""
        try:
            accuracy = float(accuracy)
        except Exception:
            accuracy = 0

        if accuracy >= 90:
            self.send_face("HAPPY")
        elif accuracy >= 70:
            self.send_face("EXCITED")
        elif accuracy >= 50:
            self.send_face("CALM")
        else:
            self.send_face("TIRED")

    def send_move_for_state(self, state: str) -> str:
        """
        Legacy method: pick one face + one body move for a player state.
        Used by robot_brain._send_servo_command() during analysis loop.
        NO IDLE_MODE here — this runs constantly during gameplay and
        sending IDLE_MODE after every move would reset the face every second.
        """
        state    = str(state).strip().lower()
        face_cmd = STATE_TO_FACE.get(state, "FACE_HAPPY")
        move_cmd = random.choice(STATE_TO_MOVE.get(state, ["HEAD_NOD"]))

        self._publish(face_cmd)
        time.sleep(0.1)
        self._publish(move_cmd)

        return f"{face_cmd} + {move_cmd}"

    # ─── GA-powered dance performance ────────────────────────────────────────

    def perform_dance(self, genre: str, score: float):
        """
        Evolve a choreography for `genre` and `score`, then perform it.
        Runs in a background thread; sends IDLE_MODE when done.

        Parameters
        ----------
        genre : str   — KNN genre e.g. "Happy", "Sad", "Angry"
        score : float — accuracy percentage 0.0–100.0
        """
        thread = threading.Thread(
            target=self._dance_thread,
            args=(genre, score),
            daemon=True,
            name="DancePerform",
        )
        thread.start()

    def _dance_thread(self, genre: str, score: float):
        """Background worker: evolve → perform → idle."""
        # Only one dance at a time
        if not self._dance_lock.acquire(blocking=False):
            print("[DANCE] Dance already running – skipping")
            return

        try:
            print(f"[DANCE] Evolving choreography for genre={genre} score={score:.1f}")
            sequence = evolve_choreography(genre, score)

            for face_cmd, body_cmd in sequence:
                self._publish(face_cmd)
                time.sleep(0.15)           # tiny gap so OLED updates first
                self._publish(body_cmd)

                move_time = MOVE_DELAY.get(body_cmd, 1.5)
                time.sleep(move_time)

            # Return to idle
            self._publish("IDLE_MODE")
            print("[DANCE] Choreography complete → IDLE_MODE")

        except Exception as e:
            print(f"[DANCE] Dance thread error: {e}")
            self._publish("IDLE_MODE")

        finally:
            self._dance_lock.release()

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _delayed_idle(self, delay: float):
        """Send IDLE_MODE after `delay` seconds (used after single moves)."""
        time.sleep(delay)
        self._publish("IDLE_MODE")