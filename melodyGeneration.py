import random
import math
from collections import defaultdict


MELODY_DATASET = [

    #happy tunes :)
    {"name": "Sunny Steps", "notes": "123456543", "difficulty": 0.2, "genre": "Happy"},
    {"name": "Twinkle Twinkle", "notes": "123456654321","difficulty": 0.2, "genre": "Happy"},
    {"name": "Jolly Skip","notes": "135135246", "difficulty": 0.3, "genre": "Happy"},
    {"name": "Cheerful Bounce", "notes": "112233445566","difficulty": 0.3, "genre": "Happy"},
    {"name": "Morning Rise", "notes": "234512345", "difficulty": 0.2, "genre": "Happy"},
    {"name": "Playful Hop", "notes": "132435465", "difficulty": 0.4, "genre": "Happy"},

    #sad tunes
    {"name": "Falling Leaves", "notes": "654321123", "difficulty": 0.2, "genre": "Sad"},
    {"name": "Goodbye Waltz", "notes": "665544332211","difficulty": 0.3, "genre": "Sad"},
    {"name": "Rainy Window", "notes": "543216543", "difficulty": 0.2, "genre": "Sad"},
    {"name": "Empty Room", "notes": "654654321", "difficulty": 0.3, "genre": "Sad"},
    {"name": "Mary's Lament", "notes": "32123333222", "difficulty": 0.2, "genre": "Sad"},
    {"name": "Lonely Echo", "notes": "6655443322", "difficulty": 0.3, "genre": "Sad"},

    #angry tunes
    {"name": "Thunder Strike", "notes": "161616161", "difficulty": 0.5, "genre": "Angry"},
    {"name": "Storm March", "notes": "116611661166","difficulty": 0.5, "genre": "Angry"},
    {"name": "Rage Run", "notes": "162534162534","difficulty": 0.6, "genre": "Angry"},
    {"name": "Battle Cry", "notes": "611611166", "difficulty": 0.5, "genre": "Angry"},
    {"name": "Jingle Fury", "notes": "333333353123","difficulty": 0.4, "genre": "Angry"},
    {"name": "Clash of Keys", "notes": "165165612", "difficulty": 0.6, "genre": "Angry"},

    #surprise tunes
    {"name": "Sudden Twist", "notes": "163625142", "difficulty": 0.5, "genre": "Surprise"},
    {"name": "Pop!", "notes": "152634152", "difficulty": 0.4, "genre": "Surprise"},
    {"name": "Magic Trick", "notes": "612345165", "difficulty": 0.5, "genre": "Surprise"},
    {"name": "Zigzag", "notes": "162536142", "difficulty": 0.6, "genre": "Surprise"},
    {"name": "Unexpected Turn", "notes": "135624135", "difficulty": 0.5, "genre": "Surprise"},
    {"name": "The Reveal", "notes": "246135624", "difficulty": 0.6, "genre": "Surprise"},
]

# ── Random name pools per genre ───────────────────────────────────────────────
GENRE_NAMES = {
    "Happy": [
        "Sunshine Stroll", "Golden Hour", "Skipping Stones", "Rooftop Dance",
        "Lemonade Stand", "Kite in the Wind", "Daydream", "Petal Parade",
        "Warm Breeze", "Rainbow Run", "Cherry Pop", "Breezy Afternoon",
        "Firefly Waltz", "Bubble Burst", "Hilltop Jig",
    ],
    "Sad": [
        "Fading Light", "Last Train Home", "Grey Skies", "Broken Strings",
        "Midnight Tears", "Willow Weep", "Lost in Rain", "Hollow Hours",
        "Distant Shore", "Winter Goodbye", "Candle Flicker", "Paper Boats",
        "Quiet Ache", "Dusk Lament", "Echoes of You",
    ],
    "Angry": [
        "Iron Fist", "Thunderclap", "Crashing Waves", "Red Alert",
        "Grinding Gears", "Volcanic Rush", "Defiance", "Breaking Point",
        "Fury Road", "Static Charge", "Blazing Trail", "War Drum",
        "Seething Storm", "Boiling Point", "Steel Resolve",
    ],
    "Surprise": [
        "Plot Twist", "Out of Nowhere", "Wildcard", "Hidden Door",
        "The Ambush", "Blink and Miss", "Curveball", "Vanishing Act",
        "Smoke and Mirrors", "Jack in the Box", "Flip the Script",
        "Catch Me Off Guard", "Twist of Fate", "Blindside", "The Switcheroo",
    ],
}

# Track used names per genre so we don't repeat within a session
_used_names = defaultdict(set)


def pick_genre_name(genre):
    """Pick a random unused name for the genre. Resets if all names are used."""
    pool = GENRE_NAMES.get(genre, [f"{genre} Melody"])
    available = [n for n in pool if n not in _used_names[genre]]
    if not available:
        _used_names[genre].clear()
        available = pool
    name = random.choice(available)
    _used_names[genre].add(name)
    return name


# ── Markov Chain melody generator ────────────────────────────────────────────
class MarkovMelodyGenerator:
    def __init__(self):
        self.transition_table = defaultdict(lambda: defaultdict(int))
        self.starting_notes = []
        self.trained = False

    def train(self, melodies):
        self.transition_table = defaultdict(lambda: defaultdict(int))
        self.starting_notes = []

        for melody in melodies:
            if len(melody) < 2:
                continue
            self.starting_notes.append(melody[0])
            for i in range(len(melody) - 1):
                current_note = melody[i]
                next_note = melody[i + 1]
                self.transition_table[current_note][next_note] += 1

        self.trained = True
        print(f"Trained on {len(melodies)} melodies.")
        print(f"Transition table: { {k: dict(v) for k, v in self.transition_table.items()} }")

    def _pick_next_note(self, current_note):
        options = self.transition_table.get(current_note)
        if not options:
            return random.choice(['1', '2', '3', '4', '5', '6'])

        notes = list(options.keys())
        counts = list(options.values())
        total = sum(counts)

        probabilities = [c / total for c in counts]
        roll = random.random()
        cumulative = 0.0
        for note, prob in zip(notes, probabilities):
            cumulative += prob
            if roll <= cumulative:
                return note

        return notes[-1]

    def generate(self, length=None):
        if not self.trained:
            raise RuntimeError("Call train() before generate().")
        if length is None:
            length = random.randint(8, 16)

        current = random.choice(self.starting_notes) if self.starting_notes else '1'
        result = [current]
        for _ in range(length - 1):
            current = self._pick_next_note(current)
            result.append(current)

        return ''.join(result)


# ── KNN Genre Classifier ──────────────────────────────────────────────────────
class KNNGenreClassifier:
    SUPPORTED_GENRES = ["Happy", "Sad", "Angry", "Surprise"]

    EMOTION_FEATURES = {
        "Happy":    [0.9, 0.8, 0.1, 0.8],
        "Sad":      [0.1, 0.2, 0.2, 0.9],
        "Angry":    [0.2, 0.9, 0.9, 0.5],
        "Fear":     [0.1, 0.7, 0.9, 0.3],
        "Disgust":  [0.2, 0.5, 0.7, 0.6],
        "Surprise": [0.6, 0.8, 0.5, 0.1],
        "Neutral":  [0.5, 0.4, 0.2, 0.7],
    }

    def __init__(self, k=3):
        self.k = k

    def _euclidean_distance(self, vec_a, vec_b):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))

    def classify(self, emotion):
        emotion = emotion.strip().capitalize()

        if emotion in self.SUPPORTED_GENRES:
            print(f"'{emotion}' is directly supported. No mapping needed.")
            return emotion

        if emotion not in self.EMOTION_FEATURES:
            print(f"Unknown emotion '{emotion}'")
            emotion = "Neutral"

        query_vec = self.EMOTION_FEATURES[emotion]
        distances = []
        for genre in self.SUPPORTED_GENRES:
            genre_vec = self.EMOTION_FEATURES[genre]
            dist = self._euclidean_distance(query_vec, genre_vec)
            distances.append((dist, genre))
            print(f"[KNN] Distance from '{emotion}' to '{genre}': {dist:.3f}")

        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        print(f"{self.k} nearest genres: {k_nearest}")

        vote_counts = defaultdict(int)
        for _, genre in k_nearest:
            vote_counts[genre] += 1

        best_genre = k_nearest[0][1]
        best_votes = 0
        for genre, votes in vote_counts.items():
            if votes > best_votes:
                best_votes = votes
                best_genre = genre

        print(f"'{emotion}' mapped to genre: '{best_genre}'")
        return best_genre


# ── Smart Melody Generator (combines Markov + KNN) ────────────────────────────
class SmartMelodyGenerator:
    def __init__(self):
        self.markov     = MarkovMelodyGenerator()
        self.classifier = KNNGenreClassifier(k=3)
        self.dataset    = MELODY_DATASET

    def _get_melodies_by_genre(self, genre):
        return [m["notes"] for m in self.dataset if m["genre"] == genre]

    def _estimate_difficulty(self, notes):
        unique_notes = len(set(notes))
        jumps = 0
        for i in range(len(notes) - 1):
            if abs(int(notes[i + 1]) - int(notes[i])) >= 2:
                jumps += 1
        difficulty = 0.1 + unique_notes * 0.08 + jumps * 0.04
        return round(min(difficulty, 1.0), 2)

    def generate_melody(self, emotion="Happy"):
        genre = self.classifier.classify(emotion)
        print(f"\nGenerating a '{genre}' melody...")

        genre_melodies = self._get_melodies_by_genre(genre)
        if not genre_melodies:
            print(f"No melodies found for genre '{genre}'")
            genre_melodies = [m["notes"] for m in self.dataset]

        self.markov.train(genre_melodies)
        notes = self.markov.generate()

        # Pick a random name from the genre pool instead of "Happy Melody" etc.
        name = pick_genre_name(genre)

        melody = {
            "name": name,
            "notes": notes,
            "genre": genre,
            "difficulty": self._estimate_difficulty(notes)
        }

        print(f"Generated: {melody}")
        return melody


# for testing
if __name__ == "__main__":
    gen = SmartMelodyGenerator()
    for emotion in ["Happy", "Sad", "Angry", "Surprise", "Fear", "Happy", "Happy"]:
        melody = gen.generate_melody(emotion=emotion)
        print(f"  → {melody['name']}\n")