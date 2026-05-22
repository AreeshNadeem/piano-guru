"""
facial_recommendation.py  —  Piano Guru AI Extension
=====================================================
Algorithms integrated:
  1. MelodyRecommender        — weighted genre scoring (baseline / cold-start)
  2. KNNRecommender           — K-Nearest Neighbours (learns from sessions)

Fixes applied over previous version:
  - FacialRecommendationScreen now uses KNNRecommender (was still on baseline)
  - _finish_scan passes session_stats to KNN predict()
  - n_neighbors capped dynamically to avoid crash when samples < 3
  - predict() guards against single-class training data
  - KNeighborsClassifier rebuilt each fit so n_neighbors stays valid
  - Removed unused imports (MLPClassifier, IsolationForest, dtw, pyttsx3, openai)
  - record_feedback stores session_stats alongside feature for traceability
"""

import time
import threading
import numpy as np
import cv2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

try:
    from kivy.uix.screenmanager import Screen
    from kivy.properties import StringProperty, ListProperty, NumericProperty
    from kivy.clock import Clock
    from kivy.app import App
    KIVY_AVAILABLE = True
except ImportError:
    KIVY_AVAILABLE = False
    class Screen:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CATALOGUES
# ══════════════════════════════════════════════════════════════════════════════

EMOTION_TO_GENRE = {
    "happy":    ["upbeat", "playful", "joyful"],
    "sad":      ["melancholic", "slow", "gentle"],
    "angry":    ["intense", "dramatic", "fast"],
    "fear":     ["tense", "mysterious", "dramatic"],
    "disgust":  ["intense", "slow", "melancholic"],
    "surprise": ["playful", "upbeat", "fast"],
    "neutral":  ["calm", "gentle", "classical"],
}

MELODIES = {
    "twinkle":      {"name": "Twinkle Twinkle",        "notes": "112345654321",       "difficulty": 0.2, "genre": ["joyful","playful","gentle","classical"]},
    "mary":         {"name": "Mary Had a Little Lamb",  "notes": "32123333222455",     "difficulty": 0.3, "genre": ["gentle","calm","classical","playful"]},
    "jingle":       {"name": "Jingle Bells",            "notes": "333333353123",       "difficulty": 0.4, "genre": ["upbeat","joyful","fast","playful"]},
    "ode_to_joy":   {"name": "Ode to Joy",              "notes": "334554321123222",    "difficulty": 0.4, "genre": ["joyful","upbeat","classical","calm"]},
    "lullaby":      {"name": "Brahms Lullaby",          "notes": "135531356531",       "difficulty": 0.3, "genre": ["gentle","slow","melancholic","calm"]},
    "dramatic_run": {"name": "Dramatic Run",            "notes": "123456654321123456", "difficulty": 0.7, "genre": ["dramatic","intense","fast"]},
    "mystery_walk": {"name": "Mystery Walk",            "notes": "135246135",          "difficulty": 0.5, "genre": ["mysterious","tense","slow"]},
    "happy_skip":   {"name": "Happy Skip",              "notes": "1325436512",         "difficulty": 0.4, "genre": ["upbeat","playful","joyful","fast"]},
    "gentle_rain":  {"name": "Gentle Rain",             "notes": "12321232",           "difficulty": 0.2, "genre": ["calm","gentle","melancholic"]},
    "jolly_golly":  {"name": "Jolly Golly",             "notes": "135246135246",       "difficulty": 0.5, "genre": ["playful","upbeat"]},
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. EMOTION DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class EmotionDetector:
    def __init__(self):
        from deepface import DeepFace
        self.DeepFace = DeepFace
        print("[EmotionDetector] Ready.")

    def detect(self, frame):
        try:
            result     = self.DeepFace.analyze(frame, actions=["emotion"],
                                               enforce_detection=False, silent=True)
            face       = result[0]
            raw_scores = face["emotion"]
            total      = sum(raw_scores.values())
            scores     = {k: v / total for k, v in raw_scores.items()}
            dominant   = max(scores, key=scores.get)
            return {"dominant_emotion": dominant, "scores": scores, "face_found": True}
        except Exception:
            return {"dominant_emotion": "neutral",
                    "scores": {e: 0.0 for e in EMOTION_TO_GENRE},
                    "face_found": False}

    def smooth_emotions(self, history):
        if not history:
            return {e: 0.0 for e in EMOTION_TO_GENRE}
        averaged = {e: 0.0 for e in EMOTION_TO_GENRE}
        for scores in history:
            for emotion, val in scores.items():
                averaged[emotion] = averaged.get(emotion, 0.0) + val
        return {e: v / len(history) for e, v in averaged.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 2. BASELINE MELODY RECOMMENDER (genre scoring)
# ══════════════════════════════════════════════════════════════════════════════

class MelodyRecommender:
    def __init__(self, melodies=None):
        self.melodies = melodies or MELODIES

    def emotions_to_genre_weights(self, emotion_scores):
        genre_weights = {}
        for emotion, prob in emotion_scores.items():
            for genre in EMOTION_TO_GENRE.get(emotion, []):
                genre_weights[genre] = genre_weights.get(genre, 0.0) + prob
        if genre_weights:
            max_w = max(genre_weights.values())
            if max_w > 0:
                genre_weights = {g: w / max_w for g, w in genre_weights.items()}
        return genre_weights

    def score_melody(self, melody_dict, genre_weights):
        tags = melody_dict.get("genre", [])
        if not tags:
            return 0.0
        return sum(genre_weights.get(t, 0.0) for t in tags) / len(tags)

    def recommend(self, emotion_scores, top_n=3):
        genre_weights = self.emotions_to_genre_weights(emotion_scores)
        dominant      = max(emotion_scores, key=emotion_scores.get)
        scored = []
        for key, melody in self.melodies.items():
            s       = self.score_melody(melody, genre_weights)
            matched = [g for g in melody.get("genre", []) if genre_weights.get(g, 0) > 0.1]
            reason  = (f"Matches your {dominant} mood — {', '.join(matched)}"
                       if matched else "General recommendation")
            scored.append({"key": key, "name": melody["name"], "notes": melody["notes"],
                           "difficulty": melody["difficulty"], "genre": melody["genre"],
                           "score": round(s, 3), "reason": reason})
        scored.sort(key=lambda x: (-x["score"], x["difficulty"]))
        return scored[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# 3. KNN RECOMMENDER (learns from user feedback)
# ══════════════════════════════════════════════════════════════════════════════

class KNNRecommender:
    """
    K-Nearest Neighbours recommender.

    Feature vector (10-dim):
      [happy, sad, angry, fear, disgust, surprise, neutral,
       accuracy/100, avg_mistakes/10, difficulty]

    Learns which songs each user enjoys based on their emotional
    state + performance history. Falls back to baseline when there
    are fewer than MIN_SAMPLES_TO_PREDICT sessions recorded OR when
    training data contains fewer than 2 unique classes (would crash
    predict_proba).

    n_neighbors is rebuilt dynamically on every fit so it never
    exceeds the number of training samples.
    """

    MIN_SAMPLES_TO_PREDICT = 5

    def __init__(self):
        self.scaler     = StandardScaler()
        self.model      = None           # built lazily in record_feedback
        self.X_train    = []
        self.y_train    = []
        self.is_trained = False
        self._baseline  = MelodyRecommender()
        print("[KNNRecommender] Ready. Needs", self.MIN_SAMPLES_TO_PREDICT, "samples to activate.")

    # ── feature engineering ──────────────────────────────────────────────────

    def _build_feature(self, emotion_scores, session_stats):
        emotions    = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"]
        emotion_vec = [emotion_scores.get(e, 0.0) for e in emotions]
        session_vec = [
            session_stats.get("accuracy",     50.0) / 100.0,
            session_stats.get("avg_mistakes",  0.0) / 10.0,
            session_stats.get("difficulty",    0.5),
        ]
        return emotion_vec + session_vec

    # ── training ─────────────────────────────────────────────────────────────

    def record_feedback(self, emotion_scores, session_stats, melody_key):
        """
        Call this after a user finishes a song to record their
        emotion state, performance, and the song they played.
        """
        feature = self._build_feature(emotion_scores, session_stats)
        self.X_train.append(feature)
        self.y_train.append(melody_key)

        unique_classes = len(set(self.y_train))

        if len(self.X_train) >= self.MIN_SAMPLES_TO_PREDICT and unique_classes >= 2:
            # Cap n_neighbors so it never exceeds sample count
            k = min(3, len(self.X_train))

            self.model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            X_scaled   = self.scaler.fit_transform(np.array(self.X_train))
            self.model.fit(X_scaled, self.y_train)
            self.is_trained = True
            print(f"[KNNRecommender] Trained — {len(self.X_train)} samples, "
                  f"{unique_classes} classes, k={k}.")
        else:
            remaining = max(0, self.MIN_SAMPLES_TO_PREDICT - len(self.X_train))
            print(f"[KNNRecommender] Collecting data — "
                  f"{len(self.X_train)} samples so far "
                  f"({remaining} more needed, {unique_classes} unique class(es)).")

    # ── inference ────────────────────────────────────────────────────────────

    def predict(self, emotion_scores, session_stats, top_n=3):
        """
        Return top_n melody recommendations.
        Falls back to baseline during cold-start or single-class situations.
        """
        # Guard: not enough data or all labels are the same class
        if not self.is_trained or len(set(self.y_train)) < 2:
            print("[KNNRecommender] Cold start — using baseline.")
            return self._baseline.recommend(emotion_scores, top_n=top_n)

        feature  = self._build_feature(emotion_scores, session_stats)
        X_scaled = self.scaler.transform([feature])
        proba    = self.model.predict_proba(X_scaled)[0]
        ranked   = sorted(zip(self.model.classes_, proba), key=lambda x: -x[1])
        dominant = max(emotion_scores, key=emotion_scores.get)

        results = []
        for melody_key, confidence in ranked:
            if melody_key not in MELODIES or len(results) >= top_n:
                continue
            melody = MELODIES[melody_key]
            results.append({
                "key":        melody_key,
                "name":       melody["name"],
                "notes":      melody["notes"],
                "difficulty": melody["difficulty"],
                "genre":      melody["genre"],
                "score":      round(float(confidence), 3),
                "reason":     (f"KNN recommends for your {dominant} mood "
                               f"({confidence * 100:.0f}% confidence)"),
            })

        # Pad with baseline results if KNN returned fewer than top_n
        if len(results) < top_n:
            existing = {r["key"] for r in results}
            for r in self._baseline.recommend(emotion_scores, top_n=top_n):
                if r["key"] not in existing:
                    results.append(r)
                if len(results) == top_n:
                    break

        return results[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# 4. KIVY SCREEN
# ══════════════════════════════════════════════════════════════════════════════

if KIVY_AVAILABLE:

    class FacialRecommendationScreen(Screen):

        emotion_text    = StringProperty("Point your face at the camera...")
        status_text     = StringProperty("Scanning...")
        recommendations = ListProperty([])
        scan_progress   = NumericProperty(0)

        SCAN_FRAMES   = 200
        SMOOTH_WINDOW = 10

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            try:
                self._detector = EmotionDetector()
            except Exception as e:
                print(f"[WARN] EmotionDetector not available: {e}")
                self._detector = None

            # FIX: use KNNRecommender so the ML algorithm is actually active
            self._recommender = KNNRecommender()
            self._camera      = None
            self._running     = False
            self._history     = []
            self._thread      = None
            self._clock_event = None
            # Store last smoothed emotions so select_recommendation can log feedback
            self._last_smoothed_emotions = {}

        # ── screen lifecycle ─────────────────────────────────────────────────

        def on_enter(self):
            self.emotion_text    = "Point your face at the camera..."
            self.status_text     = "Scanning..."
            self.recommendations = []
            self.scan_progress   = 0
            self._history        = []
            self._running        = True
            self._thread = threading.Thread(target=self._camera_loop, daemon=True)
            self._thread.start()

        def on_leave(self):
            self._stop_camera()

        def _stop_camera(self):
            self._running = False
            if self._camera:
                self._camera.release()
                self._camera = None
            cv2.destroyAllWindows()
            if self._clock_event:
                self._clock_event.cancel()
                self._clock_event = None

        # ── camera / detection loop (background thread) ──────────────────────

        def _camera_loop(self):

            if self._detector is None:
                Clock.schedule_once(
                    lambda dt: self._set_error("DeepFace not installed. Emotion detection unavailable."), 0
            )
                return
        
            self._camera = cv2.VideoCapture(0)
            #"http://192.168.0.110:81/stream", cv2.CAP_FFMPEG
            if not self._camera.isOpened():
                Clock.schedule_once(
                    lambda dt: self._set_error("ESP32-CAM stream not found. Check IP."), 0
                )
                return

            frames_collected = 0

            while self._running and frames_collected < self.SCAN_FRAMES:
                ret, frame = self._camera.read()
                if not ret:
                    continue

                cv2.imshow("Piano Guru — Scanning your mood...", frame)
                cv2.waitKey(1)

                result = self._detector.detect(frame)

                if result["face_found"]:
                    self._history.append(result["scores"])
                    print(f"Frame {frames_collected + 1}/{self.SCAN_FRAMES} | "
                          f"Dominant: {result['dominant_emotion'].upper():<10} | "
                          f"Scores: { {e: round(v, 2) for e, v in result['scores'].items()} }")

                    if len(self._history) > self.SMOOTH_WINDOW:
                        self._history.pop(0)
                    frames_collected += 1

                    dominant = result["dominant_emotion"]
                    progress = int((frames_collected / self.SCAN_FRAMES) * 100)

                    Clock.schedule_once(
                        lambda dt, d=dominant, p=progress: self._update_live(d, p), 0
                    )
                else:
                    Clock.schedule_once(
                        lambda dt: setattr(self, "emotion_text",
                                           "No face detected — move closer"), 0
                    )

                time.sleep(0.05)

            if self._running:
                Clock.schedule_once(lambda dt: self._finish_scan(), 0)

        # ── UI updates (main thread) ─────────────────────────────────────────

        def _update_live(self, dominant_emotion, progress):
            emoji_map = {
                "happy":    "😊 Happy",
                "sad":      "😢 Sad",
                "angry":    "😠 Angry",
                "fear":     "😨 Fearful",
                "disgust":  "🤢 Disgusted",
                "surprise": "😲 Surprised",
                "neutral":  "😐 Neutral",
            }
            frames_remaining  = self.SCAN_FRAMES - (progress * self.SCAN_FRAMES // 100)
            seconds_remaining = frames_remaining // 20

            self.emotion_text  = emoji_map.get(dominant_emotion, dominant_emotion.title())
            self.scan_progress = progress
            self.status_text   = (f"Scanning... {seconds_remaining}s left"
                                  if seconds_remaining > 0 else "Almost done...")

        def _finish_scan(self):
            self._stop_camera()

            if not self._history:
                self._set_error("Could not detect a face. Please try again.")
                return

            smoothed = self._detector.smooth_emotions(self._history)
            dominant = max(smoothed, key=smoothed.get)

            self._last_smoothed_emotions = smoothed

            session_stats = {"accuracy": 50.0, "avg_mistakes": 0.0, "difficulty": 0.5}
            recs = self._recommender.predict(smoothed, session_stats, top_n=3)

            self.recommendations = recs

            emoji_map = {
                "happy": "😊", "sad": "😢", "angry": "😠",
                "fear": "😨", "disgust": "🤢", "surprise": "😲", "neutral": "😐",
            }
            self.emotion_text  = f"{emoji_map.get(dominant, '')} {dominant.title()} — here are your songs:"
            self.status_text   = "Tap a song to play it!"
            self.scan_progress = 100

            # ── populate song buttons ──────────────────────────────────
            Clock.schedule_once(lambda dt: self._populate_songs(recs), 0.1)

        def _populate_songs(self, recs):
            from kivy.uix.button import Button

            container = self.ids.get('rec_container')
            if container is None:
                print("[WARN] rec_container not found in ids")
                return

            container.clear_widgets()

            for rec in recs:
                btn = Button(
                    text=f"{rec['name']}\n{rec['reason']}",
                    font_name="MightySouly-lxggD.ttf",
                    font_size=22,
                    size_hint_y=None,
                    height=90,
                    halign='center',
                    valign='middle',
                    background_normal="",
                    background_color=(0, 0, 0, 1),
                    color=(1, 1, 1, 1),
                )
                btn.bind(texture_size=lambda b, v: setattr(b, 'text_size', (b.width, None)))
                btn.bind(on_press=lambda x, r=rec: self.select_recommendation(r))
                container.add_widget(btn)

        def _set_error(self, msg):
            self.status_text  = msg
            self.emotion_text = "Error"

        # ── song selection & KNN feedback ────────────────────────────────────

        def select_recommendation(self, melody_dict):
            """
            Called when the user taps a recommended song.
            Records feedback to KNN so it learns from this choice,
            then navigates to the waiting/play screen.
            """
            # Record this selection as a positive training signal for KNN
            session_stats = {
                "accuracy":     50.0,
                "avg_mistakes": 0.0,
                "difficulty":   melody_dict.get("difficulty", 0.5),
            }
            self._recommender.record_feedback(
                self._last_smoothed_emotions,
                session_stats,
                melody_dict["key"]
            )

            app = App.get_running_app()
            app.current_mode        = "beginner"
            app.current_melody      = melody_dict["notes"]
            app.current_melody_name = melody_dict["name"]
            app.start_time          = None

            app.serial_manager.send_command(f"NOTES:{app.current_melody}")
            time.sleep(0.2)
            app.serial_manager.send_command("USERPLAYS")

            waiting_screen             = self.manager.get_screen("waiting")
            waiting_screen.melody_text = f"Playing: {app.current_melody_name}"
            waiting_screen.mode        = "beginner"
            waiting_screen.show_timer  = False
            self.manager.current       = "waiting"

        # ── navigation ───────────────────────────────────────────────────────

        def rescan(self):
            self.on_enter()

        def go_back(self):
            self._stop_camera()
            self.manager.current = "mode_selection"