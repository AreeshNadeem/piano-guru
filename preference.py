# add this feature so the user can:
# Start -> play a melody -> stop -> see the top 3 closest matches.


from difflib import SequenceMatcher
import os
import json
import time
from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty, BooleanProperty, NumericProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
import re

# ============================================================
# replace my hardcoded USER_DATA_FOLDER with this
#    so the code works on other machines too
#    I add this near my existing USER_DATA_FOLDER definition
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_FOLDER = os.path.join(BASE_DIR, "user_data")
os.makedirs(USER_DATA_FOLDER, exist_ok=True)

MELODY_LIBRARY_FILE = os.path.join(BASE_DIR, "melody_library.json")


# ============================================================
# add these helper functions AFTER log_user_performance()
#    so I can save generated/custom melodies for prediction
# ============================================================

def load_melody_library():
    """I load my saved melody library from JSON."""
    if os.path.isfile(MELODY_LIBRARY_FILE):
        try:
            with open(MELODY_LIBRARY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading melody library: {e}")
            return {}
    return {}

def save_melody_library(library):
    """I save my melody library back to JSON."""
    try:
        with open(MELODY_LIBRARY_FILE, "w", encoding="utf-8") as f:
            json.dump(library, f, indent=4)
    except Exception as e:
        print(f"Error saving melody library: {e}")

def save_melody_to_library(name, notes, source="custom"):
    """
    I save one melody into my shared library.
    source can be: built_in / generated / custom
    """
    if not name or not notes:
        return False

    if not re.match(r'^[1-6]+$', notes):
        print(f"Invalid melody notes: {notes}")
        return False

    library = load_melody_library()
    safe_key = name.strip().lower().replace(" ", "_")

    library[safe_key] = {
        "name": name.strip(),
        "notes": notes.strip(),
        "source": source
    }

    save_melody_library(library)
    return True

def delete_melody_from_library(melody_key):
    """I delete one melody from my library."""
    library = load_melody_library()
    if melody_key in library:
        del library[melody_key]
        save_melody_library(library)
        return True
    return False

def build_full_melody_library():
    """
    I combine:
    1. built-in melodies from MELODIES
    2. saved generated/custom melodies from JSON
    """
    from main import MELODIES  # lazy import to avoid circular import
    full_library = {}

    for key, melody in MELODIES.items():
        full_library[f"builtin_{key}"] = {
            "name": melody["name"],
            "notes": melody["notes"],
            "source": "built_in"
        }

    saved = load_melody_library()
    # melody_library.json can be a dict or a list depending on who saved it
    if isinstance(saved, dict):
        for key, melody in saved.items():
            full_library[key] = melody
    elif isinstance(saved, list):
        for melody in saved:
            if isinstance(melody, dict) and "name" in melody and "notes" in melody:
                safe_key = melody["name"].strip().lower().replace(" ", "_")
                full_library[safe_key] = {
                    "name": melody["name"],
                    "notes": melody["notes"],
                    "source": melody.get("genre", "generated")
                }

    return full_library


# ============================================================
# add this similarity engine AFTER the library helpers
#    This is my melody prediction logic
# ============================================================

def levenshtein_distance(s1, s2):
    """I compute standard edit distance between two note strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    return dp[m][n]

def compute_similarity(played, target):
    """
    I combine edit distance + SequenceMatcher.
    I return a score from 0.0 to 100.0
    """
    if not played or not target:
        return 0.0

    max_len = max(len(played), len(target))
    edit_sim = 1.0 - (levenshtein_distance(played, target) / max_len)
    seq_sim = SequenceMatcher(None, played, target).ratio()

    score = (0.6 * edit_sim + 0.4 * seq_sim) * 100.0
    return round(score, 1)

def get_top_matches(played_notes, top_n=3):
    """
    I fixed the old bug here:
    I do NOT pass melody_dict into this function anymore.
    I load the full library internally.
    """
    melody_dict = build_full_melody_library()
    results = []

    for key, melody in melody_dict.items():
        score = compute_similarity(played_notes, melody['notes'])
        results.append({
            "key": key,
            "name": melody['name'],
            "notes": melody['notes'],
            "source": melody.get("source", "unknown"),
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


# ============================================================
# patch SerialManager.__init__()
#    add ONLY these 2 lines inside my existing __init__
# ============================================================
# self.captured_notes = ""
# self.is_capturing = False


# ============================================================
# replace my existing SerialManager.handle_message()
#    with this version so I can capture notes in predict mode
# ============================================================

def handle_message(self, message):
    """
    I capture notes during prediction mode by listening
    to the ESP32 CASUALMODE debug prints:
    [CASUAL] Key 2 pressed
    """

    if self.is_capturing and "[CASUAL] Key" in message and "pressed" in message:
        try:
            parts = message.strip().split()
            key_index = int(parts[parts.index("Key") + 1])   # ESP32 sends 0-based index
            note_digit = str(key_index + 1)                  # I convert it to 1-based note
            self.captured_notes += note_digit
            print(f"[CAPTURE] Note captured: {note_digit} | So far: {self.captured_notes}")
        except (ValueError, IndexError):
            pass

    if ':' in message:
        msg_type, content = message.split(':', 1)

        if msg_type == "STATUS":
            Clock.schedule_once(lambda dt: self.app.update_status(content), 0)
        elif msg_type == "ERROR":
            Clock.schedule_once(lambda dt: self.app.show_error(content), 0)
        elif msg_type == "COMPLETION":
            Clock.schedule_once(lambda dt: self.app.handle_completion(content), 0)


# ============================================================
# add these two new screens AFTER AIvsSongScreen
#    These are my prediction mode screens
# ============================================================
class PredictModeScreen(Screen):
    status_text = StringProperty("Press Start, then play your melody!")
    captured_notes = StringProperty("")
    is_recording = BooleanProperty(False)

    def on_enter(self):
        # I reset everything when I enter the screen
        self.captured_notes = ""
        self.status_text = "Press Start, then play your melody!"
        self.is_recording = False

        app = App.get_running_app()
        app.serial_manager.captured_notes = ""
        app.serial_manager.is_capturing = False

    def start_recording(self):
        app = App.get_running_app()
        from main import send_lcd_message

        self.captured_notes = ""
        app.serial_manager.captured_notes = ""
        app.serial_manager.is_capturing = True
        self.is_recording = True

        self.status_text = "Recording... play your melody now!"
        send_lcd_message(app.serial_manager, "Predict Mode", "Play now!")
        app.serial_manager.send_command("CASUALMODE")

    def stop_recording(self):
        app = App.get_running_app()
        from main import send_lcd_message

        app.serial_manager.is_capturing = False
        app.serial_manager.send_command("STOP")
        self.is_recording = False

        captured = app.serial_manager.captured_notes.strip()

        if not captured:
            self.status_text = "No notes captured. Try again!"
            send_lcd_message(app.serial_manager, "No notes", "Try again!")
            return

        self.captured_notes = captured
        self.status_text = f"Captured: {captured} | Finding matches..."
        send_lcd_message(app.serial_manager, "Analysing...", "")

        # I call my fixed get_top_matches() here
        top_matches = get_top_matches(captured, top_n=3)

        result_screen = self.manager.get_screen('predict_result')
        result_screen.played_notes = captured

        result_screen.match1_name = top_matches[0]["name"] if len(top_matches) > 0 else ""
        result_screen.match1_score = top_matches[0]["score"] if len(top_matches) > 0 else 0
        result_screen.match1_source = top_matches[0]["source"] if len(top_matches) > 0 else ""

        result_screen.match2_name = top_matches[1]["name"] if len(top_matches) > 1 else ""
        result_screen.match2_score = top_matches[1]["score"] if len(top_matches) > 1 else 0
        result_screen.match2_source = top_matches[1]["source"] if len(top_matches) > 1 else ""

        result_screen.match3_name = top_matches[2]["name"] if len(top_matches) > 2 else ""
        result_screen.match3_score = top_matches[2]["score"] if len(top_matches) > 2 else 0
        result_screen.match3_source = top_matches[2]["source"] if len(top_matches) > 2 else ""

        self.manager.current = 'predict_result'

    def save_as_custom(self):
        # I let the user save what they played as a custom melody
        app = App.get_running_app()
        captured = app.serial_manager.captured_notes.strip()

        if not captured:
            self.status_text = "Nothing to save yet!"
            return

        custom_name = f"Custom Melody {int(time.time())}"
        ok = save_melody_to_library(custom_name, captured, source="custom")

        if ok:
            self.status_text = f"Saved as {custom_name}"
        else:
            self.status_text = "Could not save melody."

    def show_library_popup(self):
        # I show my saved melody library here
        library = load_melody_library()

        content = BoxLayout(orientation='vertical', spacing=8, padding=10)

        if not library:
            content.add_widget(Label(text="No saved melodies yet."))
        else:
            rows = []

            for melody_key, melody in library.items():
                row = BoxLayout(size_hint_y=None, height=40, spacing=5)

                info = Label(
                    text=f"{melody['name']} ({melody.get('source', 'unknown')}) : {melody['notes']}",
                    halign='left'
                )

                delete_btn = Button(text="Delete", size_hint_x=0.25)

                row.add_widget(info)
                row.add_widget(delete_btn)
                content.add_widget(row)

                rows.append((melody_key, delete_btn))

        close_btn = Button(text="Close", size_hint_y=None, height=45)
        content.add_widget(close_btn)

        popup = Popup(
            title="Saved Melody Library",
            content=content,
            size_hint=(0.85, 0.75)
        )

        # I use a closure here so each delete button deletes the correct melody
        if library:
            for melody_key, delete_btn in rows:
                def make_delete(k, p):
                    def do_delete(instance):
                        delete_melody_from_library(k)
                        p.dismiss()
                        self.show_library_popup()
                    return do_delete

                delete_btn.bind(on_press=make_delete(melody_key, popup))

        close_btn.bind(on_press=popup.dismiss)
        popup.open()

    def go_back(self):
        # I safely stop capture before leaving
        app = App.get_running_app()
        app.serial_manager.is_capturing = False
        app.serial_manager.send_command("STOP")
        self.manager.current = 'mode_selection'


class PredictResultScreen(Screen):
    played_notes = StringProperty("")

    match1_name = StringProperty("")
    match1_score = NumericProperty(0)
    match1_source = StringProperty("")

    match2_name = StringProperty("")
    match2_score = NumericProperty(0)
    match2_source = StringProperty("")

    match3_name = StringProperty("")
    match3_score = NumericProperty(0)
    match3_source = StringProperty("")

    def go_back(self):
        self.manager.current = 'predict_mode'

    def try_again(self):
        self.manager.current = 'predict_mode'


# ============================================================
# I update ModeSelectionScreen.select_mode()
#    I add ONLY this new branch inside the existing function
# ============================================================
# elif mode == 'predict':
#     send_lcd_message(app.serial_manager, "Predict Melody", "Press Start")
#     self.manager.current = 'predict_mode'


# ============================================================
# I fix BeginnerSongScreen random melody saving
#    I add ONLY this line inside the random branch
# ============================================================
# save_melody_to_library(melody['name'], melody['notes'], source='generated')


# ============================================================
# I fix IntermediateSongScreen random melody saving
#     I add ONLY this line inside the random branch
# ============================================================
# save_melody_to_library(melody['name'], melody['notes'], source='generated')


# ============================================================
# I fix AIModeSongScreen random melody saving
#     I add ONLY this line inside the random branch
# ============================================================
# save_melody_to_library(melody['name'], melody['notes'], source='generated')


# ============================================================
# I fix AnalysisScreen bug
#     I delete the duplicate floating on_enter(self) outside the class
#     and keep only the properly indented on_enter() inside AnalysisScreen
# ============================================================


# ============================================================
# I optionally preload built-in melodies once in build()
#     I add this inside PianoGameApp.build()
# ============================================================
# for key, melody in MELODIES.items():
#     save_melody_to_library(melody["name"], melody["notes"], source="built_in")