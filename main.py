#all needed libraries 
#import openai
import pandas as pd 
import re
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.clock import Clock
import serial
import serial.tools.list_ports
import threading
import time
from kivy.utils import get_color_from_hex
import ast
import csv
import json
import os
import pandas as pd
from datetime import datetime
from statistics import median 

# Module Imports
#Fatima-> Signing to melody Working 
#Areesh-> Emotion Detection Melody Reccomendation Working (needs to actually show recommended melodies)
#Safa -> Melody Recommendation Working (needs to save melodies with a name their genere, so areesh's module can recommend)
#Fizza -> Detect next user key
#Huzaifa-> Level + Dynamic Difficulty 
# Dynamic difficulty matching (Huzaifa)
from dynamicMatching import (
    Song as DDMSong,
    PerformanceRecord,
    recommend_for_live_session,
    load_history_from_csv as ddm_load_history,
    DEFAULT_LIBRARY as DDM_LIBRARY,
)

# Emotion-based melody recommendation
from emotionRecommendation import (
    MelodyRecommender,
    MELODIES as EMOTION_MELODIES,
    EMOTION_TO_GENRE,
)
# EmotionDetector requires deepface + webcam; import guarded so app still
# runs on machines without a camera / deepface installed.
try:
    from emotionRecommendation import EmotionDetector, FacialRecommendationScreen
    EMOTION_DETECTION_AVAILABLE = True
except Exception as _emotion_import_err:
    EMOTION_DETECTION_AVAILABLE = False
    print(f"[WARN] EmotionDetector not available: {_emotion_import_err}")

# Markov + KNN melody generation (local, no API needed)
from melodyGeneration import SmartMelodyGenerator

# Melody preference / similarity matching
from preference import (
    load_melody_library,
    save_melody_library,
    save_melody_to_library,
    delete_melody_from_library,
    build_full_melody_library,
    levenshtein_distance,
    compute_similarity,
    get_top_matches,
    PredictModeScreen,
    PredictResultScreen,
)

# Humming-to-melody pipeline
# Voice input will come from ESP32 in future; for now the capture and
# recording parts are commented out — only the processing helpers are
# imported so they're available when hardware is connected.
try:
    from hummingToMelody import (
        construct_markov,
        smooth_melody,
        filter_dominant,
        genetic_algorithm,
        extract_raw_melody
    )
    HUMMING_AVAILABLE = True
except Exception as _hum_import_err:
    HUMMING_AVAILABLE = False
    print(f"[WARN] hummingToMelody not fully available: {_hum_import_err}")

#api and other neededinfo
api_key="YOUR_API_KEY"
#3 hardcoded melodies that the user can play at anytime
MELODIES = {
    'twinkle': {'name': 'Twinkle Twinkle',        'notes': '123456654321',    'genre': 'Happy'},
    'mary':    {'name': 'Mary Had a Little Lamb',  'notes': '32123333222455',  'genre': 'Neutral'},
    'jingle':  {'name': 'Jingle Bells',            'notes': '333333353123',    'genre': 'Happy'},
}
# Portable data folder — works on any machine, not just Daniyal's laptop
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_FOLDER = os.path.join(BASE_DIR, "user_data")
os.makedirs(USER_DATA_FOLDER, exist_ok=True)

MELODY_LIBRARY_FILE = os.path.join(BASE_DIR, "melody_library.json")


def load_full_ddm_library():
    """
    Build a combined song library for DDM by merging:
      1. DEFAULT_LIBRARY (hardcoded songs)
      2. melody_library.json (generated + saved melodies)
    Returns a list of DDMSong objects with no duplicates.
    """
    from dynamicMatching import compute_song_difficulty
    library = list(DDM_LIBRARY)
    known_names = {s.name for s in library}
    try:
        if os.path.isfile(MELODY_LIBRARY_FILE):
            with open(MELODY_LIBRARY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    name  = entry.get("name", "")
                    notes = entry.get("notes", "")
                    genre = entry.get("genre", "generated")
                    if name and notes and name not in known_names:
                        library.append(DDMSong(
                            name=name,
                            notes=notes,
                            genre=genre,
                            difficulty=compute_song_difficulty(notes),
                        ))
                        known_names.add(name)
            print(f"[DDM Library] Loaded {len(library)} songs total ({len(library) - len(DDM_LIBRARY)} from melody_library.json)")
    except Exception as e:
        print(f"[DDM Library] Could not load melody_library.json: {e}")
    return library

# ── Player Level System ───────────────────────────────────────────────────────
def _level_file(user_name):
    return os.path.join(USER_DATA_FOLDER, f"{user_name.lower()}_level.json")

def load_player_level(user_name):
    return 0.0

def save_player_level(user_name, level):
    level = round(max(0.0, min(5.0, level)), 3)
    try:
        with open(_level_file(user_name), "w") as f:
            json.dump({"level": level}, f)
    except Exception as e:
        print(f"[Level] Could not save: {e}")
    return level
def update_player_level(user_name, accuracy):
    app = App.get_running_app()

    if accuracy >= 90:
        level = 1
    elif accuracy >= 70:
        level = 0.5
    elif accuracy >= 50:
        level = 0.25
    else:
        level = 0

    angle = int(level * 20)

    app.robot_brain._dance.send_command(f"LEVELMOVE:{angle}")
    print(f"[LEVEL MQTT] accuracy={accuracy}, angle={angle}")

    return 0.0

# Local (offline) melody generator — used as fallback when OpenAI is unavailable
_local_melody_gen = SmartMelodyGenerator()
def send_lcd_message(serial_manager, line1, line2=""):
    #commands the esp32 to dsiplay the message on the oled
    if line2:
        message = f"LCD:{line1}|{line2}"
        print(f"Sending to OLED: '{line1}' | '{line2}'")
    else:
        message = f"LCD:{line1}"
        print(f"Sending to OLED: '{line1}'")
    serial_manager.send_command(message) #sending the command to esp32 to display lcd 

def log_user_performance(user_name, song_name, accuracy, total_notes, correct_notes, mistakes):
    filename = f"{user_name.lower()}.csv"
    filepath = os.path.join(USER_DATA_FOLDER, filename)
    file_exists = os.path.isfile(filepath)
    #when a user finishes a song that data is sent ot their csv file 
    print(f"Logging user data...")
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "song", "accuracy", "total_notes", "correct_notes", "mistakes"])
        writer.writerow([
            datetime.now().isoformat(),
            song_name,
            accuracy,
            total_notes,
            correct_notes,
            mistakes
        ])

def load_ddm_history(user_name):
    """Load PerformanceRecord history for dynamic difficulty matching."""
    filepath = os.path.join(USER_DATA_FOLDER, f"{user_name.lower()}.csv")
    if os.path.isfile(filepath):
        try:
            return ddm_load_history(filepath)
        except Exception as e:
            print(f"[DDM] Could not load history: {e}")
    return []

#all screens code
class MelodySearchScreen(Screen):
    status_text = StringProperty("")

class NameScreen(Screen):
    nameEntered = False #check to avoid re-entering name 
    def submit_name(self):
        user_name = self.ids.name_input.text.strip()
        if user_name:
            self.manager.user_name = user_name
            self.manager.name_entered=True #user has entered the name 
            filename = f"{user_name.lower()}.csv"
            if not os.path.isfile(filename):
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "song", "accuracy", "total_notes", "correct_notes", "mistakes"])
            #after their name has been entered take them to the mode selection 
            self.manager.current = "mode_selection"

#kivy screen manageer 
class MyScreenManager(ScreenManager):
    user_name = StringProperty("")
    name_entered = BooleanProperty(False)

#helper functions to generate songs and response 
#this one generates a response depending on the accuracy 
def get_gpt_feedback(accuracy):
    try:
        prompt = f"A user just played a piano song with {accuracy}% accuracy. Respond with ONLY an emoticon (like ^_^ or T_T) on the first line be more creative with the emoticons,truly express your emotions, and a short 8-word encouragement on the second line. Format: emoticon|message"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an encouraging music teacher. Always respond in the exact format: emoticon|short message"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=50
        )
        result = response.choices[0].message.content.strip()
        #parsing the response 
        if '|' in result:
            emoticon, message = result.split('|', 1)
            return emoticon.strip(), message.strip()
        else:
           #fall back if theres a problem getting a response
            return "^_^", "Keep practicing!"
            
    except Exception as e:
        print(f"Error generating a response...")
        #generic responding 
        if accuracy >= 90:
            return "^_^", "Amazing work!"
        elif accuracy >= 70:
            return "^-^", "Good job!"
        elif accuracy >= 50:
            return "o_o", "Keep trying!"
        else:
            return "T_T", "Practice more!"

class MelodyGenerator:
    #uses api to generate a randommelody
    #real time melody generation
    def __init__(self):
        self.model = "gpt-3.5-turbo" #model and keys 
        openai.api_key = api_key
    
    def generate_melody(self):
        """Generate a random melody using OpenAI API"""
        prompt = (
            "Generate a melody of 6-20 notes to be played on a piano. "
            "The keys go from 1-6. Example: JINGLE BELLS: '222222242012'. "
            "Return ONLY a Python dictionary with this exact format: "
            "{'name': 'Melody Name', 'notes': '123456'} "
            "Return a proper name for the melody too"
            "Do not include any other text, explanation, or code blocks."
        )
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a music composition assistant. Return only valid Python dictionaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  #high temp-> gets greater creativity 
                max_tokens=100
            )
            
            #cleansing and extracting the response 
            result = response.choices[0].message.content.strip()
            result = result.replace("```python", "").replace("```", "").strip()
            melody_dict = ast.literal_eval(result)
            
            #validation procedure
            if 'name' not in melody_dict or 'notes' not in melody_dict:
                raise ValueError("Invalid dictionary format")
            notes = melody_dict['notes']
            if not re.match(r'^[1-6]+$', notes):
                raise ValueError("Notes must only contain digits 1-6")
            if len(notes) > 20:
                melody_dict['notes'] = notes[:20]
            
            print(f"Random Melody Generated : {melody_dict['name']} - {melody_dict['notes']}")
            return melody_dict
            
        except Exception as e:
            print(f"Error generating melody: {e}")
            #fall back if the connection fails or we run out of tokens :p
            return {
                'name': 'Jolly Golly',
                'notes': '135246135246'
            }


class SerialManager:
    #manages the connection between esp32 and python
    #we tried mqtt but it had a bigger delay which was causing issues with fsr detection 
    def __init__(self, app):
        self.app = app
        self.serial_port = None
        self.connected = False
        self.reading_thread = None
        self.running = False
        # Preference / predict mode — captures keys pressed in CASUALMODE
        self.captured_notes = ""
        self.is_capturing = False
    def go_idle(self):
        self.send_command("IDLE")
        send_lcd_message(self, "Ready to Play!", "Select a Mode")

    def list_ports(self):
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    def connect(self, port, baudrate=115200):
        #try connecting with esp32 
        try:
            self.serial_port = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2) 
            self.connected = True
            self.running = True
            self.reading_thread = threading.Thread(target=self.read_serial, daemon=True)
            self.reading_thread.start()
            
            #OLED issues: Basically we were facing an issue where it would die if the ython was reran but not the arduino code
            #so we sent command to display the text in the start 
            time.sleep(1)
            self.send_command("INIT")
            # Reset the physical level indicator to 0 on startup
            self.send_command("LEVELMOVE:0")
            time.sleep(0.5)
            print(f"Connection made with {port}")
            return True
        
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
        
    def read_serial(self):
        #continuously reads from serial 
        while self.running and self.connected:
            try:
                if self.serial_port and self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    if line:
                        self.handle_message(line)
            except Exception as e:
                print(f"Read error: {e}")
                time.sleep(0.1)
    
    def handle_message(self, message):
        # Capture notes during predict mode (preference.py integration)
        # Listens for ESP32 debug prints: "[CASUAL] Key 2 pressed"
        if self.is_capturing and "[CASUAL] Key" in message and "pressed" in message:
            try:
                parts = message.strip().split()
                key_index = int(parts[parts.index("Key") + 1])  # 0-based from ESP32
                note_digit = str(key_index + 1)                 # convert to 1-based
                self.captured_notes += note_digit
                print(f"[CAPTURE] Note: {note_digit} | So far: {self.captured_notes}")
            except (ValueError, IndexError):
                pass

        if ':' in message:
            msg_type, content = message.split(':', 1)
            
            # Schedule UI updates on main thread
            if msg_type == "STATUS":
                Clock.schedule_once(lambda dt: self.app.update_status(content), 0)
            elif msg_type == "ERROR":
                Clock.schedule_once(lambda dt: self.app.show_error(content), 0)
            elif msg_type == "COMPLETION":
                Clock.schedule_once(lambda dt: self.app.handle_completion(content), 0)
            elif msg_type == "FREQDATA":
                print("Received FREQDATA:", content[:100])
                frequencies = [float(f) for f in content.split(",") if f.strip()]
                print("Frequency count:", len(frequencies))
                Clock.schedule_once(lambda dt, f=frequencies: self.app.handle_humming(f), 0)



    def send_command(self, command):
        #the actual bread and butter of our code
        #we basically designed commands that link the python with the esp32
        #frompython we send the commands 
        #these commands are read and performed by the esp32
        #this keeps the esp32 code generic
        if self.connected and self.serial_port:
            try:
                command = command.strip() + '\n'
                self.serial_port.write(command.encode('utf-8'))
                self.serial_port.flush()
                print(f"Command sent: {command.strip()}")
                return True
            except Exception as e:
                print(f"Send error: {e}")
                return False
        else:
            print("Not connected to ESP32")
            return False
    
    def disconnect(self):
        #disconeect when quit resets everything nicely 
        self.running = False
        self.connected = False
        if self.serial_port:
            try:
                self.serial_port.close()
                print("Disconnected")
            except:
                pass

#actual screens 
class HomeScreen(Screen):
    def start_game(self):
        if not self.manager.name_entered: 
            #if name has not been entered then prompt the user to enter their name
            self.manager.current = 'name'
        else:
            #else, they already eneterd the name, then we go to mode sleection
            self.manager.current = 'mode_selection'
    #depending on the buttons pressed go to these screens         
    def show_how_to_play(self):
        self.manager.current = 'howToPlay'
    def show_credits(self):
        self.manager.current = 'credits'

#from this screen the user actuallypins the 
class ModeSelectionScreen(Screen):
    def select_mode(self, mode):
        app = App.get_running_app()
        #differet mode selevtions-> go to different screens 
        #then we also send a text t be displayed on the oled screen 
        if mode == "casual":
            send_lcd_message(app.serial_manager, "Casual Mode", "Select Song!")
            self.manager.current = "casual_mode"
        elif mode == "analysis":
            send_lcd_message(app.serial_manager, "Analysis", "Select Song!")
            self.manager.current = "analysis"
        elif mode == 'beginner':
            send_lcd_message(app.serial_manager, "Beginner Mode", "Select Song!")
            self.manager.current = 'beginner_song'
        elif mode == 'intermediate':
            send_lcd_message(app.serial_manager, "Intermediate", "Select Song!")
            self.manager.current = 'intermediate_song'    
        elif mode == 'AI':
            send_lcd_message(app.serial_manager, "AI Mode", "Select Song!")
            self.manager.current = 'AI_song'    
        elif mode == 'ai_vs':
            send_lcd_message(app.serial_manager, "Competitive", "Select Song!")
            self.manager.current = 'ai_vs_song'
        # ── New modes ────────────────────────────────────────────────────────
        elif mode == 'predict':
            #Preference matching: user plays freely, top-3 closest melodies shown
            send_lcd_message(app.serial_manager, "Predict Melody", "Press Start")
            self.manager.current = 'predict_mode'
        elif mode == 'emotion':
            # Emotion-based recommendation via webcam (requires deepface)
            if EMOTION_DETECTION_AVAILABLE:
                send_lcd_message(app.serial_manager, "Mood Mode", "Look at camera")
                self.manager.current = 'facial_recommendation'
            else:
                app.show_error("Emotion detection not available.\nInstall deepface and opencv.")
       
        
        elif mode == 'humming':
            app.current_mode = 'humming'
            send_lcd_message(app.serial_manager, "Hum Mode", "Start humming!")
            self.manager.current = 'waiting'
            waiting_screen = self.manager.get_screen('waiting')
            waiting_screen.melody_text = "Humming Mode"
            waiting_screen.status_text = "Listening for 10 seconds..."
            app.serial_manager.send_command("STARTFREQ")

    def go_back(self):
        self.manager.current = 'home'
class CasualModeScreen(Screen):
    status_text = StringProperty("Press any piano key to play!")
    
    def on_enter(self):
        #casual is when you can play any key
        #without melody 
        app = App.get_running_app()
        app.serial_manager.send_command("CASUALMODE")
    
    def stop_casual_mode(self):
        app = App.get_running_app()
        app.serial_manager.send_command("STOP")
        self.manager.current = 'home'
    
    def go_back(self):
        self.stop_casual_mode()

class AnalysisScreen(Screen):
    best_song = StringProperty("")
    worst_song = StringProperty("")
    average_accuracy = NumericProperty(0)
    total_songs = NumericProperty(0)
    most_practiced = StringProperty("")
    total_mistakes = NumericProperty(0)
    total_correct = NumericProperty(0)
    average_mistakes = NumericProperty(0)

    def on_enter(self):
        

        user = self.manager.user_name.lower()
        filename = f"{user}.csv"
        filepath = os.path.join(USER_DATA_FOLDER, filename)

        print(f"Looking for file: {filepath}")

        if not os.path.isfile(filepath):
            print("Error: File not found!")
            return

        df = pd.read_csv(filepath)
        print(f"File loaded! Rows: {len(df)}")
        
        if df.empty:
            print("File is empty!")
            return

        #Uupdating the kivy display 
        self.best_song = df.groupby("song")["accuracy"].mean().idxmax()
        self.worst_song = df.groupby("song")["accuracy"].mean().idxmin()
        self.total_songs = int(df["song"].count())
        self.average_accuracy = float(df["accuracy"].mean())
        self.most_practiced = df["song"].value_counts().idxmax()
        self.total_correct = int(df["correct_notes"].sum())
        self.total_mistakes = int(df["mistakes"].sum())
        self.average_mistakes = float(df["mistakes"].mean())

        print("Stats updated...")

    def go_back(self):
        self.manager.current = 'mode_selection'
    def reset_stats(self):
        #resets everything, but doesnt delete the file
        user = self.manager.user_name.lower()
        filename = f"{user}.csv"
        filepath = os.path.join(USER_DATA_FOLDER, filename)
        if not os.path.isfile(filepath):
            print("No file to reset")
            return

        try:
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader) 
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
            
            print(f"Cleared all data rows from: {filepath}")
            
            #Reset all displayed values to 0/empty
            self.best_song = ""
            self.worst_song = ""
            self.average_accuracy = 0
            self.total_songs = 0
            self.most_practiced = ""
            self.total_mistakes = 0
            self.total_correct = 0
            self.average_mistakes = 0
            
        except Exception as e:
            print(f"Error resetting stats: {e}")
def on_enter(self):
    import pandas as pd

    user = self.manager.user_name.lower()
    filename = f"{user}.csv"
    filepath = os.path.join(USER_DATA_FOLDER, filename)

    if not os.path.isfile(filepath):
        self.ids.analysis_label.text = "No data available yet."
        return

    df = pd.read_csv(filepath)
    self.best_song = df.groupby("song")["accuracy"].mean().idxmax()
    self.worst_song = df.groupby("song")["accuracy"].mean().idxmin()
    self.total_songs = df["song"].count()
    self.average_accuracy = df["accuracy"].mean()
    self.most_practiced = df["song"].value_counts().idxmax()

    if "correct_notes" in df.columns and "mistakes" in df.columns:
        self.total_correct = df["correct_notes"].sum()
        self.total_mistakes = df["mistakes"].sum()
        self.average_mistakes = df["mistakes"].mean()
    else:
        self.total_correct = int((df["accuracy"] / 100 * df["total_notes"]).sum())
        self.total_mistakes = int((df["total_notes"] - df["correct_notes"]).sum())
        self.average_mistakes = self.total_mistakes / len(df)

    self.ids.analysis_label.text = (
        f"[b]User:[/b] {user}\n"
        f"[b]Total Plays:[/b] {self.total_songs}\n"
        f"[b]Average Accuracy:[/b] {self.average_accuracy:.2f}%\n"
        f"[b]Best Song:[/b] {self.best_song}\n"
        f"[b]Worst Song:[/b] {self.worst_song}\n"
        f"[b]Most Practiced:[/b] {self.most_practiced}\n"
        f"[b]Total Correct Notes:[/b] {self.total_correct}\n"
        f"[b]Total Mistakes:[/b] {self.total_mistakes}\n"
        f"[b]Average Mistakes:[/b] {self.average_mistakes:.2f}"
    )




class BeginnerSongScreen(Screen):
    def select_song(self, song_key):
        app = App.get_running_app()
        
        if song_key == 'random':
            # Try OpenAI first; fall back to local Markov generator
            app.current_mode = 'beginner'
            try:
                melody_gen = MelodyGenerator()
                melody = melody_gen.generate_melody()
            except Exception:
                print("[INFO] OpenAI unavailable — using local melody generator")
                melody = _local_melody_gen.generate_melody()
            app.current_melody = melody['notes']
            app.current_melody_name = melody['name']
            app.robot_brain.notify_new_melody(genre=melody.get('genre', 'Neutral'))
            app.start_time = None
            # Save generated melody to library for predict/preference matching
            save_melody_to_library(melody['name'], melody['notes'], source='generated')
        else:
            melody = MELODIES[song_key]
            app.current_mode = 'beginner'
            app.current_melody = melody['notes']
            app.current_melody_name = melody['name']
            app.robot_brain.notify_new_melody(genre=melody.get('genre', 'Neutral'))
            app.start_time = None
            
        # Send to ESP32
        app.serial_manager.send_command(f"NOTES:{app.current_melody}")
        time.sleep(0.2)
        app.serial_manager.send_command("USERPLAYS")
        
        # Update waiting screen
        waiting_screen = self.manager.get_screen('waiting')
        waiting_screen.melody_text = f"Playing: {app.current_melody_name}"
        waiting_screen.mode = 'beginner'
        waiting_screen.show_timer = False
        
        self.manager.current = 'waiting'
    
    def go_back(self):
        self.manager.current = 'mode_selection'

class IntermediateSongScreen(Screen):
    def select_song(self, song_key):
        app = App.get_running_app()
        
        if song_key == 'random':
            # Try OpenAI first; fall back to local Markov generator
            app.current_mode = 'intermediate'
            try:
                melody_gen = MelodyGenerator()
                melody = melody_gen.generate_melody()
            except Exception:
                print("[INFO] OpenAI unavailable — using local melody generator")
                melody = _local_melody_gen.generate_melody()
            app.current_melody = melody['notes']
            app.current_melody_name = melody['name']
            app.start_time = None
            app.robot_brain.notify_new_melody(genre=melody.get('genre', 'Neutral'))
            # Save to library for predict/preference matching
            save_melody_to_library(melody['name'], melody['notes'], source='generated')
        else:
        
            melody = MELODIES[song_key]
            app.current_mode = 'intermediate'
            app.current_melody = melody['notes']
            app.current_melody_name = melody['name']
            app.robot_brain.notify_new_melody(genre=melody.get('genre', 'Neutral'))
        
        app.time_limit = len(app.current_melody) * 3
        app.start_time = time.time()
        
        # Send to ESP32
        app.serial_manager.send_command(f"NOTES:{app.current_melody}")
        time.sleep(0.2)
        app.serial_manager.send_command("USERPLAYS")
        
        # Update waiting screen
        waiting_screen = self.manager.get_screen('waiting')
        waiting_screen.melody_text = f"Playing: {app.current_melody_name}"
        waiting_screen.mode = 'intermediate'
        waiting_screen.show_timer = True
        waiting_screen.time_limit = app.time_limit
        waiting_screen.start_timer()
        
        self.manager.current = 'waiting'
    
    def go_back(self):
        self.manager.current = 'mode_selection'

class AIModeSongScreen(Screen):
    def select_song(self, song_key):
        app = App.get_running_app()
    
        if song_key == 'random':
            # Try OpenAI first; fall back to local Markov generator
            app.current_mode = 'AI'
            try:
                melody_gen = MelodyGenerator()
                melody = melody_gen.generate_melody()
            except Exception:
                print("[INFO] OpenAI unavailable — using local melody generator")
                melody = _local_melody_gen.generate_melody()
            app.current_melody = melody['notes']
            app.current_melody_name = melody['name']
            app.robot_brain.notify_new_melody(genre=melody.get('genre', 'Neutral'))  # ← add this
            app.start_time = None
            # Save to library for predict/preference matching
            save_melody_to_library(melody['name'], melody['notes'], source='generated')
        else:
            melody = MELODIES[song_key]
            app.current_mode = 'AI'
            app.current_melody = melody['notes']
            app.current_melody_name = melody['name']
            app.robot_brain.notify_new_melody(genre=melody.get('genre', 'Neutral'))  # ← add this
                    
        app.AI_mode_phase = 'ai'
        app.start_time = None
        
        #Send to ESP32 - AI plays first
        app.serial_manager.send_command(f"NOTES:{app.current_melody}")
        time.sleep(0.2)
        app.serial_manager.send_command("AIPLAYS")
        
        #Update waiting screen
        waiting_screen = self.manager.get_screen('waiting')
        waiting_screen.melody_text = f"AI Playing: {app.current_melody_name}"
        waiting_screen.mode = 'AI'
        waiting_screen.show_timer = False
        #then the user plays
        self.manager.current = 'waiting'
    
    def go_back(self):
        self.manager.current = 'mode_selection'

class AIvsSongScreen(Screen):
    def select_song(self, song_key):
        app = App.get_running_app()
        
        if song_key == 'random':
            label = Label(text='Subscribe to our premium version to access this feature!',
                            color=get_color_from_hex('#FF69B4'))
            popup = Popup(title='Random Mode',
              content=label,
              size_hint=(0.6, 0.3),
              background_color=get_color_from_hex('#222222')) 
            
            popup.open()
            return
        
        melody = MELODIES[song_key]
        app.current_mode = 'ai_vs'
        app.current_melody = melody['notes']
        app.current_melody_name = melody['name']
        app.ai_vs_phase = 'user'
        app.user_time = None
        app.ai_time = None
        app.start_time = time.time()
        
        # Send to ESP32 - User plays first
        app.serial_manager.send_command(f"NOTES:{melody['notes']}")
        time.sleep(0.2)
        app.serial_manager.send_command("USERPLAYS")
        
        # Update waiting screen
        waiting_screen = self.manager.get_screen('waiting')
        waiting_screen.melody_text = f"User Turn: {melody['name']}"
        waiting_screen.mode = 'ai_vs'
        waiting_screen.show_timer = True
        waiting_screen.time_limit = 0
        waiting_screen.start_timer()
        
        self.manager.current = 'waiting'
    
    def go_back(self):
        self.manager.current = 'mode_selection'

class WaitingScreen(Screen):
    status_text = StringProperty("Waiting for ESP32...")
    melody_text = StringProperty("")
    timer_text = StringProperty("Time: 0.0s")
    show_timer = BooleanProperty(False)
    mode = StringProperty("")
    time_limit = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timer_event = None
        self.start_clock = None
    
    def start_timer(self):
        if self.timer_event:
            self.timer_event.cancel()
        self.start_clock = time.time()
        self.timer_event = Clock.schedule_interval(self.update_timer, 0.1)
    
    def update_timer(self, dt):
        elapsed = time.time() - self.start_clock
        
        if self.mode == 'intermediate' and self.time_limit > 0:
            remaining = self.time_limit - elapsed
            if remaining <= 0:
                self.timer_text = "Time's Up!"
                self.cancel_with_timeout()
            else:
                self.timer_text = f"Time Left: {remaining:.1f}s"
        else:
            self.timer_text = f"Time: {elapsed:.1f}s"
    
    def cancel_with_timeout(self):
        if self.timer_event:
            self.timer_event.cancel()
        app = App.get_running_app()
        
        app.serial_manager.send_command("STOP")
        
        result_screen = self.manager.get_screen('result')
        result_screen.result_text = "Time's Up!"
        result_screen.errors = -1
        result_screen.percentage = 0
        result_screen.rating = "Failed"
        self.manager.current = 'result'
    
    def cancel(self):
        if self.timer_event:
            self.timer_event.cancel()
        app = App.get_running_app()
        
        app.serial_manager.send_command("STOP")
        self.manager.current = 'home'
    
    def on_leave(self):
        if self.timer_event:
            self.timer_event.cancel()

class ResultScreen(Screen):
    result_text = StringProperty("Results")
    errors = NumericProperty(0)
    percentage = NumericProperty(0)
    rating = StringProperty("")
    show_comparison = BooleanProperty(False)
    user_time_text = StringProperty("")
    ai_time_text = StringProperty("")
    winner_text = StringProperty("")

    def update_results(self, song_name, accuracy, total_notes, correct_notes, mistakes):
        # updates what is shown on the screen 
        self.percentage = accuracy
        self.errors = mistakes
        self.result_text = f"Accuracy: {accuracy}%\nMistakes: {mistakes}"
        
        # adds in user's performance stats 
        log_user_performance(
            self.manager.user_name,
            song_name,
            accuracy,
            total_notes,
            correct_notes,
            mistakes
        )
    
    def play_again(self):
        self.manager.current = 'mode_selection'
    
    def go_home(self):
        self.manager.current = 'home'

class HowToPlayScreen(Screen):
    def go_back(self):
        self.manager.current = 'home'

class CreditsScreen(Screen):
    def go_back(self):
        self.manager.current = 'home'

class PianoGameApp(App):
    
    def build(self):
        self.serial_manager = SerialManager(self)

        from robot_brain import RobotBrain
        self.robot_brain = RobotBrain(self.serial_manager)
        self.robot_brain.start()
        
        # loads kv file and get the root 
        root = Builder.load_file('pianoBot.kv')
        
        self.current_mode = None
        self.current_melody = ""
        self.current_melody_name = ""
        self.time_limit = 0
        self.start_time = None
        
        self.AI_mode_phase = None
        self.ai_vs_phase = None
        self.user_time = None
        self.ai_time = None

        # ── DDM: history is loaded per-user in handle_completion ─────────────
        self.ddm_history = []  # populated after user name is known

        # ── Preload built-in melodies into shared library once ────────────────
        for key, melody in MELODIES.items():
            save_melody_to_library(melody["name"], melody["notes"], source="built_in")
        
        self.connect_to_esp32()
        
        return root

    def connect_to_esp32(self):
        ports = self.serial_manager.list_ports()
        print(f"Available ports: {ports}")  # add this so you can see what's found
        
        if ports:
            port = ports[0]
            if self.serial_manager.connect(port):
                print(f"Connected to {port}")
            else:
                self.show_connection_popup()
        else:
            self.show_connection_popup()
            
    def show_connection_popup(self):
        """show a dialog box if connection fails"""
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        from kivy.uix.textinput import TextInput
        
        ports = self.serial_manager.list_ports()
        ports_text = "Available Ports:\n" + "\n".join(ports) if ports else "No ports found"
        
        content.add_widget(Label(text=ports_text, size_hint_y=0.4))
        
        port_input = TextInput(
            hint_text='Enter port (e.g., COM3 or /dev/ttyUSB0)',
            multiline=False, 
            size_hint_y=0.2
        )
        content.add_widget(port_input)
        
        popup = Popup(title='ESP32 Connection Required',
                     content=content,
                     size_hint=(0.8, 0.6),
                     auto_dismiss=False)
        
        def do_connect(instance):
            port = port_input.text.strip()
            if port:
                if self.serial_manager.connect(port):
                    popup.dismiss()
                else:
                    port_input.text = "Failed! Please try again.."
            elif ports:
                # trying the first port
                if self.serial_manager.connect(ports[0]):
                    popup.dismiss()
        
        def retry_scan(instance):
            popup.dismiss()
            self.show_connection_popup()
        
        btn_box = BoxLayout(size_hint_y=0.2, spacing=5)
        
        connect_btn = Button(text='Connect')
        connect_btn.bind(on_press=do_connect)
        
        scan_btn = Button(text='Scan Again')
        scan_btn.bind(on_press=retry_scan)
        
        btn_box.add_widget(connect_btn)
        btn_box.add_widget(scan_btn)
        content.add_widget(btn_box)
        
        popup.open()
    
    def update_status(self, message):
        try:
            waiting_screen = self.root.get_screen('waiting')
            waiting_screen.status_text = message  # always update it
        except Exception:
            pass
    def show_error(self, message):
        print(f"Error: {message}")
        popup = Popup(title='Error',
                     content=Label(text=message),
                     size_hint=(0.6, 0.3))
        popup.open()
    
    def handle_completion(self, message):

        """Handle completion based on current mode"""

        print(f"\n ===== COMPLETION RECEIVED =====")
        print(f"Mode: {self.current_mode}")
        print(f"Message: {message}")
        
        if self.current_mode == 'beginner':
            self.handle_beginner_completion(message)
        elif self.current_mode == 'intermediate':
            self.handle_intermediate_completion(message)
        elif self.current_mode == 'AI':
            self.handle_AI_completion(message)
        elif self.current_mode == 'ai_vs':
            self.handle_ai_vs_completion(message)
    
    def handle_beginner_completion(self, message):

        """Beginner: Show mistakes and percentage"""
           
        print("Processing Beginner Mode completion.")
        result_screen = self.root.get_screen('result')
        
        
        errors = 0
        if "Errors:" in message:
            try:
                errors = int(message.split("Errors:")[1].strip())
            except:
                errors = 0
        
        # calculating percentage
        total_notes = len(self.current_melody)
        correct_notes = total_notes - errors
        percentage = (correct_notes / total_notes * 100) if total_notes > 0 else 0
        self.robot_brain._dance.send_face_for_accuracy(round(percentage, 1))
        # get gpt's feedback
        emoticon, feedback = get_gpt_feedback(round(percentage, 1))
        
        # sending the feedback to oled
        send_lcd_message(self.serial_manager, f"{emoticon} {percentage:.0f}%", feedback)
        time.sleep(3)  # shows the feedback for 3 seconds only
        
        result_screen.update_results(
            self.current_melody_name,
            round(percentage, 1),
            total_notes,
            correct_notes,
            errors
        )

        # ── Player Level Update ───────────────────────────────────────────────
        user_name = self.root.user_name if hasattr(self.root, 'user_name') else ""
        if user_name:
            current_level = load_player_level(user_name)
            new_level = update_player_level(user_name, round(percentage, 1))

        # ── Dynamic Difficulty Matching ───────────────────────────────────────
        try:
            if user_name:
                self.ddm_history = load_ddm_history(user_name)

            # Build library from DEFAULT_LIBRARY + melody_library.json
            active_library = load_full_ddm_library()

            # Also make sure the song just played is in there
            if self.current_melody_name not in {s.name for s in active_library}:
                from dynamicMatching import compute_song_difficulty
                active_library.append(DDMSong(
                    name=self.current_melody_name,
                    notes=self.current_melody,
                    genre="generated",
                    difficulty=compute_song_difficulty(self.current_melody),
                ))

            rec, diag = recommend_for_live_session(
                current_song_name=self.current_melody_name,
                song_library=active_library,
                past_history=self.ddm_history,
                latest_accuracy=round(percentage, 1),
                latest_total_notes=total_notes,
                latest_mistakes=errors,
            )
            print(f"\n===== [PLAYER LEVEL] =====")
            print(f"  Accuracy     : {round(percentage, 1)}%")
            print(f"  Player Level : {new_level:.2f} / 5.0")
            print(f"  Next song    : {rec.name}")
            print(f"==========================\n")
            result_screen.rating = f"Try next: {rec.name}"
        except Exception as e:
            print(f"[DDM] Could not compute recommendation: {e}")
        
        result_screen.result_text = f"Song Completed: {self.current_melody_name}"
        result_screen.errors = errors
        result_screen.percentage = round(percentage, 1)
        result_screen.show_comparison = False
        
        print(f"Results - Errors: {errors}, Percentage: {percentage:.1f}%")
        print(f"GPT Feedback: {emoticon} - {feedback}")
        self.root.current = 'result'
        
    def handle_intermediate_completion(self, message):
        """Intermediate: Perfect/Good or cancel if below 85%"""

        print("Processing Intermediate Mode completion.")

        result_screen = self.root.get_screen('result')
        waiting_screen = self.root.get_screen('waiting')
        
        
        if waiting_screen.timer_event:
            waiting_screen.timer_event.cancel()
        
        
        errors = 0
        if "Errors:" in message:
            try:
                errors = int(message.split("Errors:")[1].strip())
            except:
                errors = 0
        
        # calculating the percentage for error
        total_notes = len(self.current_melody)
        correct_notes = total_notes - errors
        percentage = (correct_notes / total_notes * 100) if total_notes > 0 else 0
        self.robot_brain._dance.send_face_for_accuracy(round(percentage, 1))
        # check for the threshold: 85%

        if percentage < 85:
            result_screen.result_text = "You performance is below 85%. So, cancelled!"
            result_screen.errors = errors
            result_screen.percentage = round(percentage, 1)
            result_screen.rating = "Try Again"
            result_screen.show_comparison = False
            print(f"Below threshold! Percentage: {percentage:.1f}%")
        else:
            # check time 
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            
            if elapsed_time > self.time_limit:
                result_screen.rating = "Time Exceeded"
            elif percentage == 100:
                result_screen.rating = "Perfect!"
            else:
                result_screen.rating = "Good!"
            
            result_screen.result_text = f"Song Completed: {self.current_melody_name}"
            result_screen.errors = errors
            result_screen.percentage = round(percentage, 1)
            result_screen.show_comparison = False
            print(f"Results --- Rating: {result_screen.rating}, Percentage: {percentage:.1f}%")
        
        self.root.current = 'result'
    
    def handle_AI_completion(self, message):
        #ai plays and then after its completion the user plays
        print(f"Processing AI Mode completion... Phase: {self.AI_mode_phase}")
        if self.AI_mode_phase == 'ai':
            self.robot_brain._dance.send_face("CALM")
            #AI finished, now user's turn
            print("AI finished! Starting user turn...")
           #now the user will play
            self.AI_mode_phase = 'user'
            waiting_screen = self.root.get_screen('waiting')
            waiting_screen.melody_text = f"Your Turn: {self.current_melody_name}"
            waiting_screen.status_text = "Now you play!"
            time.sleep(0.5)
            self.serial_manager.send_command(f"NOTES:{self.current_melody}")
            time.sleep(0.2)
            self.serial_manager.send_command("USERPLAYS")
            print(f"User turn started - Melody: {self.current_melody}")
            
        elif self.AI_mode_phase == 'user':
            #User finished, show results
            print("User finished! Showing results...")
            result_screen = self.root.get_screen('result')
            errors = 0
            if "Errors:" in message:
                try:
                    errors = int(message.split("Errors:")[1].strip())
                except:
                    errors = 0
            
            total_notes = len(self.current_melody)
            correct_notes = total_notes - errors
            percentage = (correct_notes / total_notes * 100) if total_notes > 0 else 0
            self.robot_brain._dance.send_face_for_accuracy(round(percentage, 1))
            emoticon, feedback = get_gpt_feedback(round(percentage, 1))
            send_lcd_message(self.serial_manager, f"{emoticon} {percentage:.0f}%", feedback)
            time.sleep(3)  # Show for 3 seconds
            
            #logthe user performance, not the ai one
            result_screen.update_results(
                self.current_melody_name,
                round(percentage, 1),
                total_notes,
                correct_notes,
                errors
            )
            result_screen.result_text = f"Song Completed: {self.current_melody_name}"
            result_screen.errors = errors
            result_screen.percentage = round(percentage, 1)
            result_screen.rating = ""
            result_screen.show_comparison = False
            
            print(f"Results - Errors: {errors}, Percentage: {percentage:.1f}%")
            print(f"GPT Feedback: {emoticon} - {feedback}")
            self.root.current = 'result'

    def handle_ai_vs_completion(self, message):
        print(f"Processing AI vs User completion... Phase: {self.ai_vs_phase}")
        waiting_screen = self.root.get_screen('waiting')
        if self.ai_vs_phase == 'user':
            # User finished, record time
            self.user_time = time.time() - self.start_time
            self.ai_vs_phase = 'ai'
            
            print(f"User finished in {self.user_time:.2f}s! AI's turn to play now...")
            
            # Stop user timer
            if waiting_screen.timer_event:
                waiting_screen.timer_event.cancel()
            
            # Now AI's turn
            self.start_time = time.time()
            self.serial_manager.send_command(f"NOTES:{self.current_melody}")
            time.sleep(0.2)
            self.serial_manager.send_command("AIPLAYS")
            
            # Update waiting screen
            waiting_screen.melody_text = f"AI Turn: {self.current_melody_name}"
            waiting_screen.status_text = "AI is playing..."
            waiting_screen.start_timer()
            
        elif self.ai_vs_phase == 'ai':
            # AI finished, record time and compare
            self.ai_time = time.time() - self.start_time
            
            print(f"AI finished in {self.ai_time:.2f}s!")
            print(f"Comparing times - User: {self.user_time:.2f}s vs AI: {self.ai_time:.2f}s")
            
            # Stop AI's timer
            if waiting_screen.timer_event:
                waiting_screen.timer_event.cancel()
            
            result_screen = self.root.get_screen('result')
            result_screen.show_comparison = True
            result_screen.user_time_text = f"Your Time: {self.user_time:.2f}s"
            result_screen.ai_time_text = f"AI Time: {self.ai_time:.2f}s"
            
            if self.user_time < self.ai_time:
                result_screen.winner_text = "You Win!"
                result_screen.rating = "Victory!"

                print("User wins this round!")

            elif self.user_time > self.ai_time:
                result_screen.winner_text = "AI Wins!"
                result_screen.rating = "BOOOO! You loser >:( Try Again!"
                print("AI wins! AI WILL TAKE OVER USSSS")
            else:
                result_screen.winner_text = "It's a tie between you and AI!"
                result_screen.rating = "You did an amazing job :) But you can do better!"
                print("It's a tie!")
            
            result_screen.result_text = f"Race Completed: {self.current_melody_name}"
            result_screen.errors = 0
            result_screen.percentage = 0
            
            self.root.current = 'result'

    def median_filter(self, frequencies, window=5):
        result = []
        for i in range(len(frequencies)):
            window_slice = frequencies[max(0, i-window//2) : i+window//2+1]
            result.append(median(window_slice))
        return result
    def remove_outliers(self, frequencies, threshold=200):
        if len(frequencies) < 3:
            return frequencies
        cleaned = [frequencies[0]]
        for i in range(1, len(frequencies) - 1):
            neighbors_avg = (frequencies[i-1] + frequencies[i+1]) / 2
            if abs(frequencies[i] - neighbors_avg) < threshold:
                cleaned.append(frequencies[i])
        return cleaned

    def handle_humming(self, frequencies):
        if not HUMMING_AVAILABLE:
            self.show_error("Humming module not available.")
            return

        from itertools import groupby
        from hummingToMelody import (
            construct_markov, smooth_melody, filter_dominant,
            genetic_algorithm, closest_note, melodies, remerge, piano_notes
        )

        transition_probs = construct_markov(melodies)

        # remove zeros and unrealistic jumps
        cleaned = []
        prev = None
        for f in frequencies:
            if f < 100:
                continue
            if prev and abs(f - prev) > 250:
                continue
            cleaned.append(f)
            prev = f
        cleaned = self.median_filter(cleaned, window=5)
        cleaned=self.remove_outliers(cleaned)
        raw_melody = [closest_note(f/2) for f in cleaned]

        compressed = [(note, len(list(group))) for note, group in groupby(raw_melody)]
        compressed = filter_dominant(compressed)
        smoothed = smooth_melody(compressed, transition_probs, 0.15)
        #smoothed = remerge(smoothed)
        optimized = genetic_algorithm(
            smoothed,
            transition_probs,
            generations=200,
            pop_size=50,
            mutation_rate=0.4
        )

        print("Optimized Melody:", optimized)

        melody_digits = "".join(
            str(list(piano_notes.keys()).index(n[0]) + 1)
            for n in optimized
        )

        print("Sending humming melody to ESP32:", melody_digits)

        self.serial_manager.send_command("PLAYHUM:" + melody_digits)



    def on_stop(self):

        # cleaning up serial connection 
        print("\nApp is closing. Sending STOP command to the ESP32")
        self.serial_manager.send_command("STOP")
        if hasattr(self, "robot_brain"):
            self.robot_brain.stop()
        self.serial_manager.disconnect()


if __name__ == '__main__':
    PianoGameApp().run()