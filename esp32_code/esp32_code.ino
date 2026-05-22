//all necesaary libaraies 
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ESP32Servo.h>
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C
//pin configuration
// Audio output now goes to MAX98357A amplifier instead of passive buzzer
// tone() still works the same way — just wired to AMP_PIN instead
const int AMP_PIN  = 26;   // I2S / PWM audio out -> MAX98357A amplifier input
const int buzzer   = AMP_PIN; // alias kept so existing tone() calls need no changes

// MAX9814 Microphone Module
// OUT -> ADC1_CH0 (GPIO 36, input-only, no pinMode needed)
// AR  -> leave floating for auto-gain (or tie to 3.3 V for 40 dB fixed gain)
// GND / VDD as normal
const int MIC_PIN  = 27;   // MAX9814 analog output
const int MIC_SAMPLES     = 512;   // samples per recording window
const int MIC_SAMPLE_RATE = 8000;  // Hz — good enough for pitch detection
const int MIC_SILENCE_THRESHOLD = 100; //2 ADC counts above mid-rail = sound

// ESP32-CAM — camera is on its own module sharing the same serial bus.
// Python sends STARTCAM / STOPCAM; ESP32 signals back CAMREADY / CAMDATA:<base64>
// The actual JPEG capture uses esp_camera; pin map below is for AI-Thinker module.

const int NUM_KEYS = 6;
const int SDA_PIN  = 21;
const int SCL_PIN  = 22;
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
const int startupMelody[] = {262, 294, 330, 349, 392, 440}; // C, D, E, F, G, A
const int startupDurations[] = {150, 150, 150, 150, 200, 300}; // Duration in ms
const int startupLength = 6;
const int ledGuidance[6] = {23, 14 , 19, 18, 17, 5};
//red led,goes bright when an error is mad 
const int ledMis = 15;
const int fsrPins[6] = {32, 33, 34, 35, 36, 39};
const int fsrThresh = 1000;
const int musicalNotes[6] = {262, 294, 330, 349, 392, 440}; 
//boolean states 
bool isOpen = false;
unsigned long rotationStartTime = 0;
bool timerActive = false;
bool freqRecording = false;
unsigned long lastSampleTime = 0;
int freqSamples = 0;
const int TOTAL_SAMPLES = 100; // 10 seconds
String freqBuffer = "";
//SERVO 
Servo levelServo;
const int SERVO_PIN=12;
int currentServoAngle = 0;

void initLevelServo() {
    // IMPORTANT: full project uses tone()/LEDC too, so force a stable servo setup
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);
    levelServo.setPeriodHertz(50);
    if (!levelServo.attached()) {
        levelServo.attach(SERVO_PIN, 500, 2400);
        delay(50);
    }
}

void moveLevelServo(int angle) {
    angle = constrain(angle, 0, 180);
    initLevelServo(); // reattach in case tone()/other LEDC use disturbed PWM
    currentServoAngle = angle;
    levelServo.write(angle);
    delay(250); // give the servo time to physically move
    Serial.println("[ESP32 LEVEL] Servo moved to " + String(angle));
    Serial.println("LEVEL:" + String(angle));
    Serial.flush();
}

//variables for piano related melodies 
String melody = "";
int melodyLength = 0;
int currentNoteIndex = 0;
int errorCount = 0;
//different modes
//idle mode is basically when nothing is going on 
//user mode when the user gets to play the piano
//ai mode is when ai gets to play the piano
//piano mode is withoutmelodies
//so its like an actual piano
enum PlayMode { IDLE, USER_PLAYING, AI_PLAYING, CASUAL_MODE};
PlayMode playMode = IDLE;
//notes states 
//since were dealing with fsrs and they are aanalogue we need to be careful about how we read thenaalogue values
//basically we play a tone when a key is pressed and wait for it bereleased, ie when the force detcted goes to 0
enum NoteState {
    SHOW_LED,           //LED for expected key
    WAIT_FOR_PRESS,     //Wait for key press
    WAIT_FOR_RELEASE    //Wait for key release
};
NoteState noteState = SHOW_LED;
//error checkks
//only counting one error per one wrong key presedm we dont want to count multiple errors ehrna single mistake is made 
bool errorAlreadyCounted = false;
unsigned long lastNoteTime = 0;
const unsigned long noteInterval = 500;
String serialBuffer = "";

//setups and hardware initialization 
//starting up sound 
void playStartupSound() {
    Serial.println("playign welcome melody....");
    for (int i = 0; i < startupLength; i++) {
        digitalWrite(ledGuidance[i], HIGH);
        tone(buzzer, startupMelody[i], startupDurations[i]);
        delay(startupDurations[i]);
        digitalWrite(ledGuidance[i], LOW);
        delay(50);
    }
    
    //flasing all leds in the end, helps with debugging to see if some led  has an issue 
    for (int i = 0; i < NUM_KEYS; i++) {
        digitalWrite(ledGuidance[i], HIGH);
    }
    delay(200);
    for (int i = 0; i < NUM_KEYS; i++) {
        digitalWrite(ledGuidance[i], LOW);
    }
    //turn them off too  
    Serial.println("[STARTUP] Welcome melody complete!");
}


//animation for when the user is playing the piano
//small animation that makes the face fly
//since were usign a4 pin OLED it fits the nuumber of pins we have left 
void drawCuteFace(int yOffset) {
    display.clearDisplay();
    int centerX = 64;
    int centerY = 32 + yOffset;
    display.drawCircle(centerX, centerY, 20, SSD1306_WHITE);
    
    // Draw eyes (^_^)
    //left eye
    display.drawLine(centerX - 10, centerY - 5, centerX - 6, centerY - 8, SSD1306_WHITE);
    display.drawLine(centerX - 6, centerY - 8, centerX - 2, centerY - 5, SSD1306_WHITE);
    //right eye
    display.drawLine(centerX + 2, centerY - 5, centerX + 6, centerY - 8, SSD1306_WHITE);
    display.drawLine(centerX + 6, centerY - 8, centerX + 10, centerY - 5, SSD1306_WHITE);
    
    //smile
    display.drawLine(centerX - 8, centerY + 5, centerX - 4, centerY + 8, SSD1306_WHITE);
    display.drawLine(centerX - 4, centerY + 8, centerX + 4, centerY + 8, SSD1306_WHITE);
    display.drawLine(centerX + 4, centerY + 8, centerX + 8, centerY + 5, SSD1306_WHITE);
    
    //text to displaying 
    display.setTextSize(1);
    display.setCursor(30, 58);
    display.print("Waiting...");
    
    display.display(); ///calling diasplay function  tha actually displays on the oled 
}

//actuall abnimation code 
void showWaitingAnimation() {
    static unsigned long lastUpdate = 0;
    static int yOffset = 0;
    static int direction = 1;
    if (millis() - lastUpdate > 100) {  //updates every 100 ms 
        lastUpdate = millis();
        //in y direction because we want it to float up and down 
        yOffset += direction;
        //bounce between -3 and +3
        if (yOffset >= 3 || yOffset <= -3) {
            direction = -direction;
        }
        
        drawCuteFace(yOffset);
    }
}
//this face is shown hen the ai is playing the piano 
void drawRobotFace(int yOffset) {
    display.clearDisplay();
    int centerX = 64;
    int centerY = 32 + yOffset;
    //robotic head-> swquare 
    display.drawRect(centerX - 20, centerY - 20, 40, 40, SSD1306_WHITE);
    display.drawLine(centerX, centerY - 20, centerX, centerY - 25, SSD1306_WHITE);
    display.drawCircle(centerX, centerY - 27, 2, SSD1306_WHITE);
    //eyes 
    display.fillRect(centerX - 12, centerY - 10, 6, 6, SSD1306_WHITE);  // Left eye
    display.fillRect(centerX + 6, centerY - 10, 6, 6, SSD1306_WHITE);   // Right eye
    //mouth 
    display.drawLine(centerX - 10, centerY + 8, centerX - 3, centerY + 8, SSD1306_WHITE);
    display.drawLine(centerX + 3, centerY + 8, centerX + 10, centerY + 8, SSD1306_WHITE);
    //ears 
    display.drawRect(centerX - 24, centerY - 5, 3, 10, SSD1306_WHITE);  // Left
    display.drawRect(centerX + 21, centerY - 5, 3, 10, SSD1306_WHITE);  // Right
    //text 
    display.setTextSize(1);
    display.setCursor(25, 58);
    display.print("AI Playing");
    
    display.display();//again calling display so the text gets displayed on screen
}
void showRobotAnimation() {
    static unsigned long lastUpdate = 0;
    static int yOffset = 0;
    static int direction = 1;
    //same animation code as above 
    if (millis() - lastUpdate > 100) { 
        lastUpdate = millis();
        //bounce up and downin y direction
        yOffset += direction;
        //bounce between -3 and +3
        if (yOffset >= 3 || yOffset <= -3) {
            direction = -direction;
        }
        drawRobotFace(yOffset);
    }
}
void playCasualMode() {
    if (playMode != CASUAL_MODE) return;
    //thiscode basically runs for when 
    int pressedKey = detectPressedKey();
    //
    static int lastPressedKey = -1;
    static bool keyWasPressed = false;
    if (pressedKey != -1) {
        //Key is currently pressed
        if (!keyWasPressed || pressedKey != lastPressedKey) {
            //New key press or different key
            Serial.println("[CASUAL] Key " + String(pressedKey) + " pressed");
            //Turn on LED for visual feedbac
            digitalWrite(ledGuidance[pressedKey], HIGH);
            //Play the note
            playTone(pressedKey);
            lastPressedKey = pressedKey;
            keyWasPressed = true;
        }
    } else {
        //no key pressed turn off LED of last pressed key
        if (keyWasPressed && lastPressedKey != -1) {
            digitalWrite(ledGuidance[lastPressedKey], LOW);
            Serial.println("[CASUAL] Key " + String(lastPressedKey) + " released");
        }
        keyWasPressed = false;
        lastPressedKey = -1;
    }
}
void displayMessage(String line1, String line2 = "") {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println(line1);
    if (line2.length() > 0) {
        display.setCursor(0, 20);
        display.println(line2);
    }
    
    display.display();
}
//not beingused rn will see later 
void handleTouchServo() {
}

// ── Camera initialisation (AI-Thinker pin map) ───────────────────────────────

float detectFrequency() {
    int minVal = 4095;
    int maxVal = 0;

    for (int i = 0; i < 500; i++) {
        int val = analogRead(MIC_PIN);

        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;

        delayMicroseconds(125);
    }

    int amplitude = maxVal - minVal;

    Serial.print("[MIC AMP] ");
    Serial.println(amplitude);

    if (amplitude < 20) {
        return 0;
    }

    return amplitude * 10;
}

void handleFreqRecording() {
    if (millis() - lastSampleTime >= 100) {
        lastSampleTime = millis();

        float freq = detectFrequency();

        Serial.print("[FREQ] Sample ");
        Serial.print(freqSamples);
        Serial.print(": ");
        Serial.println(freq);

        if (freqBuffer.length() > 0) freqBuffer += ",";
        freqBuffer += String(freq, 1);
        freqSamples++;

        if (freqSamples >= TOTAL_SAMPLES) {
            freqRecording = false;
            Serial.println("[FREQ] Sending FREQDATA");
            sendResponse("FREQDATA", freqBuffer);
            freqBuffer = "";
            freqSamples = 0;
            displayMessage("Done!", "Sent to Python");
        }
    }
}



void setup() {
    Serial.begin(115200);
    delay(1000);
    //oled initialization 
    Wire.begin(SDA_PIN, SCL_PIN);
    delay(100);
    Serial.println("Initializing OLED...");
    if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        Serial.println("SSD1306 allocation failed");
        for(;;);
    }
    display.clearDisplay();
    display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(10, 10);
    display.println("Piano");
    display.setCursor(10, 30);
    display.println("Guru!");
    display.display();
    delay(2000);
    Serial.println("OLED initialized!");
    delay(2000);
    Serial.println("\n╔════════════════════════════════════╗");
    Serial.println("║   ESP32 PIANO CONTROLLER v7.0     ║");
    Serial.println("║   AMPLIFIER + MIC + CAMERA ADDED  ║");
    Serial.println("╚════════════════════════════════════╝\n");

    // Camera init — non-fatal if module not present
    //cameraReady = initCamera();
    //led initialization 
    Serial.println("Setting up LEDs...");
    for (int i = 0; i < NUM_KEYS; i++) {
        pinMode(ledGuidance[i], OUTPUT);
        digitalWrite(ledGuidance[i], LOW);
    }
    pinMode(ledMis, OUTPUT);
    digitalWrite(ledMis, LOW);
    //fsr set ups 
    Serial.println("Setting up FSRs...");
    for (int i = 0; i < NUM_KEYS; i++) {
        pinMode(fsrPins[i], INPUT);
    }
    //amplifier output pin (AMP_PIN = buzzer alias)
    pinMode(buzzer, OUTPUT);
    Serial.println("Hardware ready!");
    sendResponse("STATUS", "System Ready"); //sends to pyython
    Serial.println("Waiting for commands...\n");
    initLevelServo();
    moveLevelServo(0);
    playStartupSound();
    // Re-initialize after startup tones, because tone()/LEDC can disturb servo PWM
    initLevelServo();
    moveLevelServo(0);
}
//connections with python
//serial communication 
void sendResponse(String type, String message) {
    Serial.println(type + ":" + message);
    Serial.flush();
}
void handleCommand(String cmd) {
    cmd.trim();
    Serial.println("[CMD] " + cmd);
    if (cmd == "INIT") {
       //in start we display piano guruonthe screen 
        display.clearDisplay();
        display.setTextSize(2);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(10, 10);
        display.println("Piano");
        display.setCursor(10, 30);
        display.println("Guru!");
        display.display();
        Serial.println("[INIT] OLED reinitialized");
        sendResponse("STATUS", "System Ready");
        return;
    }
    if (cmd.startsWith("LCD:")) {
        String message = cmd.substring(4);
        int separator = message.indexOf('|');
        if (separator > 0) {
            String line1 = message.substring(0, separator);
            String line2 = message.substring(separator + 1);
            displayMessage(line1, line2);
        } else {
            displayMessage(message);
        }
        Serial.println("[LCD] Displayed: " + message);
        return;
    }
    else if (cmd.startsWith("NOTES:")) {
        melody = cmd.substring(6);
        melody.trim();
        melodyLength = melody.length();
        Serial.println("[INFO] Melody: " + melody);
        Serial.println("[INFO] Length: " + String(melodyLength));
        sendResponse("STATUS", "Melody loaded: " + String(melodyLength) + " notes");
        return;
    }
    if (cmd == "CASUALMODE") {
        playMode = CASUAL_MODE;
        resetLights();
        resetMotors();
        Serial.println("[START] Casual mode activated - Free play!");
        sendResponse("STATUS", "Casual mode activated");
        return;
    }

    if (cmd == "USERPLAYS") {
        playMode = USER_PLAYING;
        currentNoteIndex = 0;
        errorCount = 0;
        noteState = SHOW_LED;
        errorAlreadyCounted = false;
        resetLights();
        resetMotors();
        Serial.println("[START] User mode activated");
        sendResponse("STATUS", "User playing mode activated");
        return;
    }

    if (cmd == "AIPLAYS") {
        playMode = AI_PLAYING;
        currentNoteIndex = 0;
        lastNoteTime = millis();
        resetLights();
        resetMotors();
        Serial.println("[START] AI mode activated");
        sendResponse("STATUS", "AI playing mode activated");
        return;
    }

    if (cmd == "STOP") {
        playMode = IDLE;
        currentNoteIndex = 0;
        errorCount = 0;
        noteState = SHOW_LED;
        resetLights();
        resetMotors();
        // also cancel any active mic / cam session
        freqRecording = false;
        freqBuffer = "";
        Serial.println("[STOP] Stopped");
        sendResponse("STATUS", "Stopped");
        return;
    }

    // ── Microphone commands ───────────────────────────────────────────────────
    if (cmd == "STARTFREQ") {
        freqRecording = true;
        freqSamples = 0;
        freqBuffer = "";
        lastSampleTime = millis();
        displayMessage("Listening...", "Hum now!");
        Serial.println("[FREQ] Recording started");
        sendResponse("STATUS", "FREQ_RECORDING");
        return;
    }

    // ── Play melody received from Python (voice-to-melody result) ────────────
    // Python processes MICDATA, runs hummingToMelody, then sends back
    // PLAYHUM:<notes> so the ESP32 plays the generated melody through the amp
    if (cmd.startsWith("PLAYHUM:")) {
        String humNotes = cmd.substring(8);
        humNotes.trim();

        Serial.println("[HUM] Playing back: " + humNotes);
        displayMessage("Playing", "your hum!");

        for (int i = 0; i < humNotes.length(); i++) {
            int key = humNotes[i] - '0' - 1;

            if (key >= 0 && key < NUM_KEYS) {
                digitalWrite(ledGuidance[key], HIGH);
                tone(buzzer, musicalNotes[key], 300);
                delay(350);
                digitalWrite(ledGuidance[key], LOW);
                delay(100);
            }
        }

        noTone(buzzer);
        sendResponse("STATUS", "HUM_PLAYBACK_DONE");
        return;
    }
    if (cmd.startsWith("LEVELMOVE:")) {
        int angle = cmd.substring(10).toInt();
        Serial.println("[ESP32 LEVEL] Got command: " + cmd);
        moveLevelServo(angle);
        return;
    }
    
}

void readSerial() {
    while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (serialBuffer.length() > 0) {
                handleCommand(serialBuffer);
                serialBuffer = "";
            }
        } else {
            serialBuffer += c;
        }
    }
}

//helper functions 
//helps resetting things 
void resetLights() {
    for (int i = 0; i < NUM_KEYS; i++) {
        digitalWrite(ledGuidance[i], LOW);
    }
    digitalWrite(ledMis, LOW);
}
void resetMotors() {
}
int detectPressedKey() {
    //this function detects ehich fsr is prssed and it returnrns the index which then plays the respectiv esound s
    int strongestIndex = -1;
    int strongestValue = 0;
    for (int i = 0; i < NUM_KEYS; i++) {
        int val = analogRead(fsrPins[i]);
        if (val > strongestValue) {
            strongestValue = val;
            strongestIndex = i;
        }
    }
    if (strongestValue > fsrThresh) {
        return strongestIndex;
    }
    return -1;
}

void playTone(int keyNumber) {
    //take the respective index and play the note on thepassiv ebuzzer 
    if (keyNumber >= 0 && keyNumber < NUM_KEYS) {
        tone(buzzer, musicalNotes[keyNumber], 150);
    }
}
//user playing mode 
void playUserMode() {
    if (playMode != USER_PLAYING) return;
    if (currentNoteIndex >= melodyLength) {
        playMode = IDLE;
        resetLights();
        Serial.println("[COMPLETE] Song finished!");
        Serial.println("[RESULT] Total notes: " + String(melodyLength));
        Serial.println("[RESULT] Errors: " + String(errorCount));
        Serial.println("[RESULT] Correct: " + String(melodyLength - errorCount));
        float percentage = ((float)(melodyLength - errorCount) / melodyLength) * 100.0;
        Serial.println("[RESULT] Accuracy: " + String(percentage, 2) + "%");
        sendResponse("COMPLETION", "User finished. Errors: " + String(errorCount));
        return;
    }
    int expectedKey = melody[currentNoteIndex] - '0' -1 ; //fixing the indexing 
    if (expectedKey < 0 || expectedKey >= NUM_KEYS) { //8
        Serial.println("[ERROR] Invalid key at index " + String(currentNoteIndex));
        currentNoteIndex++;
        noteState = SHOW_LED;
        return;
    }
    if (noteState == SHOW_LED) {
        digitalWrite(ledGuidance[expectedKey], HIGH);
        Serial.println("Note " + String(currentNoteIndex + 1) + "/" + String(melodyLength) + ": Expecting key " + String(expectedKey));
        
        noteState = WAIT_FOR_PRESS;
        errorAlreadyCounted = false;
        return;
    }
    int pressedKey = detectPressedKey();

    //wait for press key 
    if (noteState == WAIT_FOR_PRESS) {
        if (pressedKey != -1) {
            Serial.print("  Pressed: " + String(pressedKey) + " -> ");
            playTone(pressedKey);
            if(pressedKey!=expectedKey){
                Serial.println("WRONG ✗ (expected " + String(expectedKey) + ")");
                errorCount++;
                digitalWrite(ledMis, HIGH);
                delay(150);
                digitalWrite(ledMis, LOW);
            }
            
            noteState = WAIT_FOR_RELEASE;
        }
        return;
    }
    if (noteState == WAIT_FOR_RELEASE) {
        if (pressedKey == -1) {
            Serial.println("  Released. Moving to next note.\n");
            digitalWrite(ledGuidance[expectedKey], LOW);
            currentNoteIndex++;
            noteState = SHOW_LED;
            delay(50);
        }
        return;
    }
}
void playAIMode() {
    if (playMode != AI_PLAYING) return;
    if (currentNoteIndex >= melodyLength) {
        playMode = IDLE;
        resetLights();
        resetMotors();
        
        Serial.println("[COMPLETE] AI finished playing");
        sendResponse("COMPLETION", "AI finished playing");
        return;
    }
    if (millis() - lastNoteTime < noteInterval) {
        return;
    }
    lastNoteTime = millis();
    int key = melody[currentNoteIndex] - '0'-1;
    if (key < 0 || key >= NUM_KEYS) {
        currentNoteIndex++;
        return;
    }
    Serial.println("[AI] Playing note " + String(currentNoteIndex + 1) + "/" + String(melodyLength) + ": Key " + String(key));
    digitalWrite(ledGuidance[key], HIGH);
    //play tone 
    playTone(key);
    delay(200);
    digitalWrite(ledGuidance[key], LOW);
    currentNoteIndex++;
}
//main loop
void loop() {
    readSerial();

    // ── New hardware polling ──────────────────────────────────────────────────
    if (freqRecording) handleFreqRecording();

    if (playMode == USER_PLAYING) {
        playUserMode();
        showWaitingAnimation();
    } else if (playMode == AI_PLAYING) {
        playAIMode();
        showRobotAnimation();
    }
    else if (playMode == CASUAL_MODE) {
        playCasualMode();
        showWaitingAnimation();
    }
    
    delay(10);
}