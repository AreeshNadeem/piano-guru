/*
  robot_esp32.ino  —  Servo controller for the AI Robot Companion
*/


#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64

// ═════════════════════════════════════════════════════════════════════════════
//  CONFIG
// ═════════════════════════════════════════════════════════════════════════════
const char* WIFI_SSID      = "Hell nawh";
const char* WIFI_PASSWORD  = "12345678";
const char* MQTT_BROKER    = "10.146.247.242";
const int   MQTT_PORT      = 1883;
const char* MQTT_CLIENT    = "robot_esp32";
const char* MQTT_TOPIC_SUB = "robot/servo/command";
const char* MQTT_TOPIC_PUB = "robot/servo/status";

const int PIN_LEFT_ARM  = 13;
const int PIN_RIGHT_ARM = 12;
const int PIN_HEAD      = 14;
const int PIN_WAIST_A   = 27;
const int PIN_WAIST_B   = 26;
const int LED1_PIN      = 2;
const int LED2_PIN      = 4;
const bool WAIST_B_MIRROR = true;

const int L_DOWN   = 90;
const int R_DOWN   = 90;
const int H_CENTRE = 90;
const int W_CENTRE = 90;

const int PIN_LEVEL = 25;
Servo levelServo;

// ─── Speed constants (ms per step — lower = faster) ──────────────────────────
#define SPD_FAST    5
#define SPD_NORMAL  8
#define SPD_GENTLE  10
#define SPD_SLOW    14

// ═════════════════════════════════════════════════════════════════════════════
//  GLOBALS
// ═════════════════════════════════════════════════════════════════════════════
Servo leftArm, rightArm, head, waistA, waistB;
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
WiFiClient   wifiClient;
PubSubClient mqtt(wifiClient);
String       command = "";
bool         idling  = true;

// ═════════════════════════════════════════════════════════════════════════════
//  LOGGING
// ═════════════════════════════════════════════════════════════════════════════
void mqttLog(const char* msg) {
    Serial.println(msg);
    if (mqtt.connected()) mqtt.publish(MQTT_TOPIC_PUB, msg);
}
void mqttLog(String msg) { mqttLog(msg.c_str()); }
void networkTick() {
    static bool          wifiStarted   = false;
    static unsigned long lastMqttCheck = 0;
    static unsigned long lastStatusLog = 0;

    if (!wifiStarted) {
        WiFi.mode(WIFI_STA);
        WiFi.disconnect(true);
        delay(100);
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
        wifiStarted = true;
        Serial.println(F("[WIFI] Starting connection..."));
    }

    unsigned long now = millis();

    // Print status every 3 seconds
    if (now - lastStatusLog > 3000) {
        lastStatusLog = now;
        Serial.print(F("[WIFI] Status: "));
        Serial.print(WiFi.status());
        Serial.print(F("  MQTT: "));
        Serial.println(mqtt.connected() ? "connected" : "not connected");
    }

    if (WiFi.status() != WL_CONNECTED) return;  // not connected, nothing else to do

    // WiFi is up — attempt MQTT if needed
    if (!mqtt.connected() && now - lastMqttCheck > 5000) {
        lastMqttCheck = now;
        Serial.print(F("[MQTT] Connecting to "));
        Serial.print(MQTT_BROKER);
        Serial.print(F(":"));
        Serial.println(MQTT_PORT);
        if (mqtt.connect(MQTT_CLIENT)) {
            Serial.println(F("[MQTT] Connected!"));
            mqtt.subscribe(MQTT_TOPIC_SUB);
        } else {
            Serial.print(F("[MQTT] Failed, rc="));
            Serial.println(mqtt.state());
        }
    }

    if (mqtt.connected()) mqtt.loop();
}
// ═════════════════════════════════════════════════════════════════════════════
//  MOVEMENT PRIMITIVES
// ═════════════════════════════════════════════════════════════════════════════
void moveSingleServo(Servo& servo, int target, int stepMs = SPD_NORMAL) {
    int cur = servo.read(), steps = abs(target - cur);
    if (!steps) return;
    for (int i = 1; i <= steps; i++) {
        float t = (float)i / steps;
        t = t * t * (3.0f - 2.0f * t);
        servo.write(cur + (int)((target - cur) * t));
        delay(stepMs);
    }
    servo.write(target);
}

void moveAllTo(int lArm, int rArm, int hd, int wst, int stepDelay = SPD_NORMAL) {
    int curL = leftArm.read(), curR = rightArm.read(),
        curH = head.read(),    curW = waistA.read();
    int steps = max({ abs(lArm-curL), abs(rArm-curR), abs(hd-curH), abs(wst-curW) });
    if (!steps) return;
    for (int i = 1; i <= steps; i++) {
        float t = (float)i / steps;
        t = t * t * (3.0f - 2.0f * t);
        leftArm.write(curL  + (int)((lArm - curL) * t));
        rightArm.write(curR + (int)((rArm - curR) * t));
        head.write(curH     + (int)((hd   - curH) * t));
        int wA = curW + (int)((wst - curW) * t);
        waistA.write(wA);
        waistB.write(WAIST_B_MIRROR ? (180 - wA) : wA);
        delay(stepDelay);
    }
    waistA.write(wst);
    waistB.write(WAIST_B_MIRROR ? (180 - wst) : wst);
}

void moveWaistTo(int target, int stepMs = SPD_NORMAL) {
    int cur = waistA.read(), steps = abs(target - cur);
    if (!steps) return;
    for (int i = 1; i <= steps; i++) {
        float t = (float)i / steps;
        t = t * t * (3.0f - 2.0f * t);
        int wA = cur + (int)((target - cur) * t);
        waistA.write(wA);
        waistB.write(WAIST_B_MIRROR ? (180 - wA) : wA);
        delay(stepMs);
    }
    waistA.write(target);
    waistB.write(WAIST_B_MIRROR ? (180 - target) : target);
}

void resetAll(int pauseMs = 300) {
    moveAllTo(L_DOWN, R_DOWN, H_CENTRE, W_CENTRE, SPD_FAST);
    delay(pauseMs);
}

// ═════════════════════════════════════════════════════════════════════════════
//  INDIVIDUAL JOINTS
// ═════════════════════════════════════════════════════════════════════════════
void raiseLeftArm()    { moveSingleServo(leftArm,  160, SPD_NORMAL); }
void lowerLeftArm()    { moveSingleServo(leftArm,  L_DOWN, SPD_NORMAL); }
void raiseRightArm()   { moveSingleServo(rightArm, 20,  SPD_NORMAL); }
void lowerRightArm()   { moveSingleServo(rightArm, R_DOWN, SPD_NORMAL); }
void raiseBothArms()   { moveAllTo(160, 20,    head.read(), waistA.read(), SPD_NORMAL); }
void lowerBothArms()   { moveAllTo(L_DOWN, R_DOWN, head.read(), waistA.read(), SPD_NORMAL); }
void lookLeft()        { moveSingleServo(head, 108, SPD_NORMAL); }
void lookRight()       { moveSingleServo(head,  72, SPD_NORMAL); }
void lookCentre()      { moveSingleServo(head, H_CENTRE, SPD_NORMAL); }
void moveWaistLeft()   { moveWaistTo(70,  SPD_GENTLE); }
void moveWaistRight()  { moveWaistTo(110, SPD_GENTLE); }
void moveWaistCentre() { moveWaistTo(W_CENTRE, SPD_NORMAL); }

// ═════════════════════════════════════════════════════════════════════════════
//  MOVEMENT ROUTINES
// ═════════════════════════════════════════════════════════════════════════════
void headNod() {
    for (int i = 0; i < 3; i++) {
        moveAllTo(L_DOWN, R_DOWN, 105, W_CENTRE, SPD_FAST);
        delay(50);
        moveAllTo(L_DOWN, R_DOWN, 75, W_CENTRE, SPD_FAST);
        delay(50);
    }
    moveAllTo(L_DOWN, R_DOWN, H_CENTRE, W_CENTRE, SPD_FAST);
}

void raiseHand() {
    moveAllTo(L_DOWN, 160, 80, W_CENTRE, SPD_NORMAL);
    delay(600);
    resetAll();
}

void waveForward() {
    for (int i = 0; i < 4; i++) {
        moveAllTo(L_DOWN, 130, H_CENTRE, W_CENTRE, SPD_FAST);
        delay(40);
        moveAllTo(L_DOWN, 160, H_CENTRE, W_CENTRE, SPD_FAST);
        delay(40);
    }
    resetAll();
}

void confused_move() {
    moveAllTo(120, 120, 110, W_CENTRE, SPD_NORMAL);
    delay(700);
    resetAll();
}

void celebrate() {
    moveAllTo(160, 160, H_CENTRE, W_CENTRE, SPD_FAST);
    for (int i = 0; i < 3; i++) {
        moveAllTo(160, 160, H_CENTRE,  70, SPD_FAST);
        moveAllTo(160, 160, H_CENTRE, 110, SPD_FAST);
    }
    moveAllTo(160, 160, H_CENTRE, W_CENTRE, SPD_FAST);
    for (int i = 0; i < 4; i++) {
        moveAllTo(140, 180, H_CENTRE, W_CENTRE, SPD_FAST);
        moveAllTo(180, 140, H_CENTRE, W_CENTRE, SPD_FAST);
    }
    resetAll(500);
}

void sleepy() {
    moveAllTo(60, 60, 105, W_CENTRE, SPD_SLOW);
    delay(800);
    resetAll(400);
}

void danceHappy() {
    for (int i = 0; i < 4; i++) {
        moveAllTo(150,  60, H_CENTRE,  70, SPD_FAST);
        delay(80);
        moveAllTo( 60, 150, H_CENTRE, 110, SPD_FAST);
        delay(80);
    }
    resetAll();
}

void slowDown() {
    moveAllTo(150, 150, H_CENTRE, W_CENTRE, SPD_SLOW);
    delay(500);
    moveAllTo(90,   90, H_CENTRE, W_CENTRE, SPD_SLOW);
    delay(300);
    moveAllTo(150, 150, H_CENTRE, W_CENTRE, SPD_SLOW);
    delay(500);
    resetAll();
}

void pumpUp() {
    for (int i = 0; i < 6; i++) {
        moveAllTo(160,  40, H_CENTRE, W_CENTRE, SPD_FAST - 1);
        delay(40);
        moveAllTo( 40, 160, H_CENTRE, W_CENTRE, SPD_FAST - 1);
        delay(40);
    }
    resetAll();
}

void gentleNod() {
    for (int i = 0; i < 2; i++) {
        moveAllTo(L_DOWN, R_DOWN, 100, W_CENTRE, SPD_GENTLE);
        delay(180);
        moveAllTo(L_DOWN, R_DOWN,  80, W_CENTRE, SPD_GENTLE);
        delay(180);
    }
    moveAllTo(L_DOWN, R_DOWN, H_CENTRE, W_CENTRE, SPD_GENTLE);
}

void headShake() {
    for (int i = 0; i < 3; i++) {
        moveAllTo(L_DOWN, R_DOWN, H_CENTRE,  75, SPD_FAST);
        delay(60);
        moveAllTo(L_DOWN, R_DOWN, H_CENTRE, 105, SPD_FAST);
        delay(60);
    }
    moveAllTo(L_DOWN, R_DOWN, H_CENTRE, W_CENTRE, SPD_FAST);
}

void bothArmsUp() {
    moveAllTo(170, 170, 75, W_CENTRE, SPD_NORMAL);
    delay(1000);
    resetAll();
}

void comfort() {
    for (int i = 0; i < 3; i++) {
        moveAllTo(L_DOWN, R_DOWN, H_CENTRE,  75, SPD_GENTLE);
        delay(250);
        moveAllTo(L_DOWN, R_DOWN, H_CENTRE, 105, SPD_GENTLE);
        delay(250);
    }
    resetAll();
}

void attentionBlink() {
    for (int i = 0; i < 6; i++) {
        digitalWrite(LED1_PIN, HIGH); digitalWrite(LED2_PIN, LOW);  delay(120);
        digitalWrite(LED1_PIN, LOW);  digitalWrite(LED2_PIN, HIGH); delay(120);
    }
    digitalWrite(LED1_PIN, LOW);
    digitalWrite(LED2_PIN, LOW);
}

// ═════════════════════════════════════════════════════════════════════════════
//  FACE DRAWING
// ═════════════════════════════════════════════════════════════════════════════
void drawFace(float eyeOpen, int pupilX, int browY, int browAngle, int mouthType, bool blushOn) {
    display.clearDisplay();

    const int ELX = 36, ERX = 92, ECY = 28, ERX_R = 14, ERY_F = 11;
    int eyeH = max(1, (int)(ERY_F * eyeOpen));

    // Eyes
    for (int dy = -eyeH; dy <= eyeH; dy++) {
        float r = (float)dy / eyeH;
        int hw = (int)(ERX_R * sqrt(1.0f - r * r));
        display.drawFastHLine(ELX - hw, ECY + dy, hw * 2 + 1, WHITE);
        display.drawFastHLine(ERX - hw, ECY + dy, hw * 2 + 1, WHITE);
    }

    // Pupils
    if (eyeOpen > 0.25f) {
        int px = constrain(pupilX, -5, 5);
        display.fillCircle(ELX + px, ECY, 4, BLACK);
        display.fillCircle(ERX + px, ECY, 4, BLACK);
        display.drawPixel(ELX + px + 2, ECY - 2, WHITE);
        display.drawPixel(ERX + px + 2, ECY - 2, WHITE);
    }

    // Eyelid bar
    if (eyeOpen < 0.99f) {
        int lb = ECY - eyeH, lt = ECY - ERY_F - 2, lh = lb - lt;
        if (lh > 0) {
            display.fillRect(ELX - ERX_R - 2, lt, ERX_R * 2 + 4, lh, BLACK);
            display.fillRect(ERX - ERX_R - 2, lt, ERX_R * 2 + 4, lh, BLACK);
        }
    }

    // Eyebrows
    int baseY = ECY - ERY_F - 4 + browY;
    int lBrowInnerY = baseY + browAngle, lBrowOuterY = baseY - browAngle;
    int rBrowInnerY = baseY + browAngle, rBrowOuterY = baseY - browAngle;
    display.drawLine(ELX - 8, lBrowOuterY,     ELX + 8, lBrowInnerY,     WHITE);
    display.drawLine(ELX - 8, lBrowOuterY + 1, ELX + 8, lBrowInnerY + 1, WHITE);
    display.drawLine(ERX + 8, rBrowOuterY,     ERX - 8, rBrowInnerY,     WHITE);
    display.drawLine(ERX + 8, rBrowOuterY + 1, ERX - 8, rBrowInnerY + 1, WHITE);

    // Blush
    if (blushOn) {
        for (int bx = -4; bx <= 4; bx += 2) {
            display.drawPixel(ELX - 2 + bx, ECY + ERY_F + 3, WHITE);
            display.drawPixel(ERX - 2 + bx, ECY + ERY_F + 3, WHITE);
        }
        for (int bx = -3; bx <= 3; bx += 2) {
            display.drawPixel(ELX - 2 + bx, ECY + ERY_F + 5, WHITE);
            display.drawPixel(ERX - 2 + bx, ECY + ERY_F + 5, WHITE);
        }
    }

    // Mouth
    int mX = 64, mY = 50;
    switch (mouthType) {
        case 0:
            for (int sx=-18; sx<=18; sx++) { float r=(float)sx/18.0f; int sy=7-(int)(7.0f*r*r); display.drawPixel(mX+sx,mY+sy,WHITE); display.drawPixel(mX+sx,mY+sy+1,WHITE); }
            break;
        case 1:
            for (int sx=-18; sx<=18; sx++) { float r=(float)sx/18.0f; int sy=(int)(7.0f*r*r); display.drawPixel(mX+sx,mY+sy,WHITE); display.drawPixel(mX+sx,mY+sy+1,WHITE); }
            break;
        case 2:
            display.drawFastHLine(mX-14, mY+4, 28, WHITE);
            display.drawFastHLine(mX-14, mY+5, 28, WHITE);
            break;
        case 3:
            display.drawCircle(mX, mY+2, 6, WHITE);
            break;
        case 4:
            for (int sx=-20; sx<=20; sx++) { float r=(float)sx/20.0f; int sy=9-(int)(9.0f*r*r); display.drawPixel(mX+sx,mY+sy,WHITE); display.drawPixel(mX+sx,mY+sy+1,WHITE); display.drawPixel(mX+sx,mY+sy+2,WHITE); }
            break;
        case 5:
            for (int sx=-14; sx<=14; sx++) { float r=(float)sx/14.0f; int sy=(int)(4.0f*r*r)+(int)(2.0f*sin(r*3.14f*2)); display.drawPixel(mX+sx,mY+sy,WHITE); display.drawPixel(mX+sx,mY+sy+1,WHITE); }
            break;
    }
    display.display();
}

void drawIdleFace(float eyeOpen, int pupilX) { drawFace(eyeOpen, pupilX, -2, 0, 0, false); }

// ═════════════════════════════════════════════════════════════════════════════
//  EXPRESSION PRESETS
// ═════════════════════════════════════════════════════════════════════════════
void faceHappy()     { drawFace(1.0f,  0, -4,  0, 0, false); }
void faceSad()       { drawFace(0.75f, 0, -1, -3, 5, false); }
void faceAngry()     { drawFace(0.6f,  0,  3,  4, 1, false); }
void faceEnergetic() { drawFace(1.0f,  3, -4,  0, 4, false); }
void faceTired()     { drawFace(0.3f,  0,  2, -2, 2, false); }
void faceCalm()      { drawFace(0.75f, 0, -1,  0, 0, false); }
void faceMiserable() { drawFace(0.4f,  0,  1, -4, 1, false); }
void faceBlushing()  { drawFace(0.9f,  3, -2,  0, 0, true);  }

void faceConfused() {
    display.clearDisplay();
    const int ELX=36, ERX=92, ECY=28, ERX_R=14, ERY_F=11;
    for (int dy=-ERY_F; dy<=ERY_F; dy++) {
        float r=(float)dy/ERY_F; int hw=(int)(ERX_R*sqrt(1.0f-r*r));
        display.drawFastHLine(ELX-hw,ECY+dy,hw*2+1,WHITE);
        display.drawFastHLine(ERX-hw,ECY+dy,hw*2+1,WHITE);
    }
    display.fillCircle(ELX+2,ECY,4,BLACK); display.fillCircle(ERX-2,ECY,4,BLACK);
    display.drawPixel(ELX+4,ECY-2,WHITE);  display.drawPixel(ERX,ECY-2,WHITE);
    display.drawLine(ELX-8,ECY-16,ELX+8,ECY-19,WHITE);
    display.drawLine(ELX-8,ECY-15,ELX+8,ECY-18,WHITE);
    display.drawLine(ERX-8,ECY-14,ERX+8,ECY-12,WHITE);
    display.drawLine(ERX-8,ECY-13,ERX+8,ECY-11,WHITE);
    for (int sx=-14; sx<=14; sx++) { int sy=(int)(3.0f*sin((float)sx*0.4f)); display.drawPixel(64+sx,51+sy,WHITE); display.drawPixel(64+sx,52+sy,WHITE); }
    display.drawPixel(110,10,WHITE); display.drawPixel(111,11,WHITE);
    display.drawPixel(110,12,WHITE); display.drawPixel(110,13,WHITE);
    display.display();
}

void faceExcited() {
    display.clearDisplay();
    const int ELX=36, ERX=92, ECY=28, ERX_R=15, ERY_F=13;
    for (int dy=-ERY_F; dy<=ERY_F; dy++) {
        float r=(float)dy/ERY_F; int hw=(int)(ERX_R*sqrt(1.0f-r*r));
        display.drawFastHLine(ELX-hw,ECY+dy,hw*2+1,WHITE);
        display.drawFastHLine(ERX-hw,ECY+dy,hw*2+1,WHITE);
    }
    display.fillCircle(ELX,ECY,5,BLACK); display.fillCircle(ERX,ECY,5,BLACK);
    display.drawPixel(ELX+3,ECY-3,WHITE); display.drawPixel(ERX+3,ECY-3,WHITE);
    int baseY = ECY-ERY_F-7;
    display.drawLine(ELX-8,baseY,  ELX+8,baseY-1,WHITE); display.drawLine(ELX-8,baseY+1,ELX+8,baseY,  WHITE);
    display.drawLine(ERX+8,baseY,  ERX-8,baseY-1,WHITE); display.drawLine(ERX+8,baseY+1,ERX-8,baseY,  WHITE);
    for (int sx=-22; sx<=22; sx++) { float r=(float)sx/22.0f; int sy=10-(int)(10.0f*r*r); display.drawPixel(64+sx,48+sy,WHITE); display.drawPixel(64+sx,49+sy,WHITE); display.drawPixel(64+sx,50+sy,WHITE); }
    display.drawPixel(10,5,WHITE);  display.drawPixel(12,3,WHITE);  display.drawPixel(14,6,WHITE);
    display.drawPixel(118,5,WHITE); display.drawPixel(116,3,WHITE); display.drawPixel(114,6,WHITE);
    display.display();
}

void drawNote(int note) {
    display.clearDisplay(); display.setTextSize(3); display.setTextColor(WHITE); display.setCursor(40,20);
    switch(note){case 1:display.print("C");break;case 2:display.print("D");break;case 3:display.print("E");break;case 4:display.print("F");break;case 5:display.print("G");break;case 6:display.print("A");break;default:display.print("?");}
    display.display();
}

// ═════════════════════════════════════════════════════════════════════════════
//  IDLE LOOP
// ═════════════════════════════════════════════════════════════════════════════
void idleLoop() {
    static float         eyeOpen     = 1.0f;
    static int           blinkPhase  = 0;
    static unsigned long blinkNextMs = 0, blinkStepMs = 0;
    static int           pupilX = 0, pupilDir = 1;
    static unsigned long pupilStepMs = 0;
    static unsigned long lastDraw    = 0;
    static unsigned long neckNextMs  = 0;
    static bool          headLeft    = true;
    static bool          inited      = false;

    unsigned long now = millis();
    if (!inited) { blinkNextMs=now+2000; pupilStepMs=now; neckNextMs=now+5000; inited=true; }

    if (blinkPhase==0 && now>=blinkNextMs)        { blinkPhase=1; blinkStepMs=now; }
    if (blinkPhase==1 && now-blinkStepMs>=40)     { blinkStepMs=now; eyeOpen-=0.25f; if(eyeOpen<=0){eyeOpen=0;blinkPhase=2;blinkStepMs=now;} }
    if (blinkPhase==2 && now-blinkStepMs>=80)     { blinkPhase=3; blinkStepMs=now; }
    if (blinkPhase==3 && now-blinkStepMs>=40)     { blinkStepMs=now; eyeOpen+=0.25f; if(eyeOpen>=1){eyeOpen=1;blinkPhase=0;blinkNextMs=now+2000+random(500,2500);} }

    if (now-pupilStepMs>=150) { pupilStepMs=now; pupilX+=pupilDir; if(pupilX>=4)pupilDir=-1; if(pupilX<=-4)pupilDir=1; }

    if (now>=neckNextMs) {
        int target = H_CENTRE + (headLeft ? 18 : -35);
        moveSingleServo(head, target, SPD_SLOW);
        delay(1000);
        moveSingleServo(head, H_CENTRE, SPD_SLOW);
        headLeft   = !headLeft;
        neckNextMs = millis() + 5000 + random(0, 3000);
    }

    if (now-lastDraw>=50) { lastDraw=now; drawIdleFace(eyeOpen, pupilX); }
}

void moveLevelServo(int angle) {
    angle = constrain(angle, 0, 180);
    levelServo.write(angle);
    mqttLog("[ROBOT LEVEL] moved to " + String(angle));
}

// ═════════════════════════════════════════════════════════════════════════════
//  OLED HANDLER
// ═════════════════════════════════════════════════════════════════════════════
void handleOLED(String cmd) {
    if      (cmd=="OLED:IDLE")                idling=true;
    else if (cmd.startsWith("OLED:AI_NOTE:")) drawNote(cmd.substring(13).toInt());
    else if (cmd=="OLED:EMOTION:HAPPY")       faceHappy();
    else if (cmd=="OLED:EMOTION:SAD")         faceSad();
    else if (cmd=="OLED:EMOTION:ANGRY")       faceAngry();
    else if (cmd=="OLED:EMOTION:ENERGETIC")   faceEnergetic();
    else if (cmd=="OLED:EMOTION:TIRED")       faceTired();
    else if (cmd=="OLED:EMOTION:CALM")        faceCalm();
    else if (cmd=="OLED:EMOTION:MISERABLE")   faceMiserable();
    else if (cmd=="OLED:EMOTION:CONFUSED")    faceConfused();
    else if (cmd=="OLED:EMOTION:BLUSHING")    faceBlushing();
    else if (cmd=="OLED:EMOTION:EXCITED")     faceExcited();
}

// ═════════════════════════════════════════════════════════════════════════════
//  COMMAND DISPATCHER
// ═════════════════════════════════════════════════════════════════════════════
void handleCommand(String cmd) {
    if (cmd.startsWith("OLED:")) { handleOLED(cmd); return; }
    mqttLog("[ROBOT CMD] " + cmd);
    idling = false;

    if (cmd.startsWith("LEVELMOVE:")) {
        int angle = cmd.substring(10).toInt();
        moveLevelServo(angle);
        return;
    }   

    if      (cmd=="HEAD_NOD")        headNod();
    else if (cmd=="RAISE_HAND")      raiseHand();
    else if (cmd=="WAVE_FORWARD")    waveForward();
    else if (cmd=="CONFUSED")        confused_move();
    else if (cmd=="CELEBRATE")       celebrate();
    else if (cmd=="SLEEPY")          sleepy();
    else if (cmd=="DANCE_HAPPY")     danceHappy();
    else if (cmd=="SLOW_DOWN")       slowDown();
    else if (cmd=="PUMP_UP")         pumpUp();
    else if (cmd=="GENTLE_NOD")      gentleNod();
    else if (cmd=="HEAD_SHAKE")      headShake();
    else if (cmd=="BOTH_ARMS_UP")    bothArmsUp();
    else if (cmd=="COMFORT")         comfort();
    else if (cmd=="RAISE_LEFT_ARM")  raiseLeftArm();
    else if (cmd=="LOWER_LEFT_ARM")  lowerLeftArm();
    else if (cmd=="RAISE_RIGHT_ARM") raiseRightArm();
    else if (cmd=="LOWER_RIGHT_ARM") lowerRightArm();
    else if (cmd=="RAISE_BOTH_ARMS") raiseBothArms();
    else if (cmd=="LOWER_BOTH_ARMS") lowerBothArms();
    else if (cmd=="LOOK_LEFT")       lookLeft();
    else if (cmd=="LOOK_RIGHT")      lookRight();
    else if (cmd=="LOOK_CENTRE")     lookCentre();
    else if (cmd=="WAIST_LEFT")      moveWaistLeft();
    else if (cmd=="WAIST_RIGHT")     moveWaistRight();
    else if (cmd=="WAIST_CENTRE")    moveWaistCentre();
    else if (cmd=="NEW_MELODY")      { attentionBlink(); drawNote(1); }
    else if (cmd=="FACE_HAPPY")      faceHappy();
    else if (cmd=="FACE_SAD")        faceSad();
    else if (cmd=="FACE_ANGRY")      faceAngry();
    else if (cmd=="FACE_ENERGETIC")  faceEnergetic();
    else if (cmd=="FACE_TIRED")      faceTired();
    else if (cmd=="FACE_CALM")       faceCalm();
    else if (cmd=="FACE_MISERABLE")  faceMiserable();
    else if (cmd=="FACE_CONFUSED")   faceConfused();
    else if (cmd=="FACE_BLUSHING")   faceBlushing();
    else if (cmd=="FACE_EXCITED")    faceExcited();
    else if (cmd=="IDLE_MODE")       { idling=true; mqttLog("[ROBOT] Idle mode"); }
    else if (cmd=="RESET")           { resetAll(); idling=true; }
    else                             mqttLog("[ROBOT] Unknown: "+cmd);
}

void onMqttMessage(char* topic, byte* payload, unsigned int length) {
    String cmd="";
    for (unsigned int i=0; i<length; i++) cmd+=(char)payload[i];
    handleCommand(cmd);
}

// ═════════════════════════════════════════════════════════════════════════════
//  SELF TEST
// ═════════════════════════════════════════════════════════════════════════════
void selfTest() {
    Serial.println(F("\n========== SELF TEST START =========="));

    Serial.println(F("[TEST] Idle face + blinking..."));
    unsigned long t = millis();
    while (millis() - t < 4000) { idleLoop(); networkTick(); }

    Serial.println(F("[TEST] Emotion: HAPPY  |  Move: DANCE_HAPPY"));  faceHappy();     danceHappy();
    Serial.println(F("[TEST] Emotion: EXCITED  |  Move: CELEBRATE"));  faceExcited();   celebrate();
    Serial.println(F("[TEST] Emotion: ANGRY  |  Move: HEAD_SHAKE"));   faceAngry();     headShake();
    Serial.println(F("[TEST] Emotion: SAD  |  Move: COMFORT"));        faceSad();       comfort();
    Serial.println(F("[TEST] Emotion: MISERABLE  |  Move: SLEEPY"));   faceMiserable(); sleepy();
    Serial.println(F("[TEST] Emotion: CONFUSED  |  Move: CONFUSED"));  faceConfused();  confused_move();
    Serial.println(F("[TEST] Emotion: ENERGETIC  |  Move: PUMP_UP"));  faceEnergetic(); pumpUp();
    Serial.println(F("[TEST] Emotion: TIRED  |  Move: SLOW_DOWN"));    faceTired();     slowDown();
    Serial.println(F("[TEST] Emotion: CALM  |  Move: GENTLE_NOD"));    faceCalm();      gentleNod();
    Serial.println(F("[TEST] Emotion: BLUSHING  |  Move: RAISE_HAND"));faceBlushing();  raiseHand();

    Serial.println(F("[TEST] Joint: LEFT ARM"));
    faceCalm(); raiseLeftArm(); delay(600); lowerLeftArm(); delay(300);
    Serial.println(F("[TEST] Joint: RIGHT ARM"));
    raiseRightArm(); delay(600); lowerRightArm(); delay(300);
    Serial.println(F("[TEST] Joint: BOTH ARMS"));
    raiseBothArms(); delay(600); lowerBothArms(); delay(300);
    Serial.println(F("[TEST] Joint: HEAD"));
    lookLeft(); delay(500); lookCentre(); delay(300); lookRight(); delay(500); lookCentre(); delay(300);
    Serial.println(F("[TEST] Joint: WAIST"));
    moveWaistLeft(); delay(500); moveWaistCentre(); delay(300); moveWaistRight(); delay(500); moveWaistCentre(); delay(300);
    Serial.println(F("[TEST] HEAD NOD"));     headNod();
    Serial.println(F("[TEST] BOTH ARMS UP")); faceExcited(); bothArmsUp();

    Serial.println(F("[TEST] OLED Notes"));
    const char* noteNames[] = {"C","D","E","F","G","A"};
    for (int n=1; n<=6; n++) { Serial.print("  Note: "); Serial.println(noteNames[n-1]); drawNote(n); delay(400); }

    Serial.println(F("[TEST] LEDs")); faceExcited(); attentionBlink();
    Serial.println(F("[TEST] RESET")); resetAll(500); drawIdleFace(1.0f, 0);
    Serial.println(F("========== SELF TEST COMPLETE ==========\n"));
}

// ═════════════════════════════════════════════════════════════════════════════
//  SETUP & LOOP
// ═════════════════════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(2000);
    
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) Serial.println(F("[OLED] Failed"));
    display.clearDisplay(); display.display();
    drawIdleFace(1.0f, 0);

    leftArm.attach(PIN_LEFT_ARM); rightArm.attach(PIN_RIGHT_ARM);
    head.attach(PIN_HEAD); waistA.attach(PIN_WAIST_A); waistB.attach(PIN_WAIST_B);
    pinMode(LED1_PIN, OUTPUT); pinMode(LED2_PIN, OUTPUT);

    leftArm.write(L_DOWN); rightArm.write(R_DOWN); head.write(H_CENTRE);
    waistA.write(W_CENTRE); waistB.write(WAIST_B_MIRROR ? (180-W_CENTRE) : W_CENTRE);
    delay(300);

    levelServo.attach(PIN_LEVEL);
    levelServo.write(0);

    // WiFi is intentionally NOT started here.
    // networkTick() owns the WiFi lifecycle to prevent the
    // "sta is connecting, cannot set config" double-init error.

    mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt.setCallback(onMqttMessage);

    //selfTest();

    idling = true;
}

void loop() {
    networkTick();
    if (idling) idleLoop();
    if (Serial.available()) {
        command = Serial.readStringUntil('\n');
        command.trim();
        handleCommand(command);
    }
}
