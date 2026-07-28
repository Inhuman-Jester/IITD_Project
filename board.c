#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wpa2.h"
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

const char* ssid = "IITD_WIFI"; 
#define EAP_IDENTITY "cs1221077"
#define EAP_PASSWORD "55516273"

#define OLED_RESET 4   // ← tie GPIO4 to OLED RST pin on breadboard, or -1 if not wired

WiFiUDP udp;
unsigned int localPort = 4210; 
char packetBuffer[255]; 

Adafruit_SH1106G display = Adafruit_SH1106G(128, 64, &Wire, OLED_RESET);

// --- UI QUEUE SYSTEM ---
#define MAX_QUEUE 50
String statusQueue[MAX_QUEUE];
String nameQueue[MAX_QUEUE];
unsigned long durationQueue[MAX_QUEUE];   // ← per-item duration in ms
int qHead = 0;
int qTail = 0;
int qSize = 0;
unsigned long displayStartTime = 0;
unsigned long currentDuration = 2500;
bool isShowingResult = false;

void setup() {
  Serial.begin(115200);

  // FIX 1: Give OLED time to stabilize after power-on
  delay(500);

  Wire.begin();
  Wire.setClock(400000);  // fast I2C — more reliable init

  // Explicit reset pulse if OLED_RESET pin is wired
  #if OLED_RESET > 0
    pinMode(OLED_RESET, OUTPUT);
    digitalWrite(OLED_RESET, LOW);
    delay(20);
    digitalWrite(OLED_RESET, HIGH);
    delay(20);
  #endif

  if (!display.begin(0x3C, true)) {
    Serial.println("OLED init failed! Check wiring.");
    while (true) { delay(1000); }  // halt so you know immediately
  }

  updateDisplay("BOOTING", "Connecting Wi-Fi", false);

  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);
  esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)EAP_IDENTITY, strlen(EAP_IDENTITY));
  esp_wifi_sta_wpa2_ent_set_username((uint8_t *)EAP_IDENTITY, strlen(EAP_IDENTITY));
  esp_wifi_sta_wpa2_ent_set_password((uint8_t *)EAP_PASSWORD, strlen(EAP_PASSWORD));
  esp_wifi_sta_wpa2_ent_enable();
  WiFi.begin(ssid);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.println(WiFi.localIP());

  udp.begin(localPort);
  updateDisplay("ONLINE", WiFi.localIP().toString(), false);
  delay(4000);
  updateDisplay("READY", "Scanning...", false);
}

void loop() {
  // 1. Read incoming UDP packets
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(packetBuffer, 255);
    if (len > 0) packetBuffer[len] = 0;

    String data = String(packetBuffer);
    int separatorIndex = data.indexOf(':');
    if (separatorIndex != -1) {
      String status = data.substring(0, separatorIndex);
      String name   = data.substring(separatorIndex + 1);

      // FIX 2: Assign duration based on message type
      unsigned long duration;
      if (status == "MARKED") {
        duration = 6000;
      } else if (status == "SPOOF") {
        duration = 2000;
      } else {
        duration = 2500;  // default for anything else
      }

      if (qSize < MAX_QUEUE) {
        statusQueue[qTail]   = status;
        nameQueue[qTail]     = name;
        durationQueue[qTail] = duration;
        qTail = (qTail + 1) % MAX_QUEUE;
        qSize++;
      }
    }
  }

  // 2. Display management
  if (!isShowingResult && qSize > 0) {
    bool isSuccess = (statusQueue[qHead] == "SUCCESS");
    updateDisplay(statusQueue[qHead], nameQueue[qHead], isSuccess);
    currentDuration  = durationQueue[qHead];
    qHead = (qHead + 1) % MAX_QUEUE;
    qSize--;
    displayStartTime = millis();
    isShowingResult  = true;
  }

  if (isShowingResult && (millis() - displayStartTime > currentDuration)) {
    isShowingResult = false;
    if (qSize == 0) {
      updateDisplay("READY", "Scanning...", false);
    }
  }
}

// FIX 3: largeSubtext=true uses size 2 for name, false uses size 1
void updateDisplay(String status, String subtext, bool largeSubtext) {
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);

  // Header
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("IITD ACCESS CONTROL");
  display.drawLine(0, 12, 128, 12, SH110X_WHITE);

  // Status (large)
  display.setTextSize(2);
  display.setCursor(0, 18);
  display.println(status);

  // Subtext — large for SUCCESS/SPOOF, small otherwise
  if (largeSubtext) {
    display.setTextSize(2);
    display.setCursor(0, 42);
  } else {
    display.setTextSize(1);
    display.setCursor(0, 56);
  }
  display.println(subtext);

  display.display();
}