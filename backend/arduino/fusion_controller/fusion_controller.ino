/**
 * fusion_controller.ino
 *
 * Fusion Reactor DAC Controller
 * Reads JSON commands from serial (from Python serial bridge / AI pipeline),
 * controls MCP4728 DAC electrodes A and B, reads a photodiode ADC,
 * and streams JSON reading telemetry back over serial.
 *
 * Serial JSON INPUT format (from host):
 *   Command mode:  {"type":"command","cmd":"set_electrodes","a":<0-4095>,"b":<0-4095>}
 *                  {"type":"command","cmd":"set_electrode_a","value":<0-4095>}
 *                  {"type":"command","cmd":"set_electrode_b","value":<0-4095>}
 *                  {"type":"command","cmd":"set_params","target_adc":<int>,"max_delta":<int>,"spike_threshold":<int>,"low_threshold":<int>,"photo_min":<int>,"photo_max":<int>,"dac_min":<int>,"dac_max":<int>}
 *   Backup mode:   {"type":"backup"}  -> switches to auto mapConstrain loop
 *   AI mode:       {"type":"ai"}      -> switches to AI-driven mode (awaits commands)
 *
 * Serial JSON OUTPUT format (to host):
 *   {"type":"reading","raw_adc":<int>,"electrode_a":<int>,"electrode_b":<int>,"time_ms":<long>,"flag":"<auto|ai>"}
 */

#include <Wire.h>
#include <Adafruit_MCP4728.h>
#include <ArduinoJson.h>

Adafruit_MCP4728 mcp;

// ── Pin ──────────────────────────────────────────────────────────────────────
const int PHOTO_PIN = A0;
const int NUM_SAMPLES = 10;

// ── Runtime-tunable parameters (can be updated via set_params command) ───────
int PHOTO_MIN = 550;
int PHOTO_MAX = 820;
uint16_t DAC_MIN = 0;
uint16_t DAC_MAX = 4095;
int TARGET_ADC = 685;      // midpoint default
int MAX_DELTA = 200;       // max step change per cycle
int SPIKE_THRESHOLD = 900; // raw adc value considered a spike
int LOW_THRESHOLD = 100;   // raw adc value considered too low

// ── State ─────────────────────────────────────────────────────────────────────
uint16_t electrodeA = 0;
uint16_t electrodeB = 0;
bool aiMode = false; // false = auto/backup, true = AI-driven

// ── Helpers ───────────────────────────────────────────────────────────────────
int readAveragedAnalog(int pin, int samples)
{
    long total = 0;
    for (int i = 0; i < samples; i++)
    {
        total += analogRead(pin);
        delay(2);
    }
    return (int)(total / samples);
}

uint16_t mapConstrained(int x, int in_min, int in_max, uint16_t out_min, uint16_t out_max)
{
    x = constrain(x, in_min, in_max);
    return (uint16_t)((long)(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min);
}

void setElectrodes(uint16_t valueA, uint16_t valueB)
{
    valueA = constrain(valueA, DAC_MIN, DAC_MAX);
    valueB = constrain(valueB, DAC_MIN, DAC_MAX);
    mcp.setChannelValue(MCP4728_CHANNEL_A, valueA, MCP4728_VREF_VDD, MCP4728_GAIN_1X);
    mcp.setChannelValue(MCP4728_CHANNEL_B, valueB, MCP4728_VREF_VDD, MCP4728_GAIN_1X);
}

void emitReading(int rawAdc)
{
    StaticJsonDocument<200> doc;
    doc["type"] = "reading";
    doc["raw_adc"] = rawAdc;
    doc["electrode_a"] = (int)electrodeA;
    doc["electrode_b"] = (int)electrodeB;
    doc["time_ms"] = millis();
    doc["flag"] = aiMode ? "ai" : "auto";
    serializeJson(doc, Serial);
    Serial.println();
}

// ── Command parser ────────────────────────────────────────────────────────────
void processCommand(JsonDocument &doc)
{
    const char *type = doc["type"];
    if (!type)
        return;

    if (strcmp(type, "backup") == 0)
    {
        aiMode = false;
        return;
    }

    if (strcmp(type, "ai") == 0)
    {
        aiMode = true;
        return;
    }

    if (strcmp(type, "command") == 0)
    {
        const char *cmd = doc["cmd"];
        if (!cmd)
            return;

        if (strcmp(cmd, "set_electrodes") == 0)
        {
            uint16_t a = (uint16_t)((int)doc["a"]);
            uint16_t b = (uint16_t)((int)doc["b"]);
            electrodeA = a;
            electrodeB = b;
            setElectrodes(electrodeA, electrodeB);
        }
        else if (strcmp(cmd, "set_electrode_a") == 0)
        {
            electrodeA = (uint16_t)((int)doc["value"]);
            setElectrodes(electrodeA, electrodeB);
        }
        else if (strcmp(cmd, "set_electrode_b") == 0)
        {
            electrodeB = (uint16_t)((int)doc["value"]);
            setElectrodes(electrodeA, electrodeB);
        }
        else if (strcmp(cmd, "set_params") == 0)
        {
            // Update runtime-tunable thresholds / ranges sent by the improvement agent
            if (doc.containsKey("target_adc"))
                TARGET_ADC = doc["target_adc"];
            if (doc.containsKey("max_delta"))
                MAX_DELTA = doc["max_delta"];
            if (doc.containsKey("spike_threshold"))
                SPIKE_THRESHOLD = doc["spike_threshold"];
            if (doc.containsKey("low_threshold"))
                LOW_THRESHOLD = doc["low_threshold"];
            if (doc.containsKey("photo_min"))
                PHOTO_MIN = doc["photo_min"];
            if (doc.containsKey("photo_max"))
                PHOTO_MAX = doc["photo_max"];
            if (doc.containsKey("dac_min"))
                DAC_MIN = (uint16_t)((int)doc["dac_min"]);
            if (doc.containsKey("dac_max"))
                DAC_MAX = (uint16_t)((int)doc["dac_max"]);
        }
    }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup()
{
    Serial.begin(115200);
    while (!Serial)
    {
        delay(10);
    }

    if (!mcp.begin())
    {
        // Emit error as JSON so the bridge can detect it
        Serial.println("{\"type\":\"error\",\"msg\":\"MCP4728 not found\"}");
        while (1)
        {
            delay(100);
        }
    }

    analogReference(DEFAULT);
    setElectrodes(0, 0);

    // Signal ready
    Serial.println("{\"type\":\"ready\",\"msg\":\"fusion_controller online\"}");
}

// ── Main loop ─────────────────────────────────────────────────────────────────
void loop()
{
    // ── 1. Check for incoming JSON commands ───────────────────────────────────
    if (Serial.available())
    {
        String line = Serial.readStringUntil('\n');
        line.trim();
        if (line.length() > 0)
        {
            StaticJsonDocument<256> inDoc;
            DeserializationError err = deserializeJson(inDoc, line.c_str());
            if (!err)
            {
                processCommand(inDoc);
            }
        }
    }

    // ── 2. Read sensor ────────────────────────────────────────────────────────
    int rawAdc = readAveragedAnalog(PHOTO_PIN, NUM_SAMPLES);

    // ── 3. Auto (backup) mode: mapConstrain drives electrodes ─────────────────
    if (!aiMode)
    {
        uint16_t dacValue = mapConstrained(rawAdc, PHOTO_MIN, PHOTO_MAX, DAC_MIN, DAC_MAX);
        electrodeA = dacValue;
        electrodeB = dacValue;
        setElectrodes(electrodeA, electrodeB);
    }
    // In AI mode, electrodes are only changed via set_electrodes commands.

    // ── 4. Emit telemetry reading ─────────────────────────────────────────────
    emitReading(rawAdc);

    delay(50);
}
