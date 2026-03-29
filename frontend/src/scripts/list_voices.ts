import { ElevenLabsClient } from "elevenlabs";
import * as dotenv from "dotenv";
import * as path from "path";

// Load the exact .env.local from the frontend directory
dotenv.config({ path: path.resolve(__dirname, "../../.env.local") });

async function listAvailableVoices() {
    const apiKey = process.env.ELEVENLABS_API_KEY;
    
    if (!apiKey) {
        console.error("❌ ERROR: ELEVENLABS_API_KEY not found in .env.local");
        return;
    }

    const client = new ElevenLabsClient({ apiKey });

    try {
        console.log("🔍 Fetching available voices for your account...");
        const response = await client.voices.getAll();
        
        console.log("\n--- AVAILABLE VOICES ---");
        response.voices.forEach(voice => {
            console.log(`Name: ${voice.name.padEnd(15)} | ID: ${voice.voice_id} | Category: ${voice.category}`);
        });
        console.log("------------------------\n");
        
    } catch (error) {
        console.error("❌ ERROR fetching voices:", error);
    }
}

listAvailableVoices();
