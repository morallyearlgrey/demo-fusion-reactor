import { NextRequest, NextResponse } from 'next/server';
import { ElevenLabsClient } from 'elevenlabs';

export async function POST(req: NextRequest) {
  try {
    const elevenlabs = new ElevenLabsClient({
      apiKey: process.env.ELEVENLABS_API_KEY,
    });

    const { text, voiceId, settings } = await req.json();

    if (!text || !voiceId) {
      return NextResponse.json({ error: 'Missing text or voiceId' }, { status: 400 });
    }

    if (!process.env.ELEVENLABS_API_KEY) {
      console.error("ELEVENLABS_API_KEY is not set in environment variables");
      return NextResponse.json({ error: 'Server configuration error' }, { status: 500 });
    }

    const audioStream = await elevenlabs.generate({
      voice: voiceId,
      text: text,
      model_id: "eleven_multilingual_v2",
      voice_settings: settings
    });

    // Convert the stream to a buffer to send in response
    const chunks = [];
    for await (const chunk of audioStream) {
      chunks.push(chunk);
    }
    const audioBuffer = Buffer.concat(chunks);

    return new NextResponse(audioBuffer, {
      headers: {
        'Content-Type': 'audio/mpeg',
      },
    });
  } catch (error: any) {
    console.error('ElevenLabs TTS Error:', error);
    return NextResponse.json({ error: error.message || 'Failed to generate speech' }, { status: 500 });
  }
}
