"""
save_audio.py
TTS model ka audio jawab file mein save karna 🔊
"""

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.8-flash-lite-tts",
    contents="Hello Mayank! Welcome to your Agentic AI journey. Day one is done!",
)

# Pehle jawab ke pehle tukde tak jao 🪜
part = response.candidates[0].content.parts[0]

if part.inline_data:
    print("🔊 Audio mila! Type:", part.inline_data.mime_type)

    # Bytes ko file mein likho
    with open("hello_mayank.wav", "wb") as f:
        f.write(part.inline_data.data)

    print("✅ Save ho gaya: hello_mayank.wav")
else:
    print("❌ Audio nahi mila. Text:", part.text)