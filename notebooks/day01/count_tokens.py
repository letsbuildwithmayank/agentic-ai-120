"""
count_tokens.py
Same baat, alag bhasha: kitne tokens lagte hain? 🔢
"""

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

MODEL = "gemini-3.8-flash-lite-tts"

sentences = {
    "English ": "My name is Mayank and I am learning AI.",
    "Hinglish": "Mera naam Mayank hai aur main AI seekh raha hoon.",
    "Hindi   ": "मेरा नाम मयंक है और मैं AI सीख रहा हूँ।",
}

print("🔢 Token count comparison:\n")

for language, text in sentences.items():
    result = client.models.count_tokens(model=MODEL, contents=text)
    words = len(text.split())
    print(f"   {language} | {words} words | {result.total_tokens} tokens")
