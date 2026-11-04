"""
explore_response.py
Response object ke andar level by level jhaankna 🔍
"""

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.8-flash-lite-tts",
    contents="Say hello to Mayank in one short line.",
)

print("=" * 60)
print("Level 1: response")
print("   type :", type(response).__name__)

print("\nLevel 2: response.candidates")
print("   type :", type(response.candidates).__name__)
print("   kitne jawab:", len(response.candidates))

candidate = response.candidates[0]
print("\nLevel 3: response.candidates[0]")
print("   type          :", type(candidate).__name__)
print("   finish_reason :", candidate.finish_reason)

content = candidate.content
print("\nLevel 4: .content")
print("   type :", type(content).__name__)
print("   role :", content.role)

parts = content.parts
print("\nLevel 5: .parts")
print("   type        :", type(parts).__name__)
print("   kitne tukde :", len(parts))

print("\nLevel 6 & 7: har ek part ke andar")
for i, part in enumerate(parts):
    print(f"\n   --- part [{i}] ---")
    print("   type        :", type(part).__name__)
    print("   text        :", part.text)
    if part.inline_data:
        print("   inline_data : HAI ✅")
        print("   mime_type   :", part.inline_data.mime_type)
        print("   data size   :", len(part.inline_data.data), "bytes")
    else:
        print("   inline_data : None")

print("=" * 60)