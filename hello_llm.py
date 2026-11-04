"""
hello_llm.py
Day 1: Gemini se pehli baat + token usage dekhna
Fallback ke saath: ek model busy ho to agla try karo
"""

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors

# 1. .env file se keys load karo
load_dotenv()

# 2. Check karo ki key mili ya nahi
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY nahi mili! .env file check karo.")
    sys.exit(1)

# 3. Models ki list: pehla busy ho to dusra, dusra busy ho to teesra
MODELS = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.8-flash-lite-tts",
]

# Ye error codes "thodi der baad / dusra model try karo" wale hain
RETRYABLE_CODES = (429, 503)

# 4. Gemini ka client banao (ye apne aap GEMINI_API_KEY use karega)
client = genai.Client()

# 5. Sawaal
question = "Explain what an AI agent is in 3 simple lines."
print(f"🙋 Question: {question}\n")

response = None
used_model = None

# 6. Har model ek-ek karke try karo
for model in MODELS:
    try:
        print(f"⏳ Trying model: {model} ...")
        response = client.models.generate_content(
            model=model,
            contents=question,
        )
        used_model = model
        break  # jawab mil gaya, loop se bahar niklo
    except errors.APIError as e:
        if e.code in RETRYABLE_CODES:
            print(
                f"⚠️  {model} busy hai (error {e.code}), agla model try kar raha hoon...\n"
            )
            continue  # agla model try karo
        # Koi aur error hai (jaise galat key) to ruk jao aur dikhao
        print(f"❌ Error {e.code}: {e.message}")
        sys.exit(1)

# 7. Koi bhi model nahi chala
if response is None:
    print("❌ Saare models busy hain. 2-3 minute baad dobara try karo.")
    sys.exit(1)

# 8. Jawab print karo
print(f"\n✅ Answered by: {used_model}\n")
print("🤖 Answer:")
print(response.text)

# 9. Token usage print karo
usage = response.usage_metadata
input_tokens = usage.prompt_token_count or 0
output_tokens = usage.candidates_token_count or 0
thinking_tokens = usage.thoughts_token_count or 0
total_tokens = usage.total_token_count or 0

print("\n📊 Token Usage:")
print(f"   Input tokens    : {input_tokens}")
print(f"   Output tokens   : {output_tokens}")
print(f"   Thinking tokens : {thinking_tokens}")
print(f"   Total tokens    : {total_tokens}")

# from dotenv import load_dotenv
# from google import genai

# load_dotenv()

# client = genai.Client()

# print("📋 Tumhari key se ye Flash models available hain:\n")

# for model in client.models.list():
#     if "3.8" in model.name:
#         print("  ", model.name)
