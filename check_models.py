import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ API Key not found")
else:
    genai.configure(api_key=api_key)
    print("--- Available Models ---")
    try:
        for m in genai.list_models():
            # 画像認識（generateContent）が使えるモデルだけ表示
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ {m.name}")
    except Exception as e:
        print(f"Error: {e}")