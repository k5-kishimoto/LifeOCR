import os
import json
import io
import re
import concurrent.futures
from pdf2image import convert_from_bytes
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image, ImageOps 
from dotenv import load_dotenv

load_dotenv()

class OcrEngine:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key: return
        try:
            genai.configure(api_key=self.api_key)
            self.model_name = os.environ.get("GEMINI_VERSION", "gemini-2.0-flash")
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.types.GenerationConfig(temperature=0.0, response_mime_type="application/json"),
                safety_settings={cat: HarmBlockThreshold.BLOCK_NONE for cat in HarmCategory}
            )
        except Exception as e:
            print(f"❌ Initial Error: {e}")

    def _clean_text(self, val):
        if val is None: return ""
        val = str(val).replace("\n", " ").replace("\r", " ")
        val = val.replace("■", "").replace("□", "").replace("図", "")
        return re.sub(r'\s+', ' ', val).strip()

    def _call_ai_api(self, image_part):
        prompt = """
        あなたは高精度のOCRエンジンです。画像内の表データを抽出してください。
        
        【抽出ルール】
        - 半角カナは維持。
        - 項目名（ヘッダー）と明細データをすべて抽出。
        - 右端にある「摘要」や「名義」のカナを絶対に漏らさないこと。

        【JSON形式】
        {
          "rows": [ 
             ["セル1", "セル2", "セル3"],
             ["データ1", "データ2", "データ3"]
          ]
        }
        """
        try:
            response = self.model.generate_content([prompt, image_part])
            return json.loads(response.text.strip().replace("```json", "").replace("```", ""))
        except:
            return None

    def extract_text(self, uploaded_file):
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        try:
            if uploaded_file.name.lower().endswith('.pdf'):
                images = convert_from_bytes(file_bytes, dpi=200)
            else:
                images = [Image.open(io.BytesIO(file_bytes))]
        except:
            return [[{'text': "ファイル読み込みエラー"}]]

        all_raw_data = []

        for i, img in enumerate(images):
            # ページ分割読み取りを復活（精度向上のため）
            w, h = img.size
            parts = [img.crop((0, 0, w, int(h * 0.6))), img.crop((0, int(h * 0.4), w, h))]
            
            # 各ページの「Page X」見出しもデータとして保持
            all_raw_data.append([f"--- Page {i+1} ---"])

            for part in parts:
                part_img = ImageOps.autocontrast(part.convert('RGB'), cutoff=1)
                buf = io.BytesIO()
                part_img.save(buf, format='WEBP')
                
                res = self._call_ai_api({"mime_type": "image/webp", "data": buf.getvalue()})
                if res and "rows" in res:
                    for r in res["rows"]:
                        cleaned = [self._clean_text(c) for c in r]
                        # 右側の空列ゴミを除去
                        while cleaned and not cleaned[-1]:
                            cleaned.pop()
                        if cleaned:
                            all_raw_data.append(cleaned)

        if not all_raw_data:
            return [[{'text': "データが抽出されませんでした"}]]

        # --- 🚀 解決の核心：全行の列数を最大幅に強制パディング ---
        max_cols = max(len(row) for row in all_raw_data)
        
        final_ui_rows = []
        for row in all_raw_data:
            # 1列しかない行にも空セルを足して「長方形」にする
            # これでUI側が「7列ある表」と認識し、カナ列を表示します
            padded_row = row + [""] * (max_cols - len(row))
            final_ui_rows.append([{'text': cell} for cell in padded_row])

        return final_ui_rows

engine = OcrEngine()