import os
import json
import io
import re
from pdf2image import convert_from_bytes
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image, ImageOps 
from dotenv import load_dotenv

load_dotenv()

class OcrEngine:
    def __init__(self):
        """初期化"""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key: return

        try:
            genai.configure(api_key=self.api_key)
            self.model_name = os.environ.get("GEMINI_VERSION", "gemini-2.0-flash")
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0, 
                    response_mime_type="application/json"
                ),
                safety_settings={cat: HarmBlockThreshold.BLOCK_NONE for cat in HarmCategory}
            )
            print(f"⚙️ Mode: Trailing-Trim Optimizer")
        except Exception as e:
            print(f"❌ Error: {e}")

    def _clean_text(self, val):
        if val is None: return ""
        val = str(val).replace("\n", " ").replace("\r", " ")
        # 記号ノイズの除去
        val = val.replace("■", "").replace("□", "").replace("図", "")
        return re.sub(r'\s+', ' ', val).strip()

    def _call_ai_api(self, image_part):
        # AIに対して「余計な空列を作らない」ように指示を強化
        prompt = """
        あなたは高精度のOCRエンジンです。画像内の表データを抽出してください。
        
        【ルール】
        - 改行禁止。
        - 半角カナは維持。
        - **重要：画像に存在しない空の列（空の要素）を末尾に追加しないでください。**
        - 各行、中身がある列までで出力を止めてください。

        【JSON形式】
        {
          "table_data": [ 
             ["項目1", "項目2", "項目3"],
             ["データ1", "データ2", "データ3"]
          ]
        }
        """
        try:
            response = self.model.generate_content([prompt, image_part])
            return json.loads(response.text.strip().replace("```json", "").replace("```", ""))
        except:
            return None

    def _process_rows(self, raw_rows):
        """
        末尾の空セルを物理的に削除し、最小限の列数でマトリックス化する
        """
        cleaned_table = []
        for row in raw_rows:
            # 1. 各セルのクリーニング
            cleaned_row = [self._clean_text(cell) for cell in row]
            
            # 2. ★重要：行の右端（末尾）から空文字を削除していく
            # 摘要カナ（7列目）より右にあるゴミを一掃します
            while cleaned_row and not cleaned_row[-1]:
                cleaned_row.pop()
            
            if cleaned_row:
                cleaned_table.append(cleaned_row)

        if not cleaned_table: return []

        # 3. 有効な最大列数にパディング（揃える）
        max_cols = max(len(row) for row in cleaned_table)
        
        final_matrix = []
        for row in cleaned_table:
            # 足りない列だけを補完（今回のケースでは7列に揃うはず）
            padded_row = row + [""] * (max_cols - len(row))
            final_ui_row = [{'text': cell} for cell in padded_row]
            final_matrix.append(final_ui_row)
            
        return final_matrix

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

        all_results = []
        for i, img in enumerate(images):
            if len(images) > 1:
                all_results.append([{'text': f"--- Page {i+1} ---"}])

            img = ImageOps.autocontrast(img.convert('RGB'), cutoff=1)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='WEBP')
            
            res_data = self._call_ai_api({"mime_type": "image/webp", "data": img_byte_arr.getvalue()})
            if res_data and "table_data" in res_data:
                processed_page = self._process_rows(res_data["table_data"])
                all_results.extend(processed_page)

        return all_results

engine = OcrEngine()