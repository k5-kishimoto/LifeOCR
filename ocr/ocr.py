import os
import json
import io
import re
import concurrent.futures
from pdf2image import convert_from_bytes
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image, ImageEnhance, ImageOps 
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
            print(f"⚙️ Mode: Table-Optimizer Mode")
        except Exception as e:
            print(f"❌ Error: {e}")

    def _clean_text(self, val):
        if val is None: return ""
        val = str(val).replace("\n", " ").replace("\r", " ")
        # 記号ノイズの除去
        val = val.replace("■", "").replace("□", "").replace("図", "")
        return re.sub(r'\s+', ' ', val).strip()

    def _call_ai_api(self, image_part):
        prompt = """
        あなたは高精度のOCRエンジンです。画像内の表データを抽出してください。
        
        【ルール】
        - 改行禁止。
        - 半角カナは維持。
        - 画像内のすべての行、すべての列を漏れなく抽出してください。

        【JSON形式】
        {
          "table_data": [ 
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

    def _process_rows(self, raw_rows):
        """
        行データをクリーンアップし、末尾の空セルを除去して行列化する
        """
        cleaned_table = []
        for row in raw_rows:
            # 各セルのクリーニング
            cleaned_row = [self._clean_text(cell) for cell in row]
            
            # ★重要：行の末尾にある「空の要素」をすべて削除する
            # これをしないと、UI側でカナが右側に追いやられて消える
            while cleaned_row and not cleaned_row[-1]:
                cleaned_row.pop()
            
            if cleaned_row:
                cleaned_table.append(cleaned_row)

        if not cleaned_table: return []

        # 全体の中で最大列数を把握（パディング用）
        max_cols = max(len(row) for row in cleaned_table)
        
        final_matrix = []
        for row in cleaned_table:
            # 長さを揃える（短い行にだけ空文字を足す）
            padded_row = row + [""] * (max_cols - len(row))
            final_matrix.append([{'text': cell} for cell in padded_row])
            
        return final_matrix

    def extract_text(self, uploaded_file):
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        try:
            # PDF/画像 変換
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

            # 画像最適化（コントラスト調整のみ）
            img = ImageOps.autocontrast(img.convert('RGB'), cutoff=1)
            
            # 分割せずに1ページ丸ごと投げる（ズレ防止）
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='WEBP')
            
            res_data = self._call_ai_api({"mime_type": "image/webp", "data": img_byte_arr.getvalue()})
            if res_data and "table_data" in res_data:
                processed_page = self._process_rows(res_data["table_data"])
                all_results.extend(processed_page)

        return all_results

engine = OcrEngine()