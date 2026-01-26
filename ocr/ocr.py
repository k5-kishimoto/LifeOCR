import os
import json
import io
import time
import re
import ast
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
        self.model = None
        
        if not self.api_key:
            print("❌ Error: 'GEMINI_API_KEY' not found.")
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model_name = os.environ.get("GEMINI_VERSION", "gemini-2.5-flash")
            
            self.generation_config = genai.types.GenerationConfig(
                temperature=0.0, 
                top_p=1.0,
                max_output_tokens=8192,
                response_mime_type="application/json"
            )

            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            print(f"⚙️ Initial Model config: {self.model_name} (Rectangular-Matrix Mode)")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    # =========================================================================
    # 🧹 テキスト処理 (ノイズ除去のみ)
    # =========================================================================
    
    def _clean_text(self, val):
        if val is None: return ""
        val = str(val).replace("\n", "").replace("\r", "")
        # OCR特有の誤認識文字（罫線など）をスペースに置換
        val = val.replace("■", " ").replace("□", " ").replace("図", " ")
        return re.sub(r'\s+', ' ', val).strip()

    # =========================================================================
    # 🖼️ 画像処理
    # =========================================================================

    def _optimize_image(self, img):
        max_size = 2560 
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = ImageOps.autocontrast(img, cutoff=2)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.4) 
        return img

    def _split_image(self, img):
        width, height = img.size
        split_ratio = 0.60
        overlap = 0.40 
        crop_top = img.crop((0, 0, width, int(height * split_ratio)))
        crop_bottom = img.crop((0, int(height * overlap), width, height))
        return [("Top", crop_top), ("Bottom", crop_bottom)]

    # =========================================================================
    # 🧠 データ解析
    # =========================================================================

    def _repair_json(self, text):
        if not text: return None
        try:
            cleaned = text.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:-3]
            elif cleaned.startswith("```"): cleaned = cleaned[3:-3]
            return json.loads(cleaned)
        except: pass
        
        # 簡易正規表現によるJSON抽出
        try:
            candidate_rows = re.findall(r'\[(.*?)\]', text, re.DOTALL)
            valid_rows = []
            for row_content in candidate_rows:
                if not row_content.strip(): continue
                try:
                    row_data = json.loads(f"[{row_content}]")
                    if isinstance(row_data, list): valid_rows.append(row_data)
                except: pass
            if valid_rows: return {"table_rows": valid_rows}
        except: pass
        return None

    def _call_ai_api(self, image_part, part_label):
        prompt = """
        あなたは高精度の日本語OCRエンジンです。画像内の表データを抽出してください。
        
        【重要命令】
        - **改行コード禁止。**
        - **半角カナはそのまま出力。**
        - セル内に複数の単語がある場合はスペースで区切ること。
        - 途中にある項目名（ヘッダー）も無視せず、データとして抽出してください。

        【出力フォーマット (JSON)】
        {
          "document_info": { "title": "タイトル", "org_name": "発行元", "period": "期間" },
          "table_headers": ["項目1", "項目2", ...],
          "table_rows": [ 
             ["データ1", "データ2", "データ3", ...],
          ]
        }
        """
        try:
            response = self.model.generate_content([prompt, image_part])
            return response.text
        except Exception as e:
            print(f"⚠️ API Error ({part_label}): {e}")
            return None

    # =========================================================================
    # 🔄 マージ & ★完全行列化 (全行の列数を統一)
    # =========================================================================

    def _merge_split_results(self, results):
        combined_rows = []
        seen_exact_rows = set()

        # 文書情報を最初に追加
        doc_info = results.get("Top", results.get("Bottom", {})).get("document_info", {})
        for k, v in doc_info.items():
            if v: combined_rows.append([f"{v}"])

        # 表データを追加（Top -> Bottom の順）
        for source in ["Top", "Bottom"]:
            if source not in results: continue
            
            # ヘッダーがあれば追加
            headers = results[source].get("table_headers", [])
            if headers: combined_rows.append([self._clean_text(h) for h in headers])
            
            # データ行を追加
            for row in results[source].get("table_rows", []):
                if not row or all(str(c).strip() == "" for c in row): continue
                cleaned_row = [self._clean_text(c) for c in row]
                
                # 完全一致する行のみ重複排除
                row_str = str(cleaned_row)
                if row_str not in seen_exact_rows:
                    seen_exact_rows.add(row_str)
                    combined_rows.append(cleaned_row)

        # ★【核心】最大列数を計算して、すべての行をパディングする
        max_cols = 0
        for row in combined_rows:
            max_cols = max(max_cols, len(row))
        
        # UIに渡す最終形式を作成
        final_ui_output = []
        for row in combined_rows:
            padded_row = row[:]
            while len(padded_row) < max_cols:
                padded_row.append("") # 足りない列を空文字で埋めて「枠」を確保
            
            final_ui_output.append([{'text': cell} for cell in padded_row])

        return final_ui_output

    def extract_text(self, uploaded_file):
        print(f"⏳ Starting AI OCR - Strict Matrix Mode...")
        if not self.model: return [[{'text': "Error: AI Model not initialized."}]]

        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        try:
            img = Image.open(io.BytesIO(file_bytes))
            optimized_img = self._optimize_image(img)
            parts = self._split_image(optimized_img)
            
            results = {}
            for p_name, p_img in parts:
                img_byte_arr = io.BytesIO()
                p_img.save(img_byte_arr, format='WEBP')
                image_part = {"mime_type": "image/webp", "data": img_byte_arr.getvalue()}
                
                res_text = self._call_ai_api(image_part, p_name)
                if res_text:
                    parsed = self._repair_json(res_text)
                    if parsed: results[p_name] = parsed

            return self._merge_split_results(results)

        except Exception as e:
            return [[{'text': f"Processing Error: {e}"}]]

engine = OcrEngine()