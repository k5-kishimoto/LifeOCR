import os
import json
import io
import time
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
        self.model = None
        
        if not self.api_key:
            print("❌ Error: 'GEMINI_API_KEY' not found.")
            return

        try:
            genai.configure(api_key=self.api_key)
            # 安定性の高い2.0-flashを使用
            self.model_name = os.environ.get("GEMINI_VERSION", "gemini-2.0-flash")
            
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
            print(f"⚙️ Initial Model config: {self.model_name} (Rectangular-Output Mode)")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    # =========================================================================
    # 🧹 テキスト処理
    # =========================================================================
    
    def _clean_text(self, val):
        if val is None: return ""
        val = str(val).replace("\n", "").replace("\r", "")
        # OCR特有の誤認識ノイズをスペースに
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
        img = ImageOps.autocontrast(img, cutoff=1)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.3) 
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
        あなたは日本語OCRエンジンです。画像から表データを抽出してください。
        
        【命令】
        - 改行コード禁止。
        - 半角カナはそのまま。
        - 項目名（ヘッダー）もデータの一部として、上から順にすべて抽出してください。

        【出力フォーマット (JSON)】
        {
          "document_info": { "line1": "タイトルやヘッダー以外の情報" },
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
    # 🔄 マージ & ★完全長方形化 (パディング)
    # =========================================================================

    def _merge_and_pad(self, page_results):
        raw_output_list = []
        seen_exact_rows = set()

        for res in page_results:
            # 1. document_info (タイトル等) をリストに追加
            doc_info = res.get("document_info", {})
            for v in doc_info.values():
                if v: raw_output_list.append([self._clean_text(v)])

            # 2. table_headers を追加
            headers = res.get("table_headers", [])
            if headers:
                raw_output_list.append([self._clean_text(h) for h in headers])

            # 3. table_rows を追加
            for row in res.get("table_rows", []):
                if not row or all(str(c).strip() == "" for c in row): continue
                cleaned_row = [self._clean_text(c) for c in row]
                
                # 文字列として完全に一致する行のみ重複排除
                row_str = str(cleaned_row)
                if row_str not in seen_exact_rows:
                    seen_exact_rows.add(row_str)
                    raw_output_list.append(cleaned_row)

        if not raw_output_list: return []

        # ★【核心】最大列数を算出し、全行をその長さに揃える
        max_cols = max(len(row) for row in raw_output_list)
        max_cols = max(max_cols, 1)

        final_rows = []
        for row in raw_output_list:
            padded = row[:]
            while len(padded) < max_cols:
                padded.append("") # 空セルを追加して長さを統一
            final_rows.append([{'text': cell} for cell in padded])

        return final_rows

    # =========================================================================
    # 🚀 メイン処理
    # =========================================================================

    def extract_text(self, uploaded_file):
        print(f"⏳ Starting AI OCR - Strict Matrix Mode...")
        if not self.model: return [[{'text': "Error: AI Model not initialized."}]]

        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        # ファイル形式の判別
        try:
            filename = uploaded_file.name.lower()
        except:
            filename = "unknown.jpg"

        pil_images = []
        try:
            if filename.endswith('.pdf'):
                pil_images = convert_from_bytes(file_bytes, dpi=200)
            else:
                pil_images = [Image.open(io.BytesIO(file_bytes))]
        except Exception as e:
            return [[{'text': f"❌ File Recognition Error: {e}"}]]

        all_pages_output = []

        for i, img in enumerate(pil_images):
            page_label = f"Page {i+1}"
            optimized_img = self._optimize_image(img)
            parts = self._split_image(optimized_img)
            
            page_data_list = []
            for p_name, p_img in parts:
                img_byte_arr = io.BytesIO()
                p_img.save(img_byte_arr, format='WEBP')
                image_part = {"mime_type": "image/webp", "data": img_byte_arr.getvalue()}
                
                res_text = self._call_ai_api(image_part, f"{page_label}-{p_name}")
                if res_text:
                    parsed = self._repair_json(res_text)
                    if parsed: page_data_list.append(parsed)

            padded_page = self._merge_and_pad(page_data_list)
            if len(pil_images) > 1:
                all_pages_output.append([{'text': f"--- {page_label} ---"}] + ([{'text': ''}] * (len(padded_page[0])-1 if padded_page else 0)))
            all_pages_output.extend(padded_page)

        return all_pages_output

engine = OcrEngine()