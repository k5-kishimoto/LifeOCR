import os
import json
import io
import time
import re
import concurrent.futures
from pdf2image import convert_from_bytes
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageOps 
from dotenv import load_dotenv

load_dotenv()

class OcrEngine:
    def __init__(self):
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
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
            print(f"⚙️ Initial Model config: {self.model_name} (Robust Text Mode)")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    def _optimize_image(self, img):
        max_size = 2560 
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = ImageOps.autocontrast(img, cutoff=1)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.4) 
        
        return img

    def _process_single_page(self, args):
        page_label, pil_image = args
        
        optimized_image = self._optimize_image(pil_image)
        
        img_byte_arr = io.BytesIO()
        optimized_image.save(img_byte_arr, format='WEBP', quality=90)
        img_bytes = img_byte_arr.getvalue()
        
        image_part = {"mime_type": "image/webp", "data": img_bytes}

        prompt = """
        あなたは高精度の日本語OCRエンジンです。
        提供された画像からテキスト情報を抽出し、JSONデータを返してください。

        【最重要ルール：文字種の維持】
        1. **半角カナは「半角」のまま出力すること**: `ﾌﾘｺﾐ` は `ﾌﾘｺﾐ` のまま。
        2. **記号の維持**: `ｶ)` や `ﾋ)` もそのまま。
        
        【タスク】
        1. **文書情報**: タイトル、銀行名、支店名、口座名義、期間などを抽出。
        2. **表データ**: 明細行をすべて抽出。

        【出力フォーマット (JSON)】
        {
          "document_info": {
             "title": "文書タイトル",
             "bank_name": "銀行名",
             "account_name": "口座名義",
             "other_info": "その他メタデータ"
          },
          "table_headers": ["日付", "摘要", "お支払金額", "お預り金額", "差引残高", "取扱店"],
          "table_rows": [
             ["2026-01-22", "ﾌﾘｺﾐ ﾔﾏﾀﾞﾀﾛｳ", "10,000", "", "50,000", "本店"]
          ]
        }
        """

        retry_models = [
            self.model_name,
            'gemini-2.5-pro',
            'gemini-2.0-flash'
        ]
        
        retry_models = list(dict.fromkeys(retry_models))

        for current_model_name in retry_models:
            try:
                current_model = genai.GenerativeModel(
                    current_model_name,
                    generation_config=self.generation_config
                )
                
                response = current_model.generate_content([prompt, image_part])
                raw_text = response.text
                
                formatted_rows = []

                try:
                    cleaned_text = raw_text.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:-3]
                    elif cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text[3:-3]

                    parsed_json = json.loads(cleaned_text)
                    
                    # 1. 文書情報
                    doc_info = parsed_json.get("document_info", {})
                    
                    # ★修正箇所: 強制的に文字列化してエラーを防ぐ関数
                    def safe_str(val):
                        if val is None: return ""
                        if isinstance(val, (dict, list)):
                            return str(val) # 辞書やリストならそのまま文字列表現にする
                        return str(val).strip()

                    if doc_info.get("title"):
                        formatted_rows.append([{'text': f"■ {safe_str(doc_info['title'])}", 'is_header': True}])
                    
                    meta_texts = []
                    if doc_info.get("bank_name"): meta_texts.append(safe_str(doc_info["bank_name"]))
                    if doc_info.get("account_name"): meta_texts.append(f"名義: {safe_str(doc_info['account_name'])}")
                    if doc_info.get("other_info"): meta_texts.append(safe_str(doc_info["other_info"]))
                    
                    if meta_texts:
                        # ここで join する要素が全て文字列であることが保証されるためエラーにならない
                        formatted_rows.append([{'text': " / ".join(meta_texts)}])
                        formatted_rows.append([{'text': ""}])

                    # 2. ヘッダー
                    headers = parsed_json.get("table_headers", [])
                    if headers:
                        # ヘッダー内も辞書が混じらないようにケア
                        clean_headers = [safe_str(h) for h in headers]
                        formatted_rows.append([{'text': h, 'is_header': True} for h in clean_headers])

                    # 3. データ
                    rows = parsed_json.get("table_rows", [])
                    for row in rows:
                        def clean_text(val):
                            if val is None: return ""
                            if isinstance(val, (dict, list)): return str(val) # ここも安全策
                            s = str(val).strip()
                            if s.lower() in ["null", "none"]: return ""
                            return s

                        if isinstance(row, list):
                            formatted_cells = [{'text': clean_text(cell)} for cell in row]
                        else:
                            formatted_cells = [{'text': clean_text(row)}]
                        formatted_rows.append(formatted_cells)

                except (json.JSONDecodeError, ValueError) as json_err:
                    print(f"⚠️ JSON Parse Error on {page_label}: {json_err}. Fallback.")
                    lines = raw_text.split('\n')
                    for line in lines:
                        if line.strip():
                            formatted_rows.append([{'text': line.strip()}])
                
                print(f"✅ Success ({page_label}) with {current_model_name}")
                return (page_label, formatted_rows)

            except Exception as e:
                error_msg = str(e)
                print(f"⚠️ Failed ({page_label}) with {current_model_name}: {error_msg}")
                if "404" in error_msg or "not found" in error_msg or "429" in error_msg or "500" in error_msg:
                    continue
                else:
                    return (page_label, [[{'text': f"Error: {error_msg}"}]])

        return (page_label, [[{'text': "Failed to extract text."}]])


    def extract_text(self, uploaded_file):
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Robust Mode...")
        
        if not self.model:
            return [[{'text': "Error: AI Model not initialized."}]]

        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        try:
            filename = uploaded_file.name.lower()
        except AttributeError:
            filename = "unknown.jpg"
            
        images_to_process = [] 

        if filename.endswith('.pdf'):
            try:
                pil_images = convert_from_bytes(file_bytes, dpi=300, fmt='jpeg')
                for i, img in enumerate(pil_images):
                    images_to_process.append((f"Page {i+1}", img))
            except Exception as e:
                print(f"❌ PDF Error: {e}")
                return [[{'text': f"PDF Error: {e}"}]]
        else:
            img = Image.open(io.BytesIO(file_bytes))
            images_to_process.append(("Page 1", img))

        final_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_page = {executor.submit(self._process_single_page, item): item[0] for item in images_to_process}
            
            results_dict = {}
            for future in concurrent.futures.as_completed(future_to_page):
                page_label, page_data = future.result()
                results_dict[page_label] = page_data

        for label, _ in images_to_process:
            if len(images_to_process) > 1:
                final_results.append([{'text': f'--- {label} ---', 'is_header': True}])
            
            if label in results_dict:
                final_results.extend(results_dict[label])

        return final_results

engine = OcrEngine()