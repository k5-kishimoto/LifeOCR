import os
import json
import io
import time
import re
import concurrent.futures
from pdf2image import convert_from_bytes
import google.generativeai as genai
# ★ ImageOps を追加（自動補正用）
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
            
            # JSONモードの設定
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
            print(f"⚙️ Initial Model config: {self.model_name} (Auto-Enhance Mode)")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    def _optimize_image(self, img):
        """
        ★ スマート画像補正
        画像ごとに最適な濃さを自動計算し、副作用なく視認性を上げます。
        """
        # 1. サイズ調整
        max_size = 2560 
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # RGBモードを確認（オートコントラストに必要）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 2. ★自動コントラスト調整 (Auto Contrast)
        # 画像の最も暗い部分を「完全な黒」、明るい部分を「完全な白」に引き伸ばします。
        # これにより、薄いグレーの文字は「黒」に近づき、元々黒い文字はそのまま維持されます。
        # cutoff=1 は、ノイズ（極端な点）を1%無視して計算する設定です。
        img = ImageOps.autocontrast(img, cutoff=1)
        
        # 3. 控えめなシャープネス (1.5倍は強すぎるので1.1倍に)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1) 
        
        return img

    def _process_single_page(self, args):
        page_label, pil_image = args
        
        # スマート補正の適用
        optimized_image = self._optimize_image(pil_image)
        
        img_byte_arr = io.BytesIO()
        optimized_image.save(img_byte_arr, format='WEBP', quality=85)
        img_bytes = img_byte_arr.getvalue()
        
        image_part = {"mime_type": "image/webp", "data": img_bytes}

        prompt = """
        You are an expert OCR engine for Japanese bank statements (通帳/明細).
        
        [Task]
        Extract ALL transaction rows into a JSON object.
        Focus specifically on capturing "Description (摘要)", "Branch Name (取扱店)", and "Bank Name".
        
        [Output Schema]
        Return a JSON object with a key "table_data" containing a 2D array.
        Example:
        {
          "table_data": [
            ["Year-Month-Day", "Description(摘要)", "Amount(支払)", "Amount(入金)", "Balance(差引残高)", "Branch(取扱店)"],
            ["2026-01-22", "振込 ヤマダタロウ", "10000", "", "50000", "本店"]
          ]
        }

        [Strict Rules]
        1. **Missing Text**: The Japanese text might be faint. Look closely for Kanji/Kana in the "Description" and "Branch" columns.
        2. **Empty Cells**: Use empty string "" for blank cells. Never use null.
        3. **Formatting**: 
           - Remove spaces between Japanese chars ("海 銀" -> "海銀").
           - Keep numbers as strings (e.g. "10,000").
        """

        retry_models = [
            self.model_name,            # 1. gemini-2.5-flash
            'gemini-2.5-pro',           # 2. gemini-2.5-pro
            'gemini-2.0-flash'          # 3. gemini-2.0-flash
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
                
                data_list = []
                
                try:
                    cleaned_text = raw_text.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:-3]
                    elif cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text[3:-3]

                    parsed_json = json.loads(cleaned_text)
                    
                    if isinstance(parsed_json, dict) and "table_data" in parsed_json:
                        data_list = parsed_json["table_data"]
                    elif isinstance(parsed_json, list):
                        data_list = parsed_json
                    else:
                        raise ValueError("Unexpected JSON structure")

                except (json.JSONDecodeError, ValueError) as json_err:
                    print(f"⚠️ JSON Broken on {page_label}. Recovery...")
                    rows = re.findall(r'\[(.*?)\]', raw_text)
                    if rows:
                        for r in rows:
                            try:
                                if '"' in r or "'" in r:
                                    row_data = json.loads(f"[{r}]")
                                    data_list.append(row_data)
                            except:
                                pass
                    if not data_list:
                        data_list = [[line] for line in raw_text.split('\n') if line.strip()]
                
                formatted_rows = []
                for row in data_list:
                    def clean_text(val):
                        if val is None: return ""
                        s = str(val).strip()
                        if s.lower() in ["null", "none"]: return ""
                        return s

                    if isinstance(row, list):
                        formatted_cells = [{'text': clean_text(cell)} for cell in row]
                    else:
                        formatted_cells = [{'text': clean_text(row)}]
                    formatted_rows.append(formatted_cells)
                
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
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Auto-Enhance Mode...")
        
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
                # DPIは300のままでOK（高精細化は副作用が少ないため）
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