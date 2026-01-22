import os
import json
import io
import time
import re
import concurrent.futures
from pdf2image import convert_from_bytes
import google.generativeai as genai
from PIL import Image
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
            
            # ★変更点1: JSONモードを強制する設定を追加
            # response_mime_type="application/json" を指定すると、
            # AIは構文エラーのない完璧なJSONを返すよう強制されます。
            self.generation_config = genai.types.GenerationConfig(
                temperature=0.0, 
                top_p=1.0,
                max_output_tokens=8192,
                response_mime_type="application/json"  # <--- これが決定打です
            )
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
            print(f"⚙️ Initial Model config: {self.model_name} (JSON Mode ON)")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    def _optimize_image(self, img):
        # 精度維持のため2560px
        max_size = 2560 
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img

    def _process_single_page(self, args):
        page_label, pil_image = args
        optimized_image = self._optimize_image(pil_image)
        
        # WebP変換
        img_byte_arr = io.BytesIO()
        optimized_image.save(img_byte_arr, format='WEBP', quality=85)
        img_bytes = img_byte_arr.getvalue()
        
        image_part = {"mime_type": "image/webp", "data": img_bytes}

        # ★変更点2: JSONモード用のプロンプト
        # 出力形式を配列の配列ではなく、オブジェクト {"data": [[...]]} に指定した方が
        # GeminiのJSONモードは安定します。
        prompt = """
        You are a high-precision OCR engine specialized in Japanese business documents.
        
        [Task]
        Extract the table data from the image into a JSON object.
        
        [Output Schema]
        Return a JSON object with a single key "table_data" containing a 2D array (list of lists) of strings.
        Example:
        {
          "table_data": [
            ["Header1", "Header2"],
            ["Row1Col1", "Row1Col2"]
          ]
        }

        [Strict Rules]
        1. **Accuracy**: Transcribe exactly as written. No guessing.
        2. **Empty Cells**: Use empty string "" for blank cells. Do NOT use null.
        3. **Japanese**: Remove spaces between Japanese characters (e.g., "東 京" -> "東京").
        4. **Structure**: Ensure every row has the same number of columns.
        """

        retry_models = [
            self.model_name,            
            'gemini-2.5-pro',
            'gemini-2.0-flash'
        ]
        
        retry_models = list(dict.fromkeys(retry_models))

        for current_model_name in retry_models:
            try:
                # 設定適用
                current_model = genai.GenerativeModel(
                    current_model_name,
                    generation_config=self.generation_config
                )
                
                response = current_model.generate_content([prompt, image_part])
                raw_text = response.text
                
                data_list = []
                
                # --- JSON解析（JSONモードなので非常にシンプルになります） ---
                try:
                    # JSONモードならMarkdown記法(```json)はつかないはずだが、念のためクリーニング
                    cleaned_text = raw_text.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:-3]
                    elif cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text[3:-3]

                    parsed_json = json.loads(cleaned_text)
                    
                    # プロンプトで指定したキー "table_data" を取り出す
                    if isinstance(parsed_json, dict) and "table_data" in parsed_json:
                        data_list = parsed_json["table_data"]
                    elif isinstance(parsed_json, list):
                        # 万が一配列で返ってきた場合
                        data_list = parsed_json
                    else:
                        # 想定外の構造
                        raise ValueError("Unexpected JSON structure")

                except (json.JSONDecodeError, ValueError) as json_err:
                    # それでも壊れた場合のバックアップ（リカバリー）
                    print(f"⚠️ JSON Broken on {page_label} ({json_err}). Attempting regex recovery...")
                    rows = re.findall(r'\[(.*?)\]', raw_text)
                    if rows:
                        for r in rows:
                            try:
                                # 中身が "val", "val" のようになっているかチェック
                                if '"' in r or "'" in r:
                                    row_data = json.loads(f"[{r}]")
                                    data_list.append(row_data)
                            except:
                                pass
                    if not data_list:
                        data_list = [[line] for line in raw_text.split('\n') if line.strip()]
                
                # アプリ形式に変換
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
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Native JSON Mode...")
        
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
                pil_images = convert_from_bytes(file_bytes, dpi=200, fmt='jpeg')
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