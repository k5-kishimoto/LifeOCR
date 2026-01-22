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
            self.model = genai.GenerativeModel(self.model_name)
            print(f"⚙️ Initial Model config: {self.model_name}")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    def _optimize_image(self, img):
        max_size = 1800 
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img

    def _process_single_page(self, args):
        page_label, pil_image = args
        optimized_image = self._optimize_image(pil_image)
        # OCRプロンプトの定義
        prompt = """
        You are a high-precision OCR engine specialized in Japanese documents.
        Your task is to transcribe the text in the image into a JSON 2D array.

        [Strict Rules]
        1. **Text Direction**: Automatically detect vertical (Tategaki) or horizontal (Yokogaki).
        2. **Structure**: Maintain the exact visual table structure. Output a list of lists.
        3. **Accuracy**: Transcribe exactly as written.
           - Empty Cells: Return empty string "". Do NOT return "null".
           - Remove unnecessary whitespace between Japanese characters.
        4. **Output Format**: Return RAW JSON only. NO markdown.
        """

        retry_models = [
            self.model_name,            
            'gemini-2.5-flash-lite',    
            'gemini-2.0-flash',         
            'gemini-3-flash-preview'    
        ]
        
        retry_models = list(dict.fromkeys(retry_models))

        for current_model_name in retry_models:
            try:
                current_model = genai.GenerativeModel(current_model_name)
                
                # リクエスト送信
                response = current_model.generate_content([prompt, optimized_image])
                raw_text = response.text
                
                data_list = []
                
                # --- ★★★ JSON解析ロジックの大幅強化 ★★★ ---
                try:
                    # 1. まず標準的な抽出を試みる
                    json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
                    if json_match:
                        clean_json = json_match.group(0)
                        data_list = json.loads(clean_json)
                    else:
                        raise ValueError("No JSON found")

                except (json.JSONDecodeError, ValueError):
                    # 2. 【リカバリーモード】JSONが壊れている場合
                    print(f"⚠️ JSON Broken on {page_label}. Attempting recovery...")
                    
                    # 行単位（内側の [ ... ] ）で正規表現抽出を試みる
                    # 外枠が壊れていても、中身の行さえ取れればOK
                    rows = re.findall(r'\[(.*?)\]', raw_text)
                    
                    if rows:
                        for r in rows:
                            try:
                                # 各行を個別にJSONパースしてみる
                                # "Col1", "Col2" のような文字列をリスト化
                                # カッコを補完してパース
                                row_data = json.loads(f"[{r}]")
                                data_list.append(row_data)
                            except:
                                # パースできない行はそのまま文字列として追加
                                data_list.append([r])
                    else:
                        # 3. それでもダメなら単なる改行区切りテキストとして処理
                        data_list = [[line] for line in raw_text.split('\n') if line.strip()]
                # ----------------------------------------------------

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
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Robust JSON Mode...")
        
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
                pil_images = convert_from_bytes(file_bytes, dpi=150, fmt='jpeg')
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