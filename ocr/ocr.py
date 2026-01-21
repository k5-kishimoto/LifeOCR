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
            
            # デフォルトを最新の 'gemini-2.5-flash' に設定
            self.model_name = os.environ.get("GEMINI_VERSION", "gemini-2.5-flash")
            
            self.model = genai.GenerativeModel(self.model_name)
            print(f"⚙️ Initial Model config: {self.model_name}")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    def _optimize_image(self, img):
        """
        画像をAIに送りやすいサイズに軽量化
        """
        max_size = 1800 
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img

    def _process_single_page(self, args):
        """
        1ページ分を処理する関数
        """
        page_label, pil_image = args
        
        # 画像の軽量化
        optimized_image = self._optimize_image(pil_image)

        # ★ここが変更点: 縦書き対応プロンプト
        prompt = """
        Analyze the document image and extract text/data into a JSON 2D array.
        
        [Rules]
        1. Detect the text direction automatically:
           - If Horizontal (Yokogaki): Read Top-to-Bottom, Left-to-Right.
           - If Vertical (Tategaki): Read Right-to-Left columns, Top-to-Bottom characters.
        2. Preserve the table structure if present.
        3. Output ONLY a valid JSON 2D array (list of lists).
           Example: [["Title", "Author"], ["吾輩は猫である", "夏目漱石"]]
        4. Do NOT use markdown code blocks. Return raw JSON only.
        """

        # 最強の布陣
        retry_models = [
            self.model_name,            # 1. gemini-2.5-flash
            'gemini-2.5-flash-lite',    # 2. 軽量版
            'gemini-2.0-flash',         # 3. 安定版
            'gemini-3-flash-preview'    # 4. 次世代
        ]
        
        retry_models = list(dict.fromkeys(retry_models))

        for current_model_name in retry_models:
            try:
                # モデル設定
                current_model = genai.GenerativeModel(current_model_name)
                
                # リクエスト送信
                response = current_model.generate_content([prompt, optimized_image])
                raw_text = response.text
                
                # JSON抽出
                json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
                if json_match:
                    clean_json = json_match.group(0)
                    data_list = json.loads(clean_json)
                else:
                    data_list = [[line] for line in raw_text.split('\n') if line.strip()]

                # アプリ形式に変換
                formatted_rows = []
                for row in data_list:
                    # Noneや "null" 文字列を空文字にするヘルパー関数
                    def clean_text(val):
                        if val is None:
                            return ""
                        s = str(val)
                        # AIが文字として "null" や "None" を返してきた場合も消す
                        if s.lower() in ["null", "none"]:
                            return ""
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
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Vertical Support Mode...")
        
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