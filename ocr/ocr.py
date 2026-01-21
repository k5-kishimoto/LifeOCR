import os
import json
import io
import time
import re
from pdf2image import convert_from_bytes
import google.generativeai as genai
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
            
            # ★修正点: ユーザー環境で確実に動く 'gemini-2.0-flash' をデフォルトに設定
            self.model_name = os.environ.get("GEMINI_VERSION", "gemini-2.0-flash")
            
            self.model = genai.GenerativeModel(self.model_name)
            print(f"⚙️ Initial Model config: {self.model_name}")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    def extract_text(self, uploaded_file):
        print(f"⏳ Starting Gemini AI OCR ({self.model_name})...")
        
        if not self.model:
            return [[{'text': "Error: AI Model not initialized."}]]

        # ファイル読み込み
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        try:
            filename = uploaded_file.name.lower()
        except AttributeError:
            filename = "unknown.jpg"
            
        final_results = []
        images_to_process = [] 

        # --- 画像変換 ---
        if filename.endswith('.pdf'):
            try:
                pil_images = convert_from_bytes(file_bytes, dpi=200, fmt='jpeg')
                for i, img in enumerate(pil_images):
                    images_to_process.append((f"Page {i+1}", img))
            except Exception as e:
                print(f"❌ PDF Error: {e}")
                return [[{'text': f"PDF Error: {e}"}]]
        else:
            from PIL import Image
            img = Image.open(io.BytesIO(file_bytes))
            images_to_process.append(("Page 1", img))

        # --- AI解析実行 ---
        for page_label, pil_image in images_to_process:
            if len(images_to_process) > 1 or len(final_results) > 0:
                final_results.append([{'text': f'--- {page_label} ---', 'is_header': True}])

            prompt = """
            Extract data from the table in the image.
            Output ONLY a JSON 2D array (list of lists).
            Example: [["Header1", "Header2"], ["Value1", "Value2"]]
            Do NOT use markdown. Just JSON.
            If no table, return list containing rows of text.
            """

            # ★確定した「存在するモデル」のみをリスト化
            # 1. gemini-2.0-flash (本命)
            # 2. gemini-flash-latest (予備: 常に最新のFlashを指すエイリアス)
            retry_models = [
                self.model_name,       # gemini-2.0-flash
                'gemini-flash-latest'  # Backup
            ]
            
            # 重複を除去（環境変数で同じものを指定した場合など）
            retry_models = list(dict.fromkeys(retry_models))
            
            success = False
            
            for current_model_name in retry_models:
                try:
                    # モデルをセット
                    current_model = genai.GenerativeModel(current_model_name)
                    
                    # リクエスト送信
                    response = current_model.generate_content([prompt, pil_image])
                    raw_text = response.text
                    
                    # 成功したらJSON解析へ
                    json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
                    if json_match:
                        clean_json = json_match.group(0)
                        data_list = json.loads(clean_json)
                    else:
                        data_list = [[line] for line in raw_text.split('\n') if line.strip()]

                    # アプリ形式に変換
                    formatted_rows = []
                    for row in data_list:
                        if isinstance(row, list):
                            formatted_cells = [{'text': str(cell)} for cell in row]
                        else:
                            formatted_cells = [{'text': str(row)}]
                        formatted_rows.append(formatted_cells)
                    
                    final_results.extend(formatted_rows)
                    success = True
                    # 成功したらループを抜ける
                    print(f"✅ Success with model: {current_model_name}")
                    break 

                except Exception as e:
                    error_msg = str(e)
                    print(f"⚠️ Failed with {current_model_name}: {error_msg}")
                    
                    # エラーなら即座に次のモデルへ
                    if "404" in error_msg or "not found" in error_msg or "429" in error_msg:
                        print("🔄 Switching to backup model...")
                        continue
                    else:
                        final_results.append([{'text': f"Error: {error_msg}"}])
                        success = True
                        break

            if not success:
                final_results.append([{'text': "Failed to extract text with available models."}])

            # 課金済み・最新モデルなら高速なので待機時間は短めでOK
            time.sleep(0.5)

        return final_results

engine = OcrEngine()