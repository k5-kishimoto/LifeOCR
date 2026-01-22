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
            # ★基本は最新の 2.5-flash を使用（精度と速度のバランスが最強）
            self.model_name = os.environ.get("GEMINI_VERSION", "gemini-2.5-flash")
            
            # ★設定: 創造性はゼロにする（事実のみを抽出させるため精度アップ＆迷いなし）
            self.generation_config = genai.types.GenerationConfig(
                temperature=0.0, 
                top_p=1.0,
                max_output_tokens=8192,
            )
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
            print(f"⚙️ Initial Model config: {self.model_name} (Temp=0.0 / High-Res Mode)")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    def _optimize_image(self, img):
        """
        ★精度のための画像最適化
        """
        # 1. サイズ調整: 1800pxだと小さい文字が潰れることがあるため、2560pxまで許容
        max_size = 2560 
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # 2. カラー情報は保持する（グレースケール化は廃止）
        # 赤字やマーカー、表の色分けなどの重要情報を捨てないようにします。
        
        return img

    def _process_single_page(self, args):
        page_label, pil_image = args
        
        # 最適化（高解像度キープ）
        optimized_image = self._optimize_image(pil_image)
        
        # ★通信速度対策: WebPの高画質設定(Quality=85)を使う
        # JPEGより軽いが、画質劣化は人間の目では分からないレベル
        img_byte_arr = io.BytesIO()
        optimized_image.save(img_byte_arr, format='WEBP', quality=85)
        img_bytes = img_byte_arr.getvalue()
        
        image_part = {"mime_type": "image/webp", "data": img_bytes}

        # ★プロンプト: 精度重視の「厳格モード」を採用
        prompt = """
        You are a high-precision OCR engine specialized in Japanese business documents.
        Your task is to transcribe the text in the image into a JSON 2D array.

        [Strict Rules]
        1. **Text Direction**: Automatically detect vertical (Tategaki) or horizontal (Yokogaki).
           - Vertical: Read columns right-to-left.
           - Horizontal: Read rows top-to-bottom.
        2. **Structure**: Maintain the exact visual table structure.
           - Output a list of lists: `[["Header1", "Header2"], ["Row1Col1", "Row1Col2"]]`.
        3. **Accuracy & Cleaning**:
           - Transcribe exactly as written. Do not guess.
           - **Empty Cells**: If a cell is visually empty, return an empty string "". Do NOT return "null", "None", or "-".
           - **Japanese Spacing**: Remove unnecessary whitespace between Japanese characters (e.g., convert "東 京" to "東京"). Keep spaces in English sentences.
        4. **Output Format**: 
           - Return RAW JSON only. 
           - NO markdown code blocks (```json). 
           - NO explanations.
        """

        # ★モデル優先順位の更新（提供リストに基づく）
        retry_models = [
            self.model_name,            # 1. gemini-2.5-flash (本命・バランス型)
            'gemini-2.5-pro',           # 2. gemini-2.5-pro (超高精度バックアップ)
            'gemini-2.0-flash'          # 3. gemini-2.0-flash (安定版バックアップ)
        ]
        
        retry_models = list(dict.fromkeys(retry_models))

        for current_model_name in retry_models:
            try:
                # 設定適用
                current_model = genai.GenerativeModel(
                    current_model_name,
                    generation_config=self.generation_config
                )
                
                # リクエスト送信
                response = current_model.generate_content([prompt, image_part])
                raw_text = response.text
                
                data_list = []
                
                # --- JSONリカバリーロジック ---
                try:
                    json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
                    if json_match:
                        clean_json = json_match.group(0)
                        data_list = json.loads(clean_json)
                    else:
                        raise ValueError("No JSON found")

                except (json.JSONDecodeError, ValueError):
                    print(f"⚠️ JSON Broken on {page_label}. Attempting recovery...")
                    rows = re.findall(r'\[(.*?)\]', raw_text)
                    if rows:
                        for r in rows:
                            try:
                                row_data = json.loads(f"[{r}]")
                                data_list.append(row_data)
                            except:
                                data_list.append([r])
                    else:
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
                
                # 致命的でないエラーなら次のモデルへ
                if "404" in error_msg or "not found" in error_msg or "429" in error_msg or "500" in error_msg:
                    continue
                else:
                    return (page_label, [[{'text': f"Error: {error_msg}"}]])

        return (page_label, [[{'text': "Failed to extract text."}]])


    def extract_text(self, uploaded_file):
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Balanced High-Fidelity Mode...")
        
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
                # PDF変換解像度: 200dpi (小さな文字の精度確保のため)
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

        # 並列処理維持
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