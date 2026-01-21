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
            
            # ★デフォルトを最新の 'gemini-2.5-flash' に設定
            # これが現在利用可能な中で最もバランスが良いモデルです
            self.model_name = os.environ.get("GEMINI_VERSION", "gemini-2.5-flash")
            
            self.model = genai.GenerativeModel(self.model_name)
            print(f"⚙️ Initial Model config: {self.model_name}")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    def _optimize_image(self, img):
        """
        画像をAIに送りやすいサイズに軽量化（高速化の鍵）
        """
        max_size = 1800 # 長辺1800pxあればOCRには十分
        
        if max(img.size) > max_size:
            # 高速なリサイズアルゴリズムを使用
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return img

    def _process_single_page(self, args):
        """
        1ページ分を処理する関数（並列処理用）
        """
        page_label, pil_image = args
        
        # 画像の軽量化
        optimized_image = self._optimize_image(pil_image)

        prompt = """
        Extract data from the table in the image.
        Output ONLY a JSON 2D array (list of lists).
        Example: [["Header1", "Header2"], ["Value1", "Value2"]]
        Do NOT use markdown. Just JSON.
        If no table, return list containing rows of text.
        """

        # ★最強の布陣：利用可能なモデルからベストな順序を選定
        retry_models = [
            self.model_name,            # 1. gemini-2.5-flash (本命)
            'gemini-2.5-flash-lite',    # 2. gemini-2.5-flash-lite (爆速バックアップ)
            'gemini-2.0-flash',         # 3. gemini-2.0-flash (安定版)
            'gemini-3-flash-preview'    # 4. gemini-3 (次世代プレビュー)
        ]
        
        # 重複削除（念のため）
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
                    if isinstance(row, list):
                        formatted_cells = [{'text': str(cell)} for cell in row]
                    else:
                        formatted_cells = [{'text': str(row)}]
                    formatted_rows.append(formatted_cells)
                
                print(f"✅ Success ({page_label}) with {current_model_name}")
                return (page_label, formatted_rows)

            except Exception as e:
                error_msg = str(e)
                print(f"⚠️ Failed ({page_label}) with {current_model_name}: {error_msg}")
                
                # 致命的なエラー以外は次のモデルへ
                if "404" in error_msg or "not found" in error_msg or "429" in error_msg or "500" in error_msg:
                    continue
                else:
                    return (page_label, [[{'text': f"Error: {error_msg}"}]])

        return (page_label, [[{'text': "Failed to extract text."}]])


    def extract_text(self, uploaded_file):
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Multi-Thread Mode...")
        
        if not self.model:
            return [[{'text': "Error: AI Model not initialized."}]]

        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        try:
            filename = uploaded_file.name.lower()
        except AttributeError:
            filename = "unknown.jpg"
            
        images_to_process = [] 

        # --- 画像変換 ---
        if filename.endswith('.pdf'):
            try:
                # 速度優先設定 (dpi=150)
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

        # ★並列処理 (Multi-Threading)
        # 画像枚数が多い場合でも一気に処理します
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_page = {executor.submit(self._process_single_page, item): item[0] for item in images_to_process}
            
            results_dict = {}
            for future in concurrent.futures.as_completed(future_to_page):
                page_label, page_data = future.result()
                results_dict[page_label] = page_data

        # ページ順に結合
        for label, _ in images_to_process:
            if len(images_to_process) > 1:
                final_results.append([{'text': f'--- {label} ---', 'is_header': True}])
            
            if label in results_dict:
                final_results.extend(results_dict[label])

        return final_results

engine = OcrEngine()