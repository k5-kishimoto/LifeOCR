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
            print(f"⚙️ Initial Model config: {self.model_name} (Japanese OCR Mode)")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    def _optimize_image(self, img):
        """
        スマート画像補正（薄い文字対策）
        """
        max_size = 2560 
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # オートコントラストで薄い文字をくっきりさせる
        img = ImageOps.autocontrast(img, cutoff=1)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1) 
        
        return img

    def _process_single_page(self, args):
        page_label, pil_image = args
        
        optimized_image = self._optimize_image(pil_image)
        
        img_byte_arr = io.BytesIO()
        optimized_image.save(img_byte_arr, format='WEBP', quality=85)
        img_bytes = img_byte_arr.getvalue()
        
        image_part = {"mime_type": "image/webp", "data": img_bytes}

        # ★ここを大幅修正：日本語プロンプト ＆ メタデータ抽出指示
        prompt = """
        あなたは高精度の日本語OCRエンジンです。
        提供された画像（通帳、銀行明細、請求書など）から、可能な限りすべてのテキスト情報を抽出し、構造化データとして返してください。

        【タスク】
        1. **文書情報（表の外）**: 
           - 「文書のタイトル（例：入出金明細）」「銀行名」「支店名」「口座名義」「期間」「作成日」など、表の外にある重要情報を抽出してください。
        2. **明細データ（表の中）**: 
           - 表の中身をすべて抽出してください。

        【重要ルール】
        - **翻訳禁止**: 画像に書かれている日本語をそのまま出力してください。英語（"Date", "Description"）に変換しないでください。「摘要」は「摘要」のまま出力します。
        - **全角・半角**: 数字は半角に統一しますが、カタカナや漢字は原文のままにしてください。
        - **空欄**: 何も書かれていないセルは空文字 "" にしてください。null は禁止です。

        【出力フォーマット (JSON)】
        以下の構造でJSONのみを返してください。
        {
          "document_info": {
             "title": "文書のタイトル（見つからなければ空文字）",
             "bank_name": "銀行名（あれば）",
             "account_name": "口座名義（あれば）",
             "other_info": "その他（支店名や期間など、見つかったテキスト）"
          },
          "table_headers": ["日付", "摘要", "お支払金額", "お預り金額", "差引残高", "取扱店"],
          "table_rows": [
             ["2026-01-22", "振込 ヤマダタロウ", "10,000", "", "50,000", "本店"],
             ["2026-01-23", "電気代", "5,000", "", "45,000", ""]
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
                
                # 結果格納用
                formatted_rows = []

                try:
                    # JSONクリーニング
                    cleaned_text = raw_text.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:-3]
                    elif cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text[3:-3]

                    parsed_json = json.loads(cleaned_text)
                    
                    # --- アプリ表示用にデータを整形 ---
                    
                    # 1. 文書情報（タイトル等）を最初の数行として追加
                    doc_info = parsed_json.get("document_info", {})
                    
                    # タイトルがあれば大きく表示（ヘッダー扱い）
                    if doc_info.get("title"):
                        formatted_rows.append([{'text': f"■ {doc_info['title']}", 'is_header': True}])
                    
                    # その他のメタデータを行として追加
                    meta_texts = []
                    if doc_info.get("bank_name"): meta_texts.append(doc_info["bank_name"])
                    if doc_info.get("account_name"): meta_texts.append(f"名義: {doc_info['account_name']}")
                    if doc_info.get("other_info"): meta_texts.append(doc_info["other_info"])
                    
                    if meta_texts:
                        formatted_rows.append([{'text': " / ".join(meta_texts)}])
                        formatted_rows.append([{'text': ""}]) # 空行で見やすく

                    # 2. 表ヘッダー
                    headers = parsed_json.get("table_headers", [])
                    if headers:
                        # ヘッダーを強調表示
                        formatted_rows.append([{'text': h, 'is_header': True} for h in headers])

                    # 3. 表データ
                    rows = parsed_json.get("table_rows", [])
                    for row in rows:
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

                except (json.JSONDecodeError, ValueError) as json_err:
                    print(f"⚠️ JSON Parse Error on {page_label}: {json_err}. Fallback to raw text.")
                    # JSON解析失敗時は、とりあえず生のテキストを行ごとに表示
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
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Japanese Native Mode...")
        
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
                # 日本語の細かい文字のためにDPI 300を維持
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
                # ページ区切り線
                final_results.append([{'text': f'--- {label} ---', 'is_header': True}])
            
            if label in results_dict:
                final_results.extend(results_dict[label])

        return final_results

engine = OcrEngine()