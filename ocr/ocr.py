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
            
            # JSONモード
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
            print(f"⚙️ Initial Model config: {self.model_name} (Split & Merge Mode)")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    def _optimize_image(self, img):
        # 解像度設定
        max_size = 2560 
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = ImageOps.autocontrast(img, cutoff=1)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.4) 
        
        return img

    def _call_ai_api(self, image_part, part_label):
        """
        AIへのリクエスト部分だけを切り出した関数
        """
        prompt = """
        あなたは高精度の日本語OCRエンジンです。
        画像からテキスト情報を抽出し、JSONを返してください。
        
        【重要：分割処理中】
        この画像は書類の一部（上半分または下半分）の可能性があります。
        見えている範囲のすべての情報を抽出してください。ヘッダーがなくても明細があれば抽出してください。

        【重要ルール】
        1. 文字種の維持: 半角カナ(`ﾌﾘｺﾐ`)は半角のまま。全角変換禁止。
        2. 空白の維持: 氏名の間のスペース(`ﾔﾏﾀﾞ ﾀﾛｳ`)は削除しない。
        
        【出力フォーマット (JSON)】
        {
          "document_info": {
             "bank_name": "銀行名",
             "branch_name": "支店名",
             "title": "文書タイトル",
             "account_name": "口座名義",
             "period": "期間",
             "other_info": "その他"
          },
          "table_headers": ["日付", "摘要", "お支払金額", "お預り金額", "差引残高", "取扱店"],
          "table_rows": [
             ["2026-01-22", "ﾌﾘｺﾐ ﾔﾏﾀﾞ ﾀﾛｳ", "10,000", "", "50,000", "本店"]
          ]
        }
        """

        retry_models = [
            self.model_name,
            'gemini-2.5-pro',
            'gemini-2.0-flash'
        ]
        
        for current_model_name in retry_models:
            try:
                current_model = genai.GenerativeModel(
                    current_model_name,
                    generation_config=self.generation_config
                )
                
                response = current_model.generate_content([prompt, image_part])
                return response.text
                
            except Exception as e:
                print(f"⚠️ API Error ({part_label}): {e}")
                time.sleep(1)
                continue
        
        return None

    def _process_single_page(self, args):
        page_label, pil_image = args
        optimized_image = self._optimize_image(pil_image)
        
        width, height = optimized_image.size
        
        # ★★★ 画像分割ロジック (Split) ★★★
        # 上半分(0%~60%) と 下半分(40%~100%) に分割
        # 20%のオーバーラップを持たせることで、切断線上の文字欠けを防ぐ
        split_ratio = 0.60
        overlap = 0.40 # 下半分の開始位置 (40%地点からスタート)
        
        crop_top = optimized_image.crop((0, 0, width, int(height * split_ratio)))
        crop_bottom = optimized_image.crop((0, int(height * overlap), width, height))
        
        parts = [
            ("Top", crop_top),
            ("Bottom", crop_bottom)
        ]
        
        combined_json = {
            "document_info": {},
            "table_headers": [],
            "table_rows": []
        }
        
        # 分割画像を並列でAIに投げる
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_part = {}
            for p_name, p_img in parts:
                # 画像変換
                img_byte_arr = io.BytesIO()
                p_img.save(img_byte_arr, format='WEBP', quality=85)
                image_part = {"mime_type": "image/webp", "data": img_byte_arr.getvalue()}
                
                future = executor.submit(self._call_ai_api, image_part, f"{page_label}-{p_name}")
                future_to_part[future] = p_name

            # 結果の回収とマージ
            results = {}
            for future in concurrent.futures.as_completed(future_to_part):
                p_name = future_to_part[future]
                res_text = future.result()
                
                if res_text:
                    try:
                        cleaned = res_text.strip()
                        if cleaned.startswith("```json"): cleaned = cleaned[7:-3]
                        elif cleaned.startswith("```"): cleaned = cleaned[3:-3]
                        results[p_name] = json.loads(cleaned)
                    except:
                        print(f"❌ JSON Parse Failed for {p_name}")

        # ★★★ 結合ロジック (Merge) ★★★
        
        # 1. 文書情報とヘッダーは「Top」の結果を優先
        if "Top" in results:
            combined_json["document_info"] = results["Top"].get("document_info", {})
            combined_json["table_headers"] = results["Top"].get("table_headers", [])
        elif "Bottom" in results:
            combined_json["document_info"] = results["Bottom"].get("document_info", {})
            combined_json["table_headers"] = results["Bottom"].get("table_headers", [])

        # 2. 行データの結合と重複排除
        raw_rows = []
        if "Top" in results:
            raw_rows.extend(results["Top"].get("table_rows", []))
        if "Bottom" in results:
            raw_rows.extend(results["Bottom"].get("table_rows", []))
            
        # 重複排除 (リストを文字列化してセットで管理)
        seen = set()
        unique_rows = []
        for row in raw_rows:
            # 行の中身を結合してユニークIDにする
            row_id = "".join([str(c).strip() for c in row])
            if row_id and row_id not in seen:
                seen.add(row_id)
                unique_rows.append(row)
        
        combined_json["table_rows"] = unique_rows
        
        # --- アプリ形式への変換 ---
        formatted_rows = []
        
        def safe_str(val):
            if val is None: return ""
            if isinstance(val, (dict, list)): return str(val)
            return str(val).strip()

        # 文書情報
        doc_info = combined_json.get("document_info", {})
        title_text = safe_str(doc_info.get('title')) or "明細書"
        formatted_rows.append([{'text': f"■ {title_text}", 'is_header': True}])
        
        bank_info = []
        if doc_info.get("bank_name"): bank_info.append(f"🏦 {safe_str(doc_info['bank_name'])}")
        if doc_info.get("branch_name"): bank_info.append(f"🏢 {safe_str(doc_info['branch_name'])}")
        if bank_info: formatted_rows.append([{'text': " ".join(bank_info)}])

        meta_texts = []
        if doc_info.get("account_name"): meta_texts.append(f"名義: {safe_str(doc_info['account_name'])}")
        if doc_info.get("period"): meta_texts.append(f"期間: {safe_str(doc_info['period'])}")
        if doc_info.get("other_info"): meta_texts.append(safe_str(doc_info['other_info']))
        if meta_texts: formatted_rows.append([{'text': " / ".join(meta_texts)}])
        
        formatted_rows.append([{'text': ""}])

        # ヘッダー
        headers = combined_json.get("table_headers", [])
        if headers:
            clean_headers = [safe_str(h) for h in headers]
            formatted_rows.append([{'text': h, 'is_header': True} for h in clean_headers])

        # データ
        for row in unique_rows:
            def clean_text(val):
                if val is None: return ""
                if isinstance(val, (dict, list)): return str(val)
                s = str(val).strip()
                if s.lower() in ["null", "none"]: return ""
                return s

            if isinstance(row, list):
                formatted_cells = [{'text': clean_text(cell)} for cell in row]
            else:
                formatted_cells = [{'text': clean_text(row)}]
            formatted_rows.append(formatted_cells)
        
        print(f"✅ Success ({page_label}) - Merged {len(unique_rows)} rows")
        return (page_label, formatted_rows)

    def extract_text(self, uploaded_file):
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Split & Merge Mode...")
        
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
                pil_images = convert_from_bytes(file_bytes, dpi=250, fmt='jpeg')
                for i, img in enumerate(pil_images):
                    images_to_process.append((f"Page {i+1}", img))
            except Exception as e:
                print(f"❌ PDF Error: {e}")
                return [[{'text': f"PDF Error: {e}"}]]
        else:
            img = Image.open(io.BytesIO(file_bytes))
            images_to_process.append(("Page 1", img))

        final_results = []

        # ページ単位の並列処理（各ページ内でさらに2分割並列処理が走る）
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
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