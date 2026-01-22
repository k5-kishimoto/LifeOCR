import os
import json
import io
import time
import re
import ast
import concurrent.futures
from pdf2image import convert_from_bytes
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image, ImageEnhance, ImageOps 
from dotenv import load_dotenv

load_dotenv()

class OcrEngine:
    def __init__(self):
        """初期化"""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        if not self.api_key:
            print("❌ Error: 'GEMINI_API_KEY' not found.")
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model_name = os.environ.get("GEMINI_VERSION", "gemini-2.5-flash")
            
            self.generation_config = genai.types.GenerationConfig(
                temperature=0.3, 
                top_p=0.95,
                max_output_tokens=8192,
                response_mime_type="application/json"
            )

            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            print(f"⚙️ Initial Model config: {self.model_name} (Natural-Order Mode)")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    # =========================================================================
    # 🖼️ 画像処理
    # =========================================================================

    def _optimize_image(self, img):
        max_size = 2560 
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = ImageOps.autocontrast(img, cutoff=2)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5) 
        return img

    def _split_image(self, img):
        width, height = img.size
        split_ratio = 0.60
        overlap = 0.40
        crop_top = img.crop((0, 0, width, int(height * split_ratio)))
        crop_bottom = img.crop((0, int(height * overlap), width, height))
        return [("Top", crop_top), ("Bottom", crop_bottom)]

    # =========================================================================
    # 🧠 データ解析
    # =========================================================================

    def _repair_json(self, text):
        if not text: return None
        try:
            cleaned = text.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:-3]
            elif cleaned.startswith("```"): cleaned = cleaned[3:-3]
            return json.loads(cleaned)
        except: pass

        try:
            if cleaned.count('"') % 2 != 0: cleaned += '"'
            if not cleaned.endswith("}"): cleaned += "}]}"
            return json.loads(cleaned)
        except: pass
            
        try:
            candidate_rows = re.findall(r'\[(.*?)\]', text, re.DOTALL)
            valid_rows = []
            for row_content in candidate_rows:
                if not row_content.strip(): continue
                try:
                    row_data = json.loads(f"[{row_content}]")
                    if isinstance(row_data, list): 
                        valid_rows.append(row_data)
                        continue
                except: pass
                try:
                    row_data = ast.literal_eval(f"[{row_content}]")
                    if isinstance(row_data, list): 
                        valid_rows.append(row_data)
                        continue
                except: pass
                try:
                    items = re.findall(r'"([^"]*)"', row_content)
                    if items: valid_rows.append(items)
                except: pass

            if valid_rows:
                return {"table_rows": valid_rows}
        except: pass
        return None

    def _call_ai_api(self, image_part, part_label):
        prompt = """
        あなたは日本語OCRエンジンです。画像からテキストを抽出してください。
        
        【重要命令】
        - 迷ったら推測して埋めること。空欄禁止。
        - 半角カナは半角のまま出力。
        - すべての値をダブルクォートで囲む。
        - **行の順番を変えないでください。上から順に出力してください。**

        【出力フォーマット (JSON)】
        {
          "document_info": { "title": "タイトル", "org_name": "発行元", "sub_name": "支店", "account_name": "名義", "period": "期間", "other_info": "その他" },
          "table_headers": ["項目1", "項目2", ...],
          "table_rows": [ 
             ["2026-01-22", "ﾌﾘｺﾐ ﾃｽﾄ", "10,000", "", "50,000", "本店"],
          ]
        }
        """

        retry_models = [self.model_name, 'gemini-2.5-pro', 'gemini-2.0-flash']
        
        for current_model_name in retry_models:
            try:
                current_model = genai.GenerativeModel(
                    current_model_name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                response = current_model.generate_content([prompt, image_part])
                try:
                    if not response.candidates: raise ValueError("No candidates")
                    return response.text
                except ValueError as ve:
                    if response.candidates and response.candidates[0].content.parts:
                        return response.candidates[0].content.parts[0].text
                    raise ve
            except Exception as e:
                print(f"⚠️ API Error ({part_label} - {current_model_name}): {e}")
                time.sleep(1)
                continue
        return None

    # =========================================================================
    # 🔄 スマート結合・整形（順序維持バージョン）
    # =========================================================================

    def _merge_split_results(self, results):
        combined_json = { "document_info": {}, "table_headers": [], "table_rows": [] }

        # Top情報を優先
        target_source = "Top" if "Top" in results else "Bottom"
        if target_source in results:
            combined_json["document_info"] = results[target_source].get("document_info", {})
            combined_json["table_headers"] = results[target_source].get("table_headers", [])

        # --- 順序維持のマージロジック ---
        # 1. まず「Top」の結果をそのまま採用（これが文書の上半分なので順序は正しい）
        final_rows = []
        
        top_rows = results.get("Top", {}).get("table_rows", [])
        bottom_rows = results.get("Bottom", {}).get("table_rows", [])
        
        # Topの行を追加（クリーニングしつつ）
        for row in top_rows:
            if not row or all(str(c).strip() == "" for c in row): continue
            
            cleaned_row = []
            for c in row:
                val = str(c) if isinstance(c, (dict, list)) else str(c).strip()
                val = val.replace("■", " ") # ノイズ除去
                cleaned_row.append(val)
            final_rows.append(cleaned_row)

        # 2. 「Bottom」の行をチェックして、新しい行なら末尾に追加する
        # （TopとBottomの重複部分は、Topを正として、Bottom側の情報で補完する）
        
        for b_row in bottom_rows:
            if not b_row or all(str(c).strip() == "" for c in b_row): continue

            b_cleaned = []
            for c in b_row:
                val = str(c) if isinstance(c, (dict, list)) else str(c).strip()
                val = val.replace("■", " ")
                b_cleaned.append(val)
            
            # このBottom行が、すでにTop行（final_rows）に含まれているかチェック
            match_index = -1
            
            for i, t_row in enumerate(final_rows):
                # 列数が違うなら別の行
                if len(t_row) != len(b_cleaned): continue
                
                # 内容の一致度をチェック
                # 「同じ日付」かつ「同じ金額」なら同一行とみなす、などの判定
                match_count = 0
                non_empty_count = 0
                
                for v1, v2 in zip(t_row, b_cleaned):
                    if v1 or v2: non_empty_count += 1
                    if v1 and v2 and v1 == v2: match_count += 1
                
                # 8割以上一致していれば「同じ行（重複）」とみなす
                if non_empty_count > 0 and (match_count / non_empty_count) > 0.8:
                    match_index = i
                    break
            
            if match_index != -1:
                # 重複が見つかった場合：
                # Bottomの方が情報量が多い（文字数が多い）場合のみ、既存行をアップデート（補完）する
                # ※順序は変えない！
                existing = final_rows[match_index]
                merged_row = []
                for t_val, b_val in zip(existing, b_cleaned):
                    # シンプルに長い方を採用（情報の欠損を防ぐため）
                    if len(b_val) > len(t_val):
                        merged_row.append(b_val)
                    else:
                        merged_row.append(t_val)
                final_rows[match_index] = merged_row
            else:
                # 重複が見つからない場合：
                # これはBottom部分にしかない新しい行なので、末尾に追加
                final_rows.append(b_cleaned)

        # ★重要: ここで sort をしない！
        # final_rows.sort(...) <--- これを削除しました

        combined_json["table_rows"] = final_rows
        return combined_json, len(final_rows)

    def _format_to_ui_data(self, combined_json):
        formatted_rows = []
        def safe_str(val):
            if val is None: return ""
            if isinstance(val, (dict, list)): return str(val)
            return str(val).strip()

        # 文書情報
        doc_info = combined_json.get("document_info", {})
        title_text = safe_str(doc_info.get('title')) or ""
        if title_text: formatted_rows.append([{'text': f"■ {title_text}"}])
        
        org_info = []
        if doc_info.get("org_name"): org_info.append(safe_str(doc_info['org_name']))
        if doc_info.get("sub_name"): org_info.append(safe_str(doc_info['sub_name']))
        if doc_info.get("bank_name"): org_info.append(safe_str(doc_info['bank_name']))
        if doc_info.get("branch_name"): org_info.append(safe_str(doc_info['branch_name']))
        if org_info: formatted_rows.append([{'text': " ".join(org_info)}])

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
            formatted_rows.append([{'text': h} for h in clean_headers])

        # 明細データ
        for row in combined_json.get("table_rows", []):
            formatted_cells = [{'text': safe_str(cell)} for cell in row]
            formatted_rows.append(formatted_cells)

        return formatted_rows

    # =========================================================================
    # 🚀 メイン処理
    # =========================================================================

    def _process_single_page(self, args):
        page_label, pil_image = args
        optimized_image = self._optimize_image(pil_image)
        parts = self._split_image(optimized_image)
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_part = {}
            for p_name, p_img in parts:
                img_byte_arr = io.BytesIO()
                p_img.save(img_byte_arr, format='WEBP', quality=100)
                image_part = {"mime_type": "image/webp", "data": img_byte_arr.getvalue()}
                
                future = executor.submit(self._call_ai_api, image_part, f"{page_label}-{p_name}")
                future_to_part[future] = p_name

            for future in concurrent.futures.as_completed(future_to_part):
                p_name = future_to_part[future]
                res_text = future.result()
                if res_text:
                    repaired_data = self._repair_json(res_text)
                    if repaired_data:
                        results[p_name] = repaired_data
                    else:
                        print(f"❌ JSON Repair Failed for {p_name}")

        combined_json, row_count = self._merge_split_results(results)
        formatted_rows = self._format_to_ui_data(combined_json)
        
        print(f"✅ Success ({page_label}) - Merged {row_count} rows")
        return (page_label, formatted_rows)


    def extract_text(self, uploaded_file):
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Natural-Order Mode...")
        if not self.model: return [[{'text': "Error: AI Model not initialized."}]]

        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        try: filename = uploaded_file.name.lower()
        except: filename = "unknown.jpg"
            
        images_to_process = [] 
        if filename.endswith('.pdf'):
            try:
                pil_images = convert_from_bytes(file_bytes, dpi=250, fmt='jpeg')
                for i, img in enumerate(pil_images):
                    images_to_process.append((f"Page {i+1}", img))
            except Exception as e: return [[{'text': f"PDF Error: {e}"}]]
        else:
            img = Image.open(io.BytesIO(file_bytes))
            images_to_process.append(("Page 1", img))

        final_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_page = {executor.submit(self._process_single_page, item): item[0] for item in images_to_process}
            results_dict = {}
            for future in concurrent.futures.as_completed(future_to_page):
                page_label, page_data = future.result()
                results_dict[page_label] = page_data

        for label, _ in images_to_process:
            if len(images_to_process) > 1:
                final_results.append([{'text': f'--- {label} ---'}])
            if label in results_dict:
                final_results.extend(results_dict[label])
        return final_results

engine = OcrEngine()