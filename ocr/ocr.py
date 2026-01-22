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
            
            # ★修正点: 温度(Temperature)を上げて「推測」を許容する
            self.generation_config = genai.types.GenerationConfig(
                temperature=0.3,   # 0.0(厳格) -> 0.3(推測許可)
                top_p=0.95,        # 突飛な幻覚は防ぐ
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
            print(f"⚙️ Initial Model config: {self.model_name} (Creative-Read Mode)")

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
        
        # コントラストをさらに強める
        img = ImageOps.autocontrast(img, cutoff=2) # cutoffを少し上げてノイズを飛ばす
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5) # シャープネスも強め
        
        return img

    def _split_image(self, img):
        width, height = img.size
        split_ratio = 0.60
        overlap = 0.40
        crop_top = img.crop((0, 0, width, int(height * split_ratio)))
        crop_bottom = img.crop((0, int(height * overlap), width, height))
        return [("Top", crop_top), ("Bottom", crop_bottom)]

    # =========================================================================
    # 🧠 データ解析・修復
    # =========================================================================

    def _repair_json(self, text):
        if not text: return None
        
        try:
            cleaned = text.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:-3]
            elif cleaned.startswith("```"): cleaned = cleaned[3:-3]
            return json.loads(cleaned)
        except:
            pass

        try:
            if cleaned.count('"') % 2 != 0: cleaned += '"'
            if not cleaned.endswith("}"): cleaned += "}]}"
            return json.loads(cleaned)
        except:
            pass
            
        try:
            # どんな形式でも拾うロジック
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
        except:
            pass

        return None

    def _call_ai_api(self, image_part, part_label):
        
        # ★修正: プロンプトでも「推測」を許可する
        prompt = """
        あなたは日本語OCRエンジンです。
        画像からテキストを抽出してください。
        
        【重要命令: 積極的な読み取り】
        - **迷ったら推測して書いてください。**
        - 文字が薄くても、ノイズがあっても、そこに行があるなら空欄にせず、一番近い文字を推測して埋めてください。
        - 「読めないから無視する」は禁止です。

        【抽出ルール】
        1. **項目名**: ヘッダー内の改行はつなげる（例:「お預り\n金額」→「お預り金額」）。
        2. **文字種**: 半角カナ(`ﾌﾘｺﾐ`)は半角のまま。
        3. **データ型**: すべての値をダブルクォートで囲む。
        
        【出力フォーマット (JSON)】
        {
          "document_info": { "title": "タイトル", "org_name": "発行元", "sub_name": "支店", "account_name": "名義", "period": "期間", "other_info": "その他" },
          "table_headers": ["項目1", "項目2", ...],
          "table_rows": [ 
             ["2026-01-22", "ﾌﾘｺﾐ ﾃｽﾄ", "10,000", "", "50,000", "本店"],
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
                # モデルごとに設定を適用
                current_model = genai.GenerativeModel(
                    current_model_name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                response = current_model.generate_content([prompt, image_part])
                
                try:
                    if not response.candidates:
                        raise ValueError("No candidates")
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason != 1: 
                         print(f"⚠️ Warning ({part_label}): Finish reason is {finish_reason}")
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
    # 🔄 結合・整形
    # =========================================================================

    def _merge_split_results(self, results):
        combined_json = { "document_info": {}, "table_headers": [], "table_rows": [] }

        target_source = "Top" if "Top" in results else "Bottom"
        if target_source in results:
            combined_json["document_info"] = results[target_source].get("document_info", {})
            combined_json["table_headers"] = results[target_source].get("table_headers", [])

        raw_rows = []
        if "Top" in results: raw_rows.extend(results["Top"].get("table_rows", []))
        if "Bottom" in results: raw_rows.extend(results["Bottom"].get("table_rows", []))

        seen = set()
        unique_rows = []
        for row in raw_rows:
            if not row or all(str(c).strip() == "" for c in row): continue
            
            row_vals = []
            for c in row:
                if isinstance(c, (dict, list)): row_vals.append(str(c))
                else: row_vals.append(str(c).strip())
            
            row_id = "".join(row_vals)
            if row_id and row_id not in seen:
                seen.add(row_id)
                unique_rows.append(row)
        
        combined_json["table_rows"] = unique_rows
        return combined_json, len(unique_rows)

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
            def clean_cell(val):
                if val is None: return ""
                if isinstance(val, (dict, list)): return str(val)
                s = str(val).strip()
                if s.lower() in ["null", "none"]: return ""
                return s

            if isinstance(row, list):
                formatted_cells = [{'text': clean_cell(cell)} for cell in row]
            else:
                formatted_cells = [{'text': clean_cell(row)}]
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
                # 最高画質を維持
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
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Creative-Read Mode...")
        
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