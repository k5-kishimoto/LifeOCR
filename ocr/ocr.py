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
            print(f"⚙️ Initial Model config: {self.model_name} (Top-Priority Mode)")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    # =========================================================================
    # 🧹 クリーニング & 指紋生成
    # =========================================================================
    
    def _clean_text(self, val):
        if val is None: return ""
        if isinstance(val, (dict, list)): val = str(val)
        val = str(val)
        val = val.replace("\n", "").replace("\r", "")
        val = val.replace("■", " ") 
        val = re.sub(r'\s+', ' ', val)
        return val.strip()

    def _is_header_row(self, row):
        """ヘッダー行判定"""
        header_keywords = ["日付", "摘要", "金額", "入金", "出金", "残高", "借方", "貸方", "区分", "支店名"]
        match_count = 0
        for cell in row:
            text = str(cell)
            if any(k in text for k in header_keywords):
                match_count += 1
        return match_count >= 2

    def _get_row_fingerprint(self, row):
        """
        行の同一性を判定するための指紋を作成
        「日付」と「金額（数値）」のセットをキーにする
        """
        parts = []
        for cell in row:
            text = self._clean_text(cell)
            # 日付
            if re.search(r'\d{4}[/-年]\d{1,2}[/-月]\d{1,2}', text) or re.search(r'\d{1,2}[/-]\d{1,2}', text):
                parts.append(text)
            # 金額 (3桁区切り または 3桁以上の数値)
            elif re.search(r'\d{1,3}(,\d{3})+', text) or (text.isdigit() and len(text) > 2):
                parts.append(text.replace(",", ""))
        
        return "_".join(sorted(parts))

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
                    if isinstance(row_data, list): valid_rows.append(row_data)
                    continue
                except: pass
                try:
                    row_data = ast.literal_eval(f"[{row_content}]")
                    if isinstance(row_data, list): valid_rows.append(row_data)
                    continue
                except: pass
                try:
                    items = re.findall(r'"([^"]*)"', row_content)
                    if items: valid_rows.append(items)
                except: pass
            if valid_rows: return {"table_rows": valid_rows}
        except: pass
        return None

    def _call_ai_api(self, image_part, part_label):
        prompt = """
        あなたは日本語OCRエンジンです。画像からテキストを抽出してください。
        
        【重要命令】
        - **改行コード禁止。1行につなげる。**
        - **あるがまま抽出すること。**
        - 空欄禁止。推測して埋める。
        - 半角カナは半角のまま。
        - 途中にあるヘッダー行は無視してデータ行だけ抽出。

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
    # 🔄 Top優先マージ（重複時はBottomを捨てる）
    # =========================================================================

    def _merge_split_results(self, results):
        combined_json = { "document_info": {}, "table_headers": [], "table_rows": [] }

        # Top情報を優先
        target_source = "Top" if "Top" in results else "Bottom"
        if target_source in results:
            combined_json["document_info"] = results[target_source].get("document_info", {})
            combined_json["table_headers"] = results[target_source].get("table_headers", [])

        final_rows = []
        seen_fingerprints = set()

        # 1. Topの行は「無条件で正」として採用
        # ログを見る限り、Topの方が半角カナも列構造も正確であるため
        top_rows = results.get("Top", {}).get("table_rows", [])
        for row in top_rows:
            if not row or all(str(c).strip() == "" for c in row): continue
            
            # ヘッダー行判定は一応残すが、データ行なら通す
            if self._is_header_row(row): continue

            cleaned_row = [self._clean_text(c) for c in row]
            
            # 指紋を記録（日付_金額）
            fp = self._get_row_fingerprint(cleaned_row)
            if fp: 
                seen_fingerprints.add(fp)
            
            final_rows.append(cleaned_row)

        # 2. Bottomの行は「Topにない新しい行」だけ採用
        bottom_rows = results.get("Bottom", {}).get("table_rows", [])
        
        for b_row in bottom_rows:
            if not b_row or all(str(c).strip() == "" for c in b_row): continue
            if self._is_header_row(b_row): continue

            b_cleaned = [self._clean_text(c) for c in b_row]
            b_fp = self._get_row_fingerprint(b_cleaned)
            
            # もしTopに同じ日付・金額の行があれば、「重複」とみなしてBottom側を捨てる
            # (Top側の半角カナデータ守るため、合成もしない)
            if b_fp and b_fp in seen_fingerprints:
                continue 
            
            # 新しい行なら追加
            final_rows.append(b_cleaned)

        combined_json["table_rows"] = final_rows
        return combined_json, len(final_rows)

    def _format_to_ui_data(self, combined_json):
        formatted_rows = []

        # 1. 文書情報
        doc_info = combined_json.get("document_info", {})
        title_text = self._clean_text(doc_info.get('title'))
        if title_text: formatted_rows.append([{'text': f"■ {title_text}"}])
        
        org_info = []
        for key in ['org_name', 'sub_name', 'bank_name', 'branch_name']:
            val = self._clean_text(doc_info.get(key))
            if val: org_info.append(val)
        
        if org_info: formatted_rows.append([{'text': " ".join(org_info)}])

        meta_texts = []
        if doc_info.get("account_name"): meta_texts.append(f"名義: {self._clean_text(doc_info['account_name'])}")
        if doc_info.get("period"): meta_texts.append(f"期間: {self._clean_text(doc_info['period'])}")
        if doc_info.get("other_info"): meta_texts.append(self._clean_text(doc_info['other_info']))
        if meta_texts: formatted_rows.append([{'text': " / ".join(meta_texts)}])
        
        formatted_rows.append([{'text': ""}])

        # 2. ヘッダー
        headers = combined_json.get("table_headers", [])
        if headers:
            clean_headers = [self._clean_text(h) for h in headers]
            formatted_rows.append([{'text': h} for h in clean_headers])

        # 3. 明細データ
        for row in combined_json.get("table_rows", []):
            formatted_cells = [{'text': self._clean_text(cell)} for cell in row]
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
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Top-Priority Mode...")
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