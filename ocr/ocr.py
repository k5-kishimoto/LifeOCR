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
        """初期化: Gemini APIの設定とモデルの準備"""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        if not self.api_key:
            print("❌ Error: 'GEMINI_API_KEY' not found.")
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model_name = os.environ.get("GEMINI_VERSION", "gemini-2.5-flash")
            
            # JSONモード設定
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
            print(f"⚙️ Initial Model config: {self.model_name} (Refactored Mode)")

        except Exception as e:
            print(f"❌ API Configuration Error: {e}")

    # =========================================================================
    # 🖼️ 画像処理関連メソッド
    # =========================================================================

    def _optimize_image(self, img):
        """画像をOCR向けに最適化（リサイズ、コントラスト、シャープネス）"""
        # メモリ対策: 解像度を2560pxに制限
        max_size = 2560 
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # オートコントラストで文字をくっきりさせる
        img = ImageOps.autocontrast(img, cutoff=1)
        
        # シャープネス強化（細い文字対策）
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.4) 
        
        return img

    def _split_image(self, img):
        """
        長い明細書を読み切るために、画像を上下に分割する。
        境界線の文字切れを防ぐため、20%ほど重複（オーバーラップ）させる。
        """
        width, height = img.size
        split_ratio = 0.60 # 上部は60%まで
        overlap = 0.40     # 下部は40%から開始（20%重複）
        
        crop_top = img.crop((0, 0, width, int(height * split_ratio)))
        crop_bottom = img.crop((0, int(height * overlap), width, height))
        
        return [("Top", crop_top), ("Bottom", crop_bottom)]

    # =========================================================================
    # 🧠 AI・データ処理関連メソッド
    # =========================================================================

    def _repair_json(self, text):
        """壊れたJSON文字列を可能な限り修復して辞書型に変換する"""
        # 1. まず標準的なパースを試みる
        try:
            cleaned = text.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:-3]
            elif cleaned.startswith("```"): cleaned = cleaned[3:-3]
            return json.loads(cleaned)
        except:
            pass

        # 2. 軽微な破損（閉じ忘れ）の修復
        try:
            if cleaned.count('"') % 2 != 0: cleaned += '"'
            if not cleaned.endswith("}"): cleaned += "}]}"
            return json.loads(cleaned)
        except:
            pass
            
        # 3. 最終手段: 正規表現で「行データらしきもの」だけを抜き出す
        try:
            rows = re.findall(r'\[\s*"(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")*\s*\]', text, re.DOTALL)
            if rows:
                valid_rows = []
                for r in rows:
                    try:
                        row_data = json.loads(r)
                        if isinstance(row_data, list): valid_rows.append(row_data)
                    except: pass
                if valid_rows:
                    return {"table_rows": valid_rows}
        except:
            pass

        return None

    def _call_ai_api(self, image_part, part_label):
        """Gemini APIを呼び出す（リトライロジック付き）"""
        prompt = """
        あなたは高精度の日本語OCRエンジンです。
        画像は書類の一部（上半分または下半分）です。
        見えている範囲のすべての情報を抽出し、JSONを返してください。

        【重要ルール】
        1. 文字種の維持: 半角カナ(`ﾌﾘｺﾐ`)は半角のまま。全角変換禁止。
        2. 空白の維持: 氏名の間のスペースは削除しない。
        
        【出力フォーマット (JSON)】
        {
          "document_info": { "bank_name": "銀行名", "branch_name": "支店名", "title": "文書タイトル", "account_name": "口座名義", "period": "期間", "other_info": "その他" },
          "table_headers": ["日付", "摘要", "お支払金額", "お預り金額", "差引残高", "取扱店"],
          "table_rows": [ ["2026-01-22", "ﾌﾘｺﾐ ﾔﾏﾀﾞ ﾀﾛｳ", "10,000", "", "50,000", "本店"] ]
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
                print(f"⚠️ API Error ({part_label} - {current_model_name}): {e}")
                time.sleep(1)
                continue
        
        return None

    # =========================================================================
    # 🔄 データ結合・整形メソッド
    # =========================================================================

    def _merge_split_results(self, results):
        """TopとBottomの解析結果を結合し、重複行を削除する"""
        combined_json = { "document_info": {}, "table_headers": [], "table_rows": [] }

        # 1. ヘッダー情報の取得（Topを優先）
        target_source = "Top" if "Top" in results else "Bottom"
        if target_source in results:
            combined_json["document_info"] = results[target_source].get("document_info", {})
            combined_json["table_headers"] = results[target_source].get("table_headers", [])

        # 2. 行データの結合
        raw_rows = []
        if "Top" in results: raw_rows.extend(results["Top"].get("table_rows", []))
        if "Bottom" in results: raw_rows.extend(results["Bottom"].get("table_rows", []))

        # 3. 重複行の排除（オーバーラップ部分の処理）
        seen = set()
        unique_rows = []
        for row in raw_rows:
            # 行の中身をすべて結合してユニークIDを作る
            row_vals = []
            for c in row:
                if isinstance(c, (dict, list)): row_vals.append(str(c))
                else: row_vals.append(str(c).strip())
            
            row_id = "".join(row_vals)
            
            # 初めて見る行なら採用
            if row_id and row_id not in seen:
                seen.add(row_id)
                unique_rows.append(row)
        
        combined_json["table_rows"] = unique_rows
        return combined_json, len(unique_rows)

    def _format_to_ui_data(self, combined_json):
        """JSONデータをアプリ表示用の形式（リストのリスト）に変換する"""
        formatted_rows = []

        # ヘルパー: 安全な文字列変換
        def safe_str(val):
            if val is None: return ""
            if isinstance(val, (dict, list)): return str(val)
            return str(val).strip()

        # 1. 文書情報（タイトル・銀行名など）
        doc_info = combined_json.get("document_info", {})
        
        # タイトル
        title_text = safe_str(doc_info.get('title')) or "明細書"
        formatted_rows.append([{'text': f"■ {title_text}", 'is_header': True}])
        
        # 銀行・支店
        bank_info = []
        if doc_info.get("bank_name"): bank_info.append(f"🏦 {safe_str(doc_info['bank_name'])}")
        if doc_info.get("branch_name"): bank_info.append(f"🏢 {safe_str(doc_info['branch_name'])}")
        if bank_info: formatted_rows.append([{'text': " ".join(bank_info)}])

        # 口座・期間など
        meta_texts = []
        if doc_info.get("account_name"): meta_texts.append(f"名義: {safe_str(doc_info['account_name'])}")
        if doc_info.get("period"): meta_texts.append(f"期間: {safe_str(doc_info['period'])}")
        if doc_info.get("other_info"): meta_texts.append(safe_str(doc_info['other_info']))
        if meta_texts: formatted_rows.append([{'text': " / ".join(meta_texts)}])
        
        formatted_rows.append([{'text': ""}]) # 空行

        # 2. 表ヘッダー
        headers = combined_json.get("table_headers", [])
        if headers:
            clean_headers = [safe_str(h) for h in headers]
            formatted_rows.append([{'text': h, 'is_header': True} for h in clean_headers])

        # 3. 明細データ
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
    # 🚀 メイン処理フロー
    # =========================================================================

    def _process_single_page(self, args):
        """1ページ分の処理を実行（画像最適化 -> 分割 -> 並列OCR -> 結合 -> 整形）"""
        page_label, pil_image = args
        
        # 1. 画像の最適化
        optimized_image = self._optimize_image(pil_image)
        
        # 2. 画像の分割（Top/Bottom）
        parts = self._split_image(optimized_image)
        
        # 3. 並列でAIリクエスト送信
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_part = {}
            for p_name, p_img in parts:
                # WebP変換
                img_byte_arr = io.BytesIO()
                p_img.save(img_byte_arr, format='WEBP', quality=85)
                image_part = {"mime_type": "image/webp", "data": img_byte_arr.getvalue()}
                
                future = executor.submit(self._call_ai_api, image_part, f"{page_label}-{p_name}")
                future_to_part[future] = p_name

            # 結果の回収と修復
            for future in concurrent.futures.as_completed(future_to_part):
                p_name = future_to_part[future]
                res_text = future.result()
                
                if res_text:
                    repaired_data = self._repair_json(res_text)
                    if repaired_data:
                        results[p_name] = repaired_data
                    else:
                        print(f"❌ JSON Repair Failed for {p_name}")

        # 4. 結果の結合（マージ）
        combined_json, row_count = self._merge_split_results(results)
        
        # 5. アプリ表示用に整形
        formatted_rows = self._format_to_ui_data(combined_json)
        
        print(f"✅ Success ({page_label}) - Merged {row_count} rows")
        return (page_label, formatted_rows)


    def extract_text(self, uploaded_file):
        """外部から呼ばれるメインメソッド"""
        print(f"⏳ Starting Gemini AI OCR ({self.model_name}) - Refactored Mode...")
        
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

        # ページ単位の並列処理
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