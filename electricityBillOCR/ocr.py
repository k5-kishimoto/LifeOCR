import os
import json
import io
import re
import concurrent.futures
from typing import List, Dict, Any, Optional

from pdf2image import convert_from_bytes
import google.generativeai as genai
from PIL import Image, ImageOps
from dotenv import load_dotenv

load_dotenv()

class OcrEngine:
    """
    電気料金明細から情報を抽出し、CSV形式（2次元リスト）で返却するエンジン。
    """

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model_name = os.environ.get("GEMINI_VERSION", "gemini-2.0-flash")
        self.model = self._setup_model()

    def _setup_model(self) -> Optional[genai.GenerativeModel]:
        if not self.api_key:
            print("❌ エラー: GEMINI_API_KEYが設定されていません。")
            return None
        
        try:
            genai.configure(api_key=self.api_key)
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            return genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json"
                },
                safety_settings=safety_settings
            )
        except Exception as e:
            print(f"❌ モデルセットアップ失敗: {e}")
            return None

    def _clean_value(self, val: Any) -> str:
        if val is None:
            return ""
        s = str(val).strip()
        if s.lower() in ["none", "null"]:
            return ""
        return s.replace("\n", " ").replace("\r", " ")

    def _prepare_image_bytes(self, img: Image.Image) -> bytes:
        img = ImageOps.autocontrast(img.convert('RGB'), cutoff=1)
        buf = io.BytesIO()
        img.save(buf, format='WEBP', quality=90)
        return buf.getvalue()

    def _request_ocr(self, image_bytes: bytes) -> Any:
        prompt = """
        あなたは電気料金明細から重要情報を抽出するAIアシスタントです。
        画像から以下の情報を抽出し、JSON形式で出力してください。
        
        抽出するキー:
        - 「会社名」: 電力会社の名称
        - 「請求年月」: 請求月（例：2025-01、yyyy-mm）
        - 「支払期日」: 支払期日（例：2025/01/05）
        - 「請求総額」: 請求総額（整数のみ、カンマと通貨記号を除去）
        - 「電力使用量」: 電力使用量（kWh単位）
        - 「ご契約種別」： ご契約種別（例：従量電灯、低圧電力など）
        - 「部屋番号」: 部屋番号（例：共用、201、4-A）
        - 「科目種別」: 「電気料(部屋番号)」の形式。部屋番号をカッコで囲んでください（例：電気料(共用)、電気料(201)）
        - 「物件名」: 物件名（例：6floor、5floor、メゾンソレイユ、Anchor）
        - 「場所」: 使用場所の住所
        
        出力例:
        {
            "会社名": "沖縄電力",
            "請求年月": "2025-01",
            "支払期日": "2025/01/05",
            "請求総額": 3350,
            "電力使用量": 45,
            "ご契約種別": "従量電灯",
            "部屋番号": "201",
            "科目種別": "電気料(201)",
            "物件名": "メゾンソレイユ",
            "場所": "糸満市..."
        }
        
        RAW JSONのみを返す。マークダウンは不可。
        """
        try:
            content = [{"mime_type": "image/webp", "data": image_bytes}, prompt]
            response = self.model.generate_content(content)
            
            raw_text = response.text.strip()
            cleaned_json = re.sub(r'^```json\s*|\s*```$', '', raw_text, flags=re.MULTILINE)
            parsed = json.loads(cleaned_json)
            
            if isinstance(parsed, dict):
                # 基本的にフラットな辞書を返すが、AIが気を利かせてリスト化したとき用の保険
                for key in ["data", "table", "rows"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                return parsed

            return parsed if isinstance(parsed, list) else None

        except Exception as e:
            print(f"⚠️ OCRリクエスト中にエラーが発生: {e}")
            return None

    def _process_page(self, page_index: int, img: Image.Image) -> Any:
        image_bytes = self._prepare_image_bytes(img)
        return self._request_ocr(image_bytes)

    def extract_text(self, uploaded_file) -> List[List[Dict[str, str]]]:
        if not self.model:
            return [[{"text": "AIモデルが正常に初期化されていません。"}] ]

        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        try:
            filename = uploaded_file.name.lower()
            if filename.endswith('.pdf'):
                images = convert_from_bytes(file_bytes, dpi=200)
            else:
                images = [Image.open(io.BytesIO(file_bytes))]
        except Exception as e:
            return [[{"text": f"ファイルの読み込みに失敗しました: {e}"}]]

        results_list = [None] * len(images)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_idx = {
                executor.submit(self._process_page, i, img): i 
                for i, img in enumerate(images)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results_list[idx] = future.result()

        final_rows = []
        # 有効なデータが入っている最初のページをサンプルとして取得
        valid_sample = next((d for d in results_list if isinstance(d, dict)), None)
        
        if not valid_sample:
            return [[{"text": "データを抽出できませんでした。"}]]

        # ヘッダー行の作成
        headers = list(valid_sample.keys())
        final_rows.append([{"text": h} for h in headers])
        
        # データ行の作成
        for page_data in results_list:
            if isinstance(page_data, dict):
                # --- ★科目種別の形式を強制適用: 電気料(部屋番号) ---
                room_no = self._clean_value(page_data.get("部屋番号", ""))
                page_data["科目種別"] = f"電気料({room_no})"
                # -----------------------------------------------
                
                row = [{"text": self._clean_value(page_data.get(h, ""))} for h in headers]
                final_rows.append(row)

        return final_rows

engine = OcrEngine()