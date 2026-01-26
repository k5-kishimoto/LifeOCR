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

# .envファイルから環境変数を読み込み
load_dotenv()

class OcrEngine:
    """
    銀行明細などの画像/PDFからテキストを抽出するOCRエンジンクラス。
    Gemini APIを利用し、構造化された2次元リスト形式でデータを返却します。
    """

    def __init__(self):
        # APIキーの取得
        self.api_key = os.environ.get("GEMINI_API_KEY")
        # モデル名は環境変数から取得（デフォルトは安定の2.0-flash）
        self.model_name = os.environ.get("GEMINI_VERSION", "gemini-2.0-flash")
        # モデルの初期化
        self.model = self._setup_model()

    def _setup_model(self) -> Optional[genai.GenerativeModel]:
        """
        AIモデルの初期設定を行います。
        セーフティ設定による400エラーを回避するための重要なフェーズです。
        """
        if not self.api_key:
            print("❌ エラー: GEMINI_API_KEYが設定されていません。")
            return None
        
        try:
            genai.configure(api_key=self.api_key)
            
            # 400エラー（リクエスト拒絶）を避けるため、
            # サポートされている基本4カテゴリーのみに制限してBLOCK_NONEを設定
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            return genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.0, # 抽出精度を上げるためランダム性を排除
                    "response_mime_type": "application/json" # JSONで返却を強制
                },
                safety_settings=safety_settings
            )
        except Exception as e:
            print(f"❌ モデルセットアップ失敗: {e}")
            return None

    # -------------------------------------------------------------------------
    # データクレンジング・前処理
    # -------------------------------------------------------------------------

    def _clean_value(self, val: Any) -> str:
        """
        AIから返ってきた生データを、人間（とUI）が見やすい形式に整えます。
        特に'None'や'null'の混入を徹底的に防ぎます。
        """
        if val is None:
            return ""
        
        # 文字列に変換して前後の空白を除去
        s = str(val).strip()
        
        # AIが文字列として "None" や "null" と返してきた場合も空文字にする
        if s.lower() in ["none", "null"]:
            return ""
        
        # セル内改行は表が崩れる原因になるため、スペースに置換
        return s.replace("\n", " ").replace("\r", " ")

    def _prepare_image_bytes(self, img: Image.Image) -> bytes:
        """
        OCRの精度を最大化するために画像を最適化します。
        オートコントラストをかけ、AIが読みやすいWEBP形式に変換します。
        """
        # オートコントラスト：暗い文字をはっきりさせ、背景のノイズ（影など）を飛ばす
        img = ImageOps.autocontrast(img.convert('RGB'), cutoff=1)
        
        buf = io.BytesIO()
        # WEBPは軽量で高画質なため、API転送速度と精度のバランスが良い
        img.save(buf, format='WEBP', quality=90)
        return buf.getvalue()

    # -------------------------------------------------------------------------
    # AI 実行コアロジック
    # -------------------------------------------------------------------------

    def _request_ocr(self, image_bytes: bytes) -> List[List[str]]:
        """
        Gemini APIにリクエストを投げ、レスポンスからデータの塊を抽出します。
        """
        prompt = """
        あなたは日本語OCRエンジンです。画像内の情報を「2次元の配列」として抽出してください。
        【重要】
        - 行と列の構造を維持すること。
        - カナ、数字、記号は見たままを維持すること。
        - 摘要や名前などの「右端の列」を絶対に省略しないこと。
        - 空欄（データなし）の場所はnullとして出力すること。
        """
        try:
            # 画像データとプロンプトをセット
            content = [{"mime_type": "image/webp", "data": image_bytes}, prompt]
            response = self.model.generate_content(content)
            
            # JSON文字列のクリーンアップ
            raw_text = response.text.strip()
            # Markdownのコードブロック(```json ... ```)を正規表現で除去
            cleaned_json = re.sub(r'^```json\s*|\s*```$', '', raw_text, flags=re.MULTILINE)
            parsed = json.loads(cleaned_json)
            
            # AIがどのようなキー名（data, table, rowsなど）で返してきても対応できるよう探索
            if isinstance(parsed, dict):
                # 可能性のある主要なキーをチェック
                for key in ["data", "table", "rows"]:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                # キーが見当たらない場合、中身にある「最初のリスト」を拾い上げる
                for v in parsed.values():
                    if isinstance(v, list): return v
            
            # 辞書ではなく最初からリストで返ってきた場合
            if isinstance(parsed, list):
                return parsed
                
            return []
        except Exception as e:
            print(f"⚠️ OCRリクエスト中にエラーが発生: {e}")
            return []

    # -------------------------------------------------------------------------
    # ページ管理・並列実行
    # -------------------------------------------------------------------------

    def _process_page(self, page_index: int, img: Image.Image) -> List[List[Dict[str, str]]]:
        """
        特定の1ページを処理するユニット。
        スレッド並列化のためにメソッドを分離しています。
        """
        # 画像のバイト変換
        image_bytes = self._prepare_image_bytes(img)
        # AI実行
        matrix = self._request_ocr(image_bytes)
        
        page_results = []
        # 画面上でわかりやすいようにページ見出しを挿入
        page_results.append([{"text": f"--- Page {page_index + 1} ---"}])
        
        if not matrix:
            page_results.append([{"text": "⚠️ このページのデータ抽出に失敗しました"}])
            return page_results

        # 各行をUI用の辞書形式 [{'text': '...'}] にマッピング
        for row in matrix:
            if isinstance(row, list):
                # リスト内の各要素をクレンジングして格納
                page_results.append([{"text": self._clean_value(cell)} for cell in row])
            else:
                # 1列しかない行（タイトルなど）の場合
                page_results.append([{"text": self._clean_value(row)}])
        
        return page_results

    # -------------------------------------------------------------------------
    # 公開メソッド（外部から呼び出すのはこれだけ）
    # -------------------------------------------------------------------------

    def extract_text(self, uploaded_file) -> List[List[Dict[str, str]]]:
        """
        アップロードされたファイルを読み取り、全ページのOCR結果を返します。
        PDFの場合は自動的にページ分割して並列処理します。
        """
        if not self.model:
            return [[{"text": "AIモデルが正常に初期化されていません。"}]]

        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        
        # 1. 画像への変換処理
        try:
            filename = uploaded_file.name.lower()
            if filename.endswith('.pdf'):
                # PDFはDPI 200程度が精度と速度のバランスが良い
                images = convert_from_bytes(file_bytes, dpi=200)
            else:
                # 一般画像
                images = [Image.open(io.BytesIO(file_bytes))]
        except Exception as e:
            return [[{"text": f"ファイルの読み込みに失敗しました: {e}"}]]

        # 2. 並列処理の実行（スレッドプールを使用）
        final_output = []
        # max_workersはサーバー負荷に合わせて調整。4〜8が一般的
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # ページごとにタスクを割り振り
            future_to_page = {
                executor.submit(self._process_page, i, img): i 
                for i, img in enumerate(images)
            }
            
            # ページ順がバラバラにならないよう、インデックスをキーに格納
            results_by_page = {}
            for future in concurrent.futures.as_completed(future_to_page):
                idx = future_to_page[future]
                results_by_page[idx] = future.result()

        # 3. ページ順（0, 1, 2...）に並べ直して最終リストを作成
        for i in range(len(images)):
            if i in results_by_page:
                final_output.extend(results_by_page[i])

        return final_output

# インスタンス化
engine = OcrEngine()