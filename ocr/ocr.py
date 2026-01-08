from paddleocr import PaddleOCR
import numpy as np
import cv2
from pdf2image import convert_from_bytes # 追加

class OcrEngine:
    def __init__(self):
        # 初期化
        self.ocr = PaddleOCR(use_angle_cls=True, lang='japan')

    def extract_text(self, uploaded_file):
        """
        ファイルを受け取り、PDFか画像かを判定して処理を振り分ける
        """
        # ファイルの中身をバイト列として読み込む
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name.lower()
        
        all_rows = []

        # --- A. PDFの場合 ---
        if filename.endswith('.pdf'):
            try:
                # PDFを画像のリストに変換 (DPI=200くらいが丁度いいです)
                pil_images = convert_from_bytes(file_bytes, dpi=200)
                
                for i, pil_img in enumerate(pil_images):
                    # PIL形式からOpenCV形式(numpy)に変換
                    open_cv_image = np.array(pil_img)
                    # 色の並びをRGBからBGRに変換 (OpenCV用)
                    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
                    
                    # 1ページ分のOCR実行
                    page_rows = self._process_one_image(open_cv_image)
                    
                    # 結果を統合
                    if page_rows:
                        # ページ区切りのためのダミー行を入れる（見やすくするため）
                        if i > 0:
                            all_rows.append([{'text': f'--- {i+1}ページ目 ---', 'score': ''}])
                        all_rows.extend(page_rows)
                        
            except Exception as e:
                print(f"PDF Error: {e}")
                return []

        # --- B. 画像の場合 ---
        else:
            # バイト列を画像データにデコード
            img_np = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            if img is not None:
                all_rows = self._process_one_image(img)

        return all_rows

    def _process_one_image(self, img):
        """
        1枚の画像データ(numpy array)を受け取り、整形された行リストを返す
        （前回のロジックをここに移動しました）
        """
        result = self.ocr.ocr(img)

        raw_items = []
        
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                data = result[0]
                dt_boxes = data.get('dt_polys', [])
                rec_texts = data.get('rec_texts', [])
                rec_scores = data.get('rec_scores', [])
                for box, text, score in zip(dt_boxes, rec_texts, rec_scores):
                    raw_items.append({'box': box, 'text': text, 'score': score})

            elif isinstance(result[0], list):
                for line in result[0]:
                    raw_items.append({'box': line[0], 'text': line[1][0], 'score': line[1][1]})

        if not raw_items:
            return []

        # Y座標でソート
        raw_items.sort(key=lambda x: x['box'][0][1])

        # 行のグルーピング処理
        rows = []
        current_row = []
        last_y = -1
        threshold = 15

        for item in raw_items:
            current_y = item['box'][0][1]
            
            if last_y == -1:
                current_row.append(item)
                last_y = current_y
            elif abs(current_y - last_y) < threshold:
                current_row.append(item)
            else:
                current_row.sort(key=lambda x: x['box'][0][0])
                rows.append(current_row)
                current_row = [item]
                last_y = current_y

        if current_row:
            current_row.sort(key=lambda x: x['box'][0][0])
            rows.append(current_row)

        return rows

engine = OcrEngine()