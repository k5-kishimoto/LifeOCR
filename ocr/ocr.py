from paddleocr import PaddleOCR
import numpy as np
import cv2
from pdf2image import convert_from_bytes

class OcrEngine:
    def __init__(self):
        # ★重要：起動時にはモデルを読み込まない（Noneにしておく）
        self._ocr_model = None

    @property
    def ocr(self):
        """
        必要な時に初めてモデルを読み込む（遅延ロード）
        """
        if self._ocr_model is None:
            print("⏳ Loading PaddleOCR model for the first time...")
            # ここで初めて重い処理が走る
            self._ocr_model = PaddleOCR(use_angle_cls=True, lang='japan')
            print("✅ Model loaded!")
        return self._ocr_model

    def extract_text(self, uploaded_file):
        # ファイル読み込み処理
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name.lower()
        all_rows = []

        # --- A. PDFの場合 ---
        if filename.endswith('.pdf'):
            try:
                pil_images = convert_from_bytes(file_bytes, dpi=200)
                for i, pil_img in enumerate(pil_images):
                    open_cv_image = np.array(pil_img)
                    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
                    
                    page_rows = self._process_one_image(open_cv_image)
                    
                    if page_rows:
                        if i > 0:
                            all_rows.append([{'text': f'--- {i+1}ページ目 ---', 'score': ''}])
                        all_rows.extend(page_rows)
            except Exception as e:
                print(f"PDF Error: {e}")
                return []

        # --- B. 画像の場合 ---
        else:
            img_np = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if img is not None:
                all_rows = self._process_one_image(img)

        return all_rows

    def _process_one_image(self, img):
        # ★ここで self.ocr を呼ぶと、自動的にロード処理が走ります
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

        raw_items.sort(key=lambda x: x['box'][0][1])

        rows = []
        current_row = []
        last_y = -1
        threshold = 60

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