import numpy as np
import cv2
from pdf2image import convert_from_bytes

class OcrEngine:
    def __init__(self):
        self._ocr_model = None

    @property
    def ocr(self):
        """
        必要な時に初めてモデルを読み込む
        """
        if self._ocr_model is None:
            print("⏳ Loading PaddleOCR model with acceleration...")
            from paddleocr import PaddleOCR
            
            # ★高速化のポイント1：MKLDNNを有効にする
            # これだけでCPUでの推論速度が数倍になることがあります
            self._ocr_model = PaddleOCR(
                use_angle_cls=True,  # 画像の向き補正（不要ならFalseにするとさらに速い）
                lang='japan', 
                enable_mkldnn=True,  # ★CPU高速化ON
                use_gpu=False        # GPUは使わない
            )
            print("✅ Model loaded!")
        return self._ocr_model

    def extract_text(self, uploaded_file):
        file_bytes = uploaded_file.read()
        
        try:
            filename = uploaded_file.name.lower()
        except AttributeError:
            filename = "unknown.jpg"
            
        all_rows = []

        if filename.endswith('.pdf'):
            try:
                # ★高速化のポイント2：PDF変換時の解像度を下げる (200 -> 150)
                # 150dpiでもレシートの文字なら十分読めます
                pil_images = convert_from_bytes(file_bytes, dpi=150)
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

        else:
            img_np = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if img is not None:
                # ★高速化のポイント3：巨大な画像をリサイズする
                # スマホ写真は4000pxとかあるので、幅1280px程度に縮小すると爆速になります
                img = self._resize_image_if_too_large(img)
                all_rows = self._process_one_image(img)

        return all_rows

    def _resize_image_if_too_large(self, img, max_width=1280):
        """
        画像が大きすぎる場合にリサイズする関数
        """
        height, width = img.shape[:2]
        if width > max_width:
            # アスペクト比を維持して縮小
            scale = max_width / width
            new_height = int(height * scale)
            img = cv2.resize(img, (max_width, new_height), interpolation=cv2.INTER_AREA)
        return img

    def _process_one_image(self, img):
        result = self.ocr.ocr(img)

        raw_items = []
        if isinstance(result, list) and len(result) > 0:
            if result[0] is None:
                return []
            if isinstance(result[0], dict):
                data = result[0]
                dt_boxes = data.get('dt_polys', [])
                rec_texts = data.get('rec_texts', [])
                rec_scores = data.get('rec_scores', [])
                for box, text, score in zip(dt_boxes, rec_texts, rec_scores):
                    raw_items.append({'box': box, 'text': text, 'score': score})
            elif isinstance(result[0], list):
                for line in result[0]:
                    if line is not None:
                        raw_items.append({'box': line[0], 'text': line[1][0], 'score': line[1][1]})

        if not raw_items:
            return []

        raw_items.sort(key=lambda x: x['box'][0][1])

        rows = []
        current_row = []
        last_y = -1
        threshold = 50 # リサイズした場合はこの閾値も調整が必要かも（一旦そのまま）

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