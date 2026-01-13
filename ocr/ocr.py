import os
import multiprocessing

# --- CPU設定 ---
try:
    num_cores = str(multiprocessing.cpu_count())
except Exception:
    num_cores = '1'

os.environ['OMP_NUM_THREADS'] = num_cores
os.environ['MKL_NUM_THREADS'] = num_cores
os.environ['PADDLE_NUM_THREADS'] = num_cores

print(f"🚀 CPU Optimization: Using {num_cores} threads.")

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
            print("⏳ Loading PaddleOCR model (Speed & Memory Optimized)...")
            from paddleocr import PaddleOCR
            
            self._ocr_model = PaddleOCR(
                # 高速化のため向き補正OFF（必要ならTrueに戻してください）
                use_angle_cls=True,
                lang='japan', 
                enable_mkldnn=False, 
                # ★【高速化の秘策】
                # 「文字を探す（Detection）」時の画像サイズを小さく制限します。
                # デフォルトは960ですが、これを736などに下げると検索が爆速になります。
                # ※「文字を読む（Recognition）」時は元の高画質画像を使うので、
                #   読み取り精度はほとんど落ちません！
                det_limit_side_len=736,
                # ★【高速化2】認識バッチサイズを増やす (デフォルト6 -> 20)
                # 小さい文字領域をまとめて一気に処理させることで速度を稼ぎます
                rec_batch_num=20,
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

        # --- A. PDFの場合 ---
        if filename.endswith('.pdf'):
            try:
                # 【高速化・メモリ節約】
                # grayscale=True で最初から白黒で読み込む
                # dpi=200 で解像度を抑える
                pil_images = convert_from_bytes(file_bytes, dpi=200, grayscale=True)
                
                for i, pil_img in enumerate(pil_images):
                    # PIL(Gray) -> NumPy(Gray 1ch)
                    gray_image = np.array(pil_img)
                    
                    # リサイズ（白黒のまま行うので計算量が1/3で済みます）
                    gray_image = self._resize_image_if_too_large(gray_image)
                    
                    # PaddleOCRは3チャンネル入力を好むため、最後にBGR形式に変換
                    # （見た目は白黒のままですが、データ形式だけ合わせます）
                    bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                    
                    page_rows = self._process_one_image(bgr_image)
                    
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
            
            # 【高速化】cv2.IMREAD_GRAYSCALE で白黒として読み込む
            gray_image = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
            
            if gray_image is not None:
                # リサイズ（白黒のまま行うので高速）
                gray_image = self._resize_image_if_too_large(gray_image)
                
                # PaddleOCR用に3チャンネル形式へ変換
                bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                
                all_rows = self._process_one_image(bgr_image)

        return all_rows

    def _resize_image_if_too_large(self, img, max_width=1440):
        """
        画像が大きすぎる場合にリサイズする関数
        白黒画像のまま処理するため非常に高速です
        """
        # img.shape は (高さ, 幅) または (高さ, 幅, チャンネル)
        h, w = img.shape[:2]
        
        if w > max_width:
            scale = max_width / w
            new_height = int(h * scale)
            # 縮小処理
            img = cv2.resize(img, (max_width, new_height), interpolation=cv2.INTER_AREA)
        return img

    def _process_one_image(self, img):
        # OCR実行
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
        # 画像サイズ縮小済のため、閾値は小さめに設定
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