import os
import multiprocessing
import psutil  # ★デバッグ用

# --- リソースデバッグ用関数 ---
def log_resources(tag=""):
    """
    現在のメモリ使用量(RSS)と、
    '前回の呼び出しから現在まで'のCPU使用率を表示する
    """
    process = psutil.Process(os.getpid())
    
    # メモリ (MB)
    mem_mb = process.memory_info().rss / 1024 / 1024
    
    # CPU (%) 
    # interval=None は '前回の呼び出し以降の平均' を返します
    # 初回呼び出し時は 0.0 になる仕様ですが、初期化呼び出しを入れています
    cpu_pct = process.cpu_percent(interval=None)
    
    print(f"📊 [RES] MEM: {mem_mb:7.2f} MB | CPU: {cpu_pct:6.1f}% | {tag}")

# --- CPU設定 ---
try:
    num_cores = str(multiprocessing.cpu_count())
except Exception:
    num_cores = '1'

os.environ['OMP_NUM_THREADS'] = num_cores
os.environ['MKL_NUM_THREADS'] = num_cores
os.environ['PADDLE_NUM_THREADS'] = num_cores

print(f"🚀 CPU Optimization: Using {num_cores} threads.")

# ★CPU計測の基準点を作るため、一度空呼び出しします（戻り値は捨てます）
psutil.Process(os.getpid()).cpu_percent(interval=None)
log_resources("Script Start") # ★デバッグ

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
            log_resources("Before Model Load") # ★デバッグ
            print("⏳ Loading PaddleOCR model (Speed & Memory Optimized)...")
            from paddleocr import PaddleOCR
            
            self._ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang='japan', 
                enable_mkldnn=True, 
                det_limit_side_len=640,
                rec_batch_num=100,
            )
            print("✅ Model loaded!")
            log_resources("After Model Load") # ★デバッグ
        return self._ocr_model

    def extract_text(self, uploaded_file):
        print("⏳ Starting text extraction...")
        log_resources("Start extract_text") # ★デバッグ

        file_bytes = uploaded_file.read()
        
        try:
            print("⏳ Determining file type...")
            filename = uploaded_file.name.lower()
        except AttributeError:
            filename = "unknown.jpg"
            
        all_rows = []

        # --- A. PDFの場合 ---
        if filename.endswith('.pdf'):
            try:
                print("⏳X1. grayscale PDF converted to images.")
                log_resources("Before PDF Convert") # ★デバッグ
                
                pil_images = convert_from_bytes(file_bytes, dpi=200, grayscale=True)
                
                # PDF変換処理でどれくらいCPUを使ったか確認
                log_resources(f"After PDF Convert (Pages: {len(pil_images)})") # ★デバッグ
                
                for i, pil_img in enumerate(pil_images):
                    log_resources(f"Processing Page {i+1} Start") # ★デバッグ

                    gray_image = np.array(pil_img)
                    
                    print("⏳X2. Image resized for OCR.")
                    gray_image = self._resize_image_if_too_large(gray_image)
                    
                    print("⏳X3. Image converted to BGR format for PaddleOCR.")
                    bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                    
                    print("⏳X4. OCR processing completed for one page.")
                    log_resources(f"Page {i+1} Before OCR") # ★デバッグ
                    
                    page_rows = self._process_one_image(bgr_image)
                    
                    if page_rows:
                        if i > 0:
                            print("⏳X5. Page separator added.")
                            all_rows.append([{'text': f'--- {i+1}ページ目 ---', 'score': ''}])
                        all_rows.extend(page_rows)
                    
                    pil_img = None 
                    bgr_image = None
                    # OCR処理でどれくらいCPUを使ったか確認
                    log_resources(f"Processing Page {i+1} End") # ★デバッグ

            except Exception as e:
                print(f"PDF Error: {e}")
                return []

        # --- B. 画像の場合 ---
        else:
            print("⏳X0. Processing image file for OCR.")
            log_resources("Before Image Decode") # ★デバッグ

            img_np = np.frombuffer(file_bytes, np.uint8)
            
            print("⏳X1. Image loaded in grayscale for OCR.")
            gray_image = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
            
            if gray_image is not None:
                print("⏳X2. Image resized for OCR.")
                gray_image = self._resize_image_if_too_large(gray_image)
                
                print("⏳X3. Image converted to BGR format for PaddleOCR.")
                bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                
                print("⏳X4. OCR processing completed for image.")
                log_resources("Before Image OCR") # ★デバッグ
                all_rows = self._process_one_image(bgr_image)
                
        log_resources("End extract_text") # ★デバッグ
        return all_rows

    def _resize_image_if_too_large(self, img, max_width=1440):
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_height = int(h * scale)
            print("⏳XX1. Image resized to width:", max_width)
            img = cv2.resize(img, (max_width, new_height), interpolation=cv2.INTER_AREA)
        return img

    def _process_one_image(self, img):
        print("⏳XX2. Running OCR on the image.")
        result = self.ocr.ocr(img)

        log_resources("OCR Complete") # ★デバッグ
        raw_items = []
        if isinstance(result, list) and len(result) > 0:
            if result[0] is None:
                print("⏳XX3. No text detected in the image.")
                return []
            if isinstance(result[0], dict):
                print("⏳XX3. Processing OCR results in dict format.")
                data = result[0]
                dt_boxes = data.get('dt_polys', [])
                rec_texts = data.get('rec_texts', [])
                rec_scores = data.get('rec_scores', [])
                print(f"⏳XX4. Detected {len(dt_boxes)} text boxes.")
                for box, text, score in zip(dt_boxes, rec_texts, rec_scores):
                    raw_items.append({'box': box, 'text': text, 'score': score})
            elif isinstance(result[0], list):
                print("⏳XX3. Processing OCR results in list format.")
                for line in result[0]:
                    if line is not None:
                        raw_items.append({'box': line[0], 'text': line[1][0], 'score': line[1][1]})

        if not raw_items:
            return []
        print(f"⏳XX5. Total {len(raw_items)} text items extracted.")
        raw_items.sort(key=lambda x: x['box'][0][1])

        rows = []
        current_row = []
        last_y = -1
        threshold = 15 
        print("⏳XX6. Grouping text items into rows.")
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
            print("⏳XX7. Finalizing last row.")
            current_row.sort(key=lambda x: x['box'][0][0])
            rows.append(current_row)

        return rows

engine = OcrEngine()