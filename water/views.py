from django.shortcuts import render
from .ocr import engine

def ocr_view(request):
    results = []
    
    # ページを開いた時
    print("--- View called ---")

    if request.method == 'POST':
        print("1. POST request received")
        
        # 画像ファイルが届いているか確認
        if 'image' in request.FILES:
            print("2. Image file found in request")
            uploaded_file = request.FILES['image']
            
            try:
                print("3. Starting OCR processing...")
                results = engine.extract_text(uploaded_file)
                print(f"4. OCR Results: {results}")
            except Exception as e:
                print(f"!!! Error during OCR: {e}")
        else:
            print("!!! No image file in request.FILES (HTML check required)")
    
    else:
        print("GET request (Initial page load)")

    return render(request, 'ocr.html', {'results': results})