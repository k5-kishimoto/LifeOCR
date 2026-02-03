from django.contrib import admin
from django.urls import path, include  # <--- ★ include を追加してください！

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # ここでアプリのurls.pyを読み込みます
    # http://127.0.0.1:8000/ にアクセスするとOCR画面が出るようになります
    path('', include('ocr.urls')), 
    path('elec/', include('electricityBillOCR.urls')), 
    path('water/', include('water.urls')), 
]