from django.urls import path
from . import views

urlpatterns = [
    # path('', ビュー関数名, name='識別名')
    # '' は「アプリのトップページ」を意味します
    path('', views.ocr_view, name='ocr_view'),
]