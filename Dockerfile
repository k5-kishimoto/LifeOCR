ARG PYTHON_VERSION=3.12-slim
FROM python:${PYTHON_VERSION}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. システムライブラリのインストール
# PaddleOCR(libgomp1, libgl1), OpenCV(libglib2.0), PDF(poppler-utils) に必要
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# 2. Pythonライブラリのインストール
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /root/.cache/

COPY . /code

EXPOSE 8000
# ❌ 変更前: 開発用サーバー（不安定なのでNG）
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# ⭕ 変更後: 本番用サーバー Gunicorn を使用
# --timeout 120 : OCRが120秒かかっても切断されないようにする
# --workers 1   : メモリ節約のため1プロセスに制限
CMD ["gunicorn", "LifeOCR.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120"]