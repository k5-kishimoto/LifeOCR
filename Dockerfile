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

# 3. ポート8080を開放
EXPOSE 8080

# 4. 起動コマンド
# 0.0.0.0:8080 で起動するように指定します
CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]