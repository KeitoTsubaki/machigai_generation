# PyTorch の公式イメージ (CUDA + cuDNN 付き)
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# タイムゾーン設定をスキップする
ENV DEBIAN_FRONTEND=noninteractive

# 必要なパッケージをインストール
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git vim locales libgl1-mesa-glx libglib2.0-0\
    && rm -rf /var/lib/apt/lists/*

# ロケールを UTF-8 に設定
RUN locale-gen ja_JP.UTF-8 \
    && update-locale LANG=ja_JP.UTF-8

# 環境変数を設定（UTF-8）
ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8

# pip をアップグレード
RUN pip install --upgrade pip

# 追加の Python ライブラリをインストール
COPY requirements3.txt .
RUN pip install --no-cache-dir -r requirements3.txt

# vi の設定を UTF-8 に
RUN echo "set encoding=utf-8" >> /root/.vimrc \
    && echo "set fileencodings=utf-8" >> /root/.vimrc

# タイムゾーンを指定して環境変数を設定
ENV TZ=Asia/Tokyo