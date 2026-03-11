FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data')"
ENV NLTK_DATA=/app/nltk_data

COPY configs/ configs/
COPY vision/ vision/
COPY text/ text/
COPY multimodal/ multimodal/
COPY training/ training/
COPY dataset/ dataset/
COPY inference/ inference/
COPY evaluation/ evaluation/
COPY agents/ agents/
COPY retrieval/ retrieval/
COPY distributed/ distributed/
COPY experiments/ experiments/
COPY utils/ utils/
COPY scripts/ scripts/
COPY data/ data/
COPY knowledge/ knowledge/
COPY demo.py .

RUN mkdir -p outputs

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 8000

CMD ["python", "-m", "inference.generate"]
