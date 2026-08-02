FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*
    
# Install C/C++ compilation tools and CMake
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements-docker.txt ./
RUN pip install --upgrade pip && pip install -r requirements-docker.txt

COPY requirements-docker.txt ./
RUN pip install --upgrade pip && pip install -r requirements-docker.txt

COPY . .

RUN mkdir -p /data/registered_faces /root/.insightface

EXPOSE 5000

CMD ["python", "app.py"]
