FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
# Download the model at build time so the container starts fast
RUN python -c "from emotion import EmotionPredictor; EmotionPredictor()"
EXPOSE 7860
CMD ["python", "app.py"]
