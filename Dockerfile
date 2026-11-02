FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    GEMELO_DATA_DIR=/data \
    GEMELO_CONFIG=/data/gemelo_config.json \
    GEMELO_USERS=/data/users.json

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Usuario sin privilegios; /data guarda historial, usuarios y umbrales (montar como volumen)
RUN useradd --create-home app && mkdir /data && chown app /data
USER app
VOLUME /data
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
