FROM python:3.13-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync

COPY /rag ./rag
COPY /ui ./ui
COPY main.py .

EXPOSE $STREAMLIT_SERVER_PORT

ENTRYPOINT ["uv", "run", "streamlit", "run", "main.py"]

