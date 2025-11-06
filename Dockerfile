FROM python:3.13-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync

COPY . .

EXPOSE 8501

ENTRYPOINT ["uv", "run", "streamlit", "run", "main.py"]

