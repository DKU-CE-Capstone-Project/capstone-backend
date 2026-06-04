# EconMind 백엔드 — api/worker 공용 이미지
# 기본 CMD = api(uvicorn). worker Deployment는 command를 override:
#   command: ["python", "-m", "app.worker"]
FROM python:3.11-slim

# 컨테이너 로그 즉시 flush (worker 처리 로그가 데모 영상에 실시간 표시됨)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 시스템 의존성 (lxml/trafilatura 빌드 안정화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# 패키지 메타 + 소스 복사 후 설치
COPY pyproject.toml ./
COPY app ./app
COPY fixtures ./fixtures
RUN pip install --no-cache-dir .

EXPOSE 8000

# 기본 진입점: api 프로세스
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
