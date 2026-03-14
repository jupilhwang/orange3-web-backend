# Orange3 web Backend

Orange3 데이터 분석 플랫폼의 백엔드 서버
Orange3 3.40+ 위젯을 REST API/WebSocket으로 노출하는 FastAPI 서버로, 멀티 테넌시, mDNS 서비스 디스커버리, JWT 인증, 실시간 WebSocket을 지원합니다.

---

## 목차

1. [기술 스택](#1-기술-스택)
2. [시스템 요구사항](#2-시스템-요구사항)
3. [설치 및 실행](#3-설치-및-실행)
4. [설정](#4-설정)
5. [API 엔드포인트](#5-api-엔드포인트)
6. [지원 위젯](#6-지원-위젯)
7. [멀티 테넌시](#7-멀티-테넌시)
8. [테스트](#8-테스트)
9. [알려진 경고](#9-알려진-경고)
10. [라이선스](#10-라이선스)

---

## 1. 기술 스택

| 분류 | 기술 |
|------|------|
| Language | Python 3.12+ |
| Web Framework | FastAPI + uvicorn |
| Data Analysis | Orange3 3.40+, Orange3-Text 1.16+, PyQt6 |
| Database | SQLAlchemy + SQLite (기본) / PostgreSQL / MySQL |
| Observability | OpenTelemetry |
| Service Discovery | mDNS (zeroconf) |

---

## 2. 시스템 요구사항

- **Python** 3.12 이상
- **uv** (권장) 또는 pip
- **Conda / Miniforge** — Orange3 설치 시 권장
- **PyQt6** — Orange3 의존성

---

## 3. 설치 및 실행

### 3.1 uv 사용 (권장)

```bash
cd backend
uv venv --python 3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3.2 pip 사용

```bash
cd backend
python -m venv venv
source venv/bin/activate
uv sync
# 또는
pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3.3 Conda 사용 (Orange3 권장 방법)

Orange3는 conda-forge를 통한 설치를 공식 권장합니다.

```bash
conda create -n orange3-web python=3.12
conda activate orange3-web
conda install -c conda-forge orange3 orange3-text pyqt
pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3.4 Docker 사용

```bash
docker build -f ../docker/Dockerfile.backend -t orange3-web-backend .
docker run -p 8000:8000 orange3-web-backend
```

---

## 4. 설정

### 4.1 설정 파일

`orange3-web-backend.properties` 파일을 통해 서버를 설정합니다.

**탐색 순서** (앞에 있을수록 우선순위 높음):

1. `./orange3-web-backend.properties` (현재 디렉토리)
2. `/etc/orange3-web/orange3-web-backend.properties`
3. `~/.orange3-web/orange3-web-backend.properties`

**주요 설정 항목:**

```properties
server.host=0.0.0.0
server.port=8000
server.workers=4

storage.type=sqlite

# PostgreSQL 사용 시:
# database.url=postgresql+asyncpg://user:pass@localhost:5432/orange3

path.upload=/var/lib/orange3-web/uploads
path.database=/var/lib/orange3-web

mdns.enabled=true
mdns.service_name=orange3-backend

log.level=INFO

otel.enabled=false
# otel.endpoint=localhost:4317
```

### 4.2 환경변수

환경변수로도 설정 가능합니다. 단, **설정 파일보다 높은 우선순위**가 적용됩니다.

| 환경변수 | 설명 |
|----------|------|
| `HOST` | 서버 바인딩 호스트 |
| `PORT` | 서버 포트 |
| `WORKERS` | 워커 프로세스 수 |
| `DATABASE_URL` | 데이터베이스 연결 URL |
| `UPLOAD_DIR` | 파일 업로드 경로 |
| `LOG_LEVEL` | 로그 레벨 (DEBUG/INFO/WARNING/ERROR) |
| `OTEL_ENABLED` | OpenTelemetry 활성화 여부 |
| `OTEL_ENDPOINT` | OpenTelemetry Collector 엔드포인트 |

---

## 5. API 엔드포인트

### 인증 (Authentication)

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/api/v1/auth/register` | 사용자 등록 |
| `POST` | `/api/v1/auth/login` | 로그인 (JWT 발급) |

### 테넌트 (Tenants)

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/api/tenants` | 테넌트 목록 조회 |
| `POST` | `/api/tenants` | 테넌트 생성 |

### 워크플로우 (Workflows)

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/api/workflows` | 워크플로우 목록 조회 |
| `POST` | `/api/workflows` | 워크플로우 생성 |
| `GET` | `/api/workflows/{id}` | 워크플로우 상세 조회 |
| `PUT` | `/api/workflows/{id}` | 워크플로우 수정 |
| `DELETE` | `/api/workflows/{id}` | 워크플로우 삭제 |

### 위젯 (Widgets)

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/api/widgets` | 사용 가능한 위젯 목록 |
| `POST` | `/api/v1/widgets/{widget_name}/run` | 위젯 실행 |

### 데이터 (Data)

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/api/v1/data/upload` | 파일 업로드 |
| `GET` | `/api/v1/data/uploaded` | 업로드된 파일 목록 |

### WebSocket

| Protocol | Endpoint | 설명 |
|----------|----------|------|
| `WS` | `/ws/{workflow_id}` | 실시간 워크플로우 동기화 |

### 헬스체크 (Health)

| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/health` | 서버 상태 확인 |
| `GET` | `/metrics` | Prometheus 메트릭 |

---

## 6. 지원 위젯

### 데이터 (Data)
파일 업로드, 데이터셋, 열 선택, 행 선택, 데이터 샘플러, 데이터 정보, 그룹 바이, 피처 통계

### 모델 (Models)
선형 회귀, 로지스틱 회귀, SVM, KNN, 나이브 베이즈, 랜덤 포레스트, 신경망

### 비지도학습 (Unsupervised)
PCA, K-Means

### 시각화 (Visualization)
산점도, 히트맵, 박스 플롯, 분포도, 막대 그래프

### 평가 (Evaluation)
혼동 행렬, 테스트 및 점수

### 텍스트 마이닝 (Text Mining)
코퍼스, BoW, 텍스트 전처리, 워드 클라우드

---

## 7. 멀티 테넌시

모든 API 요청에 `X-Tenant-ID` 헤더로 테넌트를 지정합니다.

```http
X-Tenant-ID: my-tenant
```

- **기본값:** `default`
- 테넌트별 파일, 워크플로우가 완전히 분리됩니다
- 헤더 미지정 시 `default` 테넌트가 사용됩니다

---

## 8. 테스트

```bash
cd backend
pytest tests/ -v
```

---

## 9. 알려진 경고

| 경고 메시지 | 원인 | 영향 |
|-------------|------|------|
| `pkg_resources is deprecated` | Orange3 내부에서 사용하는 레거시 API | 동작에 영향 없음 |
| `Could not import optional dependency 'tensorflow'` | 선택적 의존성 미설치 | 해당 기능 미사용 시 무시 가능 |
| `Could not import optional dependency 'fairlearn'` | 선택적 의존성 미설치 | 해당 기능 미사용 시 무시 가능 |

> 선택적 의존성이 필요한 경우 `pip install tensorflow fairlearn` 으로 별도 설치하세요.

---

## 10. 라이선스

**GPL 3.0** — Orange3 의존성으로 인해 GPL 3.0 라이선스가 적용됩니다.

- 프로젝트: <https://github.com/jupilhwang/orange3-web-backend>
- Orange3 라이선스: <https://github.com/biolab/orange3/blob/master/LICENSE>
