# AGENTS.md

Python 백엔드 코드베이스에 대한 추가 정보입니다.

## 프로젝트 개요

Orange3 Web Backend는 Orange3 데이터 마이닝 도구의 웹 버전을 위한 FastAPI 기반 백엔드입니다.

### 주요 특징
- **FastAPI**: 비동기 웹 프레임워크
- **SQLite + aiosqlite**: 비동기 데이터베이스, 향후 운영환경에서는 MySQL, Postgres, Oracle 등을 사용할 수 있음
- **Multi-tenant**: X-Tenant-ID 헤더를 통한 멀티테넌시 지원
- **WebSocket**: 실시간 워크플로우 업데이트
- **Orange3 래핑**: 기존 Orange3 코드를 재사용

---

## 빌드
- uv venv 환경 사용
- uvicorn 사용

---

## 핵심 모듈

- python 버전: 3.12
- orange3 버전: 3.40
- orange3-text 버전: 1.16.3 이상
- 포매터: `black` (line length 88)
- import 정리: `isort`
- 린터: `flake8` / `ruff`
- 모든 public 함수/메서드는 타입힌트 필수
- 서비스 레이어에서는 비즈니스 규칙만, API 레이어에서는 HTTP 관련만 처리

### 1. main.py - 애플리케이션 진입점

```python
# FastAPI 앱 생성
app = FastAPI(
    title="Orange3 Web API",
    lifespan=lifespan  # 앱 수명주기 관리
)

# CORS 미들웨어
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

# API 라우터 (v1 prefix)
api_v1 = APIRouter(prefix="/api/v1")

# 위젯 라우터 등록
api_v1.include_router(scatter_plot_router)
api_v1.include_router(datasets_router)
# ... 기타 위젯 라우터들

app.include_router(api_v1)
```

### 2. core/config.py - 설정 관리

설정 우선순위:
1. `orange3-web-backend.properties` 파일
2. 환경 변수
3. 기본값

```python
# 설정 가져오기
from app.core.config import get_config, get_upload_dir

config = get_config()
upload_dir = get_upload_dir()
```

### 3. core/models.py - Pydantic 모델

주요 모델:
- `Workflow`, `WorkflowSummary` - 워크플로우
- `WorkflowNode`, `NodeCreate` - 노드
- `WorkflowLink`, `LinkCreate` - 연결
- `WidgetDescription`, `WidgetCategory` - 위젯 정의
- `Tenant` - 테넌트

### 4. widgets/ - 위젯 API

각 위젯은 독립적인 모듈로 구성:
```
widgets/
├── scatter_plot.py    # /api/v1/visualize/scatter-plot
├── distributions.py   # /api/v1/visualize/distributions
├── datasets.py        # /api/v1/datasets/*
├── knn.py            # /api/v1/models/knn/*
└── ...
```

---


## 디렉토리 구조

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱 진입점
│   ├── routes.py            # 워크플로우 & 위젯 레지스트리 라우트
│   ├── tasks.py             # 비동기 태스크 정의
│   ├── orange_adapter.py    # Orange3 래퍼
│   ├── websocket_manager.py # WebSocket 관리
│   ├── core/
│   │   ├── config.py        # 설정 관리
│   │   ├── database.py      # DB 연결
│   │   ├── db_models.py     # SQLAlchemy 모델 (Task Queue 포함)
│   │   ├── models.py        # Pydantic 모델
│   │   ├── tenant.py        # 테넌트 관리
│   │   ├── locks.py         # 비동기 락
│   │   ├── file_storage.py  # 파일 저장소
│   │   ├── task_queue.py    # DB 기반 Task Queue
│   │   ├── task_api.py      # Task Queue API 라우터
│   │   ├── data_utils.py    # 데이터 유틸리티
│   │   ├── text_mining_utils.py # 텍스트 마이닝 유틸리티
│   │   └── telemetry.py     # OpenTelemetry
│   └── widgets/
│       ├── __init__.py      # 위젯 라우터 export
│       ├── scatter_plot.py  # 시각화 위젯
│       ├── datasets.py      # 데이터셋 위젯
│       ├── knn.py           # ML 모델 위젯
│       ├── kmeans.py        # K-Means 클러스터링
│       ├── corpus.py        # 텍스트 코퍼스
│       └── ...              # 기타 위젯들
├── tests/
│   ├── conftest.py          # pytest fixtures
│   └── test_*.py            # 테스트 파일들
├── datasets_cache/          # 데이터셋 캐시
├── uploads/                 # 업로드 파일
├── requirements.txt
└── run.py                   # 실행 스크립트
```

## 위젯 추가 패턴

### 새 위젯 추가 단계

1. **위젯 모듈 생성** (`app/widgets/my_widget.py`)

```python
"""
My Widget API endpoints.
"""
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mywidget", tags=["MyWidget"])

# Orange3 가용성 체크
try:
    from Orange.data import Table
    ORANGE_AVAILABLE = True
except ImportError:
    ORANGE_AVAILABLE = False


# Request/Response 모델
class MyWidgetRequest(BaseModel):
    """위젯 요청 모델."""
    data_path: str
    param1: Optional[str] = None
    param2: int = 10


class MyWidgetResponse(BaseModel):
    """위젯 응답 모델."""
    result: dict
    status_bar: dict  # 상태바 정보


# API 엔드포인트
@router.post("")
async def process_my_widget(
    request: MyWidgetRequest,
    x_session_id: Optional[str] = Header(None)
):
    """위젯 처리 엔드포인트."""
    if not ORANGE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Orange3 not available")
    
    try:
        from .data_utils import load_data
        
        # 데이터 로드
        data = load_data(request.data_path, session_id=x_session_id)
        if data is None:
            raise HTTPException(status_code=400, detail="Failed to load data")
        
        # 처리 로직
        result = process_data(data, request)
        
        # 상태바 정보 (input/output 데이터 수)
        status_bar = {
            "input_count": len(data),
            "output_count": len(result.get("output", [])),
            "message": "처리 완료"
        }
        
        return {"result": result, "status_bar": status_bar}
        
    except Exception as e:
        logger.error(f"MyWidget error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

2. **라우터 등록** (`app/widgets/__init__.py`)

```python
from .my_widget import router as my_widget_router

__all__ = [
    # ... 기존 라우터들
    "my_widget_router",
]
```

3. **main.py에 추가**

```python
from .widgets import my_widget_router

api_v1.include_router(my_widget_router)
```

4. **테스트 작성** (`tests/test_my_widget.py`)

```python
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_my_widget_basic():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/mywidget",
            json={"data_path": "datasets/iris.tab"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "status_bar" in data
```

---

## API 구조

### 기본정보
- **BASE URL**: `/api/v1`
- **응답포멧**: JSON
- **버전**: URL기반 버저닝 (`/api/v1`)

### 주요 엔드포인트

| 경로 | 메서드 | 설명 |
|------|--------|------|
| `/health` | GET | 헬스 체크 |
| `/health/ready` | GET | 준비 상태 확인 |
| `/health/live` | GET | 생존 상태 확인 |
| `/api/v1/workflows` | GET/POST | 워크플로우 목록/생성 |
| `/api/v1/workflows/{id}` | GET/DELETE | 워크플로우 조회/삭제 |
| `/api/v1/workflows/{id}/nodes` | POST | 노드 추가 |
| `/api/v1/workflows/{id}/links` | POST | 연결 추가 |
| `/api/v1/widgets` | GET | 위젯 목록 |
| `/api/v1/widgets/categories` | GET | 위젯 카테고리 |
| `/api/v1/datasets` | GET | 데이터셋 목록 |
| `/api/v1/visualize/*` | POST | 시각화 위젯 |
| `/api/v1/models/*` | POST | ML 모델 위젯 |

### 공통헤더
- `Content-Type: application/json`
- `Authorization: Bearer <access_token>` (인증 필요 시)
- `X-Request-ID`: 요청 트래킹용 (없으면 미들웨어가 생성)

### 헤더

| 헤더 | 설명 |
|------|------|
| `X-Tenant-ID` | 테넌트 식별자 (멀티테넌시) |
| `X-Session-ID` | 세션 식별자 |

---

## 데이터 유틸리티

### data_utils.py 주요 함수

```python
from app.widgets.data_utils import load_data, save_temp_data

# 데이터 로드 (다양한 소스 지원)
data = load_data("datasets/iris.tab")           # 빌트인 데이터셋
data = load_data("uploads/file.csv")            # 업로드 파일
data = load_data("kmeans:node-123")             # K-Means 결과
data = load_data("session:abc123/data.tab")     # 세션별 데이터

# 임시 데이터 저장
temp_path = save_temp_data(data, "my_result", session_id)
```

---

## 설정 파일 형식

### orange3-web-backend.properties

```properties
# Database
database.url=sqlite+aiosqlite:///./orange_web.db
# database.url=postgresql+asyncpg://user:pass@localhost:5432/orange3

# Server
server.host=0.0.0.0
server.port=8000
server.workers=4

# Paths
path.upload=/var/lib/orange3-web/uploads
path.corpus=/var/lib/orange3-web/corpus
path.datasets_cache=/var/lib/orange3-web/datasets_cache

# Storage
storage.type=sqlite
storage.max_db_file_size=52428800

# Logging
log.level=INFO
log.database_echo=false
```

---

## 테스트 실행

```bash
# 가상환경 활성화
cd backend
source venv/bin/activate

# 전체 테스트
pytest tests/ -v

# 특정 테스트
pytest tests/test_scatter_plot.py -v

# 커버리지
pytest tests/ --cov=app --cov-report=html
```

---

## 코딩 컨벤션

### Python 스타일

```python
# PEP 8 준수
# 들여쓰기: 4 spaces
# 최대 줄 길이: 100자
# 타입 힌트 사용 권장

from typing import Optional, List, Dict, Any

def my_function(
    param1: str,
    param2: Optional[int] = None
) -> Dict[str, Any]:
    """
    함수 설명.
    
    Args:
        param1: 첫 번째 파라미터
        param2: 두 번째 파라미터 (선택)
        
    Returns:
        결과 딕셔너리
    """
    pass
```

### 로깅

```python
import logging

logger = logging.getLogger(__name__)

# 레벨별 사용
logger.debug("디버그 정보")
logger.info("일반 정보")
logger.warning("경고")
logger.error("에러")
logger.exception("예외 (스택 트레이스 포함)")
```

### 예외 처리

```python
from fastapi import HTTPException

# 클라이언트 오류 (4xx)
raise HTTPException(status_code=400, detail="잘못된 요청")
raise HTTPException(status_code=404, detail="리소스를 찾을 수 없음")

# 서버 오류 (5xx)
raise HTTPException(status_code=500, detail="내부 서버 오류")
raise HTTPException(status_code=501, detail="구현되지 않음")
```

### Pydantic 모델

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class MyModel(BaseModel):
    """모델 설명."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    count: int = 0
    items: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

---

## 상태바 패턴

모든 위젯은 `status_bar` 정보를 반환해야 합니다:

```python
status_bar = {
    "input_count": len(input_data),      # 입력 데이터 수
    "output_count": len(output_data),    # 출력 데이터 수
    "message": "처리 완료",               # 상태 메시지
    "warnings": [],                       # 경고 목록 (선택)
    "errors": []                          # 오류 목록 (선택)
}
```

---

## 의존성 버전

주요 패키지 버전 (requirements.txt 참조):

| 패키지 | 버전 |
|--------|------|
| fastapi | >=0.128.0,<1.0 |
| uvicorn | >=0.40.0,<1.0 |
| pydantic | >=2.12.0,<3.0 |
| sqlalchemy | >=2.0.45 |
| aiosqlite | >=0.22.0 |
| orange3 | >=3.40.0,<4.0 |
| orange3-text | >=1.16.3,<2.0 |
| numpy | >=2.0.0,<3.0 |
| scipy | >=1.14.0,<2.0 |
| scikit-learn | >=1.5.0,<2.0 |
| pandas | >=2.2.0,<3.0 |

---

## 서버 실행

```bash
# 개발 모드 (자동 리로드)
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 모드
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# run.py 사용
python run.py
```

---

## 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `DATABASE_URL` | 데이터베이스 URL | sqlite:///./orange_web.db |
| `UPLOAD_DIR` | 업로드 디렉토리 | ./uploads |
| `HOST` | 서버 호스트 | 0.0.0.0 |
| `PORT` | 서버 포트 | 8000 |
| `LOG_LEVEL` | 로그 레벨 | INFO |
| `OTEL_ENABLED` | OpenTelemetry 활성화 | true |
| `OTEL_ENDPOINT` | OTLP 엔드포인트 | - |
| `FRONTEND_URL` | 프론트엔드 URL (LB 등록용) | http://localhost:3000 |
| `BACKEND_URL` | 백엔드 URL (LB 등록용) | http://localhost:8000 |
| `LB_ENABLED` | 로드밸런서 등록 활성화 | true |

