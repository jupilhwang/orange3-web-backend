#!/usr/bin/env python3
"""
Orange3-Web Backend Server Runner

설정 우선순위: CLI 옵션 > 설정 파일 > 환경 변수 > 기본값

사용법:
    python run.py                           # 기본 설정으로 실행
    python run.py --port 9000               # 포트 지정
    python run.py --workers 4               # 워커 수 지정
    python run.py --reload                  # 개발 모드 (자동 리로드)
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def load_properties(filepath: str) -> dict:
    """properties 파일 로드"""
    config = {}
    if not os.path.exists(filepath):
        return config
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config


def find_config_file() -> str | None:
    """설정 파일 찾기"""
    search_paths = [
        './orange3-web-backend.properties',
        '/etc/orange3-web/orange3-web-backend.properties',
        os.path.expanduser('~/.orange3-web/orange3-web-backend.properties'),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Orange3-Web Backend Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                           # 기본 설정으로 실행
    python run.py --port 9000               # 포트 지정
    python run.py --workers 4 --port 8000   # 프로덕션 설정
    python run.py --reload                  # 개발 모드
        """
    )
    
    parser.add_argument('--host', type=str, help='바인딩 호스트 (기본: 0.0.0.0)')
    parser.add_argument('--port', type=int, help='서버 포트 (기본: 8000)')
    parser.add_argument('--workers', type=int, help='워커 수 (기본: 1)')
    parser.add_argument('--reload', action='store_true', help='자동 리로드 (개발 모드)')
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    
    args = parser.parse_args()
    
    # 설정 파일 로드
    config_file = args.config or find_config_file()
    config = {}
    if config_file:
        config = load_properties(config_file)
        print(f"[Config] Loaded from: {config_file}")
    
    # 설정 우선순위: CLI > 설정 파일 > 환경 변수 > 기본값
    host = args.host or config.get('server.host') or os.environ.get('HOST', '0.0.0.0')
    port = args.port or int(config.get('server.port', 0)) or int(os.environ.get('PORT', 8000))
    workers = args.workers or int(config.get('server.workers', 0)) or int(os.environ.get('WORKERS', 1))
    reload_mode = args.reload or config.get('server.reload', '').lower() == 'true'
    
    print(f"[Config] Host: {host}")
    print(f"[Config] Port: {port}")
    print(f"[Config] Workers: {workers}")
    print(f"[Config] Reload: {reload_mode}")
    print()
    
    # uvicorn 실행
    import uvicorn
    
    uvicorn_config = {
        'app': 'app.main:app',
        'host': host,
        'port': port,
        'reload': reload_mode,
    }
    
    # 워커 수 (reload 모드에서는 사용 불가)
    if workers > 1 and not reload_mode:
        uvicorn_config['workers'] = workers
    
    print(f"🍊 Starting Orange3-Web Backend on http://{host}:{port}")
    print()
    
    uvicorn.run(**uvicorn_config)


if __name__ == '__main__':
    main()
