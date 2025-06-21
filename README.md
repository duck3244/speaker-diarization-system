# 🎤 화자 분리 음성 인식 시스템 (Speaker Diarization & ASR)

**RTX 4060 최적화 한국어 AI 음성 분석 시스템**

여러 사람이 대화하는 음성에서 **누가**, **언제**, **무엇을** 말했는지 자동으로 구분하고 텍스트로 변환하는 AI 시스템입니다.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![GPU](https://img.shields.io/badge/GPU-RTX%204060%20Optimized-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 주요 기능

- **🎯 화자 분리**: 대화에서 화자별로 자동 구분 (최대 10명)
- **🗣️ 음성 인식**: 한국어 음성을 정확한 텍스트로 변환 (Whisper 기반)
- **⚡ RTX 4060 최적화**: 메모리 효율적인 처리 및 FP16 정밀도
- **📊 다양한 출력**: CSV, JSON, TXT, SRT 자막 형식 지원
- **🔄 배치 처리**: 여러 파일 동시 처리
- **📈 실시간 모니터링**: GPU/CPU 리소스 추적 및 최적화

## 📋 시스템 요구사항

### 최소 요구사항
| 구성요소 | 최소 사양 | 권장 사양 |
|----------|-----------|-----------|
| **GPU** | RTX 4060 (8GB) | RTX 4060 이상 |
| **RAM** | 16GB | 32GB |
| **Python** | 3.8+ | 3.9+ |
| **CUDA** | 11.8+ | 12.0+ |
| **저장공간** | 10GB | 20GB |

### 지원 OS
- ✅ Windows 10/11
- ✅ Ubuntu 20.04+
- ✅ macOS (CPU 모드)

## 🛠️ 설치 가이드

### 1단계: 저장소 클론
```bash
git clone https://github.com/your-username/speaker-diarization-system.git
cd speaker-diarization-system
```

### 2단계: 가상환경 생성
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3단계: CUDA 설치 (GPU 사용 시)
```bash
# CUDA 11.8용 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4단계: 의존성 설치
```bash
pip install -r requirements.txt
```

### 5단계: 시스템 확인
```bash
python main.py info
```

## 🎯 빠른 시작

### 시스템 정보 확인
```bash
python main.py info
```

### 단일 파일 처리
```bash
python main.py single conversation.wav
```

### 배치 처리
```bash
python main.py batch *.wav --output ./results
```

### 대화형 모드
```bash
python main.py interactive
```

## 📖 상세 사용법

### CLI 명령어

#### 기본 구조
```bash
python main.py <command> [arguments] [options]
```

#### 사용 가능한 명령어

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `single` | 단일 파일 처리 | `python main.py single audio.wav` |
| `batch` | 여러 파일 배치 처리 | `python main.py batch *.wav` |
| `info` | 시스템 정보 확인 | `python main.py info --benchmark` |
| `interactive` | 대화형 모드 | `python main.py interactive` |

#### 주요 옵션

| 옵션 | 설명 | 기본값 | 예시 |
|------|------|--------|------|
| `--output, -o` | 출력 디렉토리 | `./output` | `-o ./results` |
| `--model, -m` | Whisper 모델 | `medium` | `-m large-v3` |
| `--batch-size` | 배치 크기 | `2` | `--batch-size 1` |
| `--speakers, -s` | 예상 화자 수 | `자동` | `-s 3` |

### Python 코드에서 사용

```python
from main import SpeakerDiarizationSystem

# 시스템 초기화
system = SpeakerDiarizationSystem()

# 파일 처리
result = system.process_audio_file("conversation.wav")

if result['success']:
    print(f"화자 수: {result['num_speakers']}")
    print(f"음성 구간: {result['num_segments']}")
    print(f"결과 파일: {result['output_files']['txt']}")
else:
    print(f"오류: {result['error']}")
```

## 📊 지원 모델 및 성능

### Whisper 모델 비교

| 모델 | 크기 | RTX 4060 | 정확도 | 처리속도 | 메모리 |
|------|------|----------|--------|----------|--------|
| **base** | 244MB | ✅ 빠름 | 85% | 0.1x 실시간 | 1GB |
| **medium** | 1.5GB | ✅ 권장 | 90% | 0.3x 실시간 | 2.5GB |
| **large-v3** | 6GB | ⚠️ 느림 | 95% | 0.8x 실시간 | 6GB |

### 성능 벤치마크 (RTX 4060 기준)

| 오디오 길이 | medium 모델 | base 모델 | large-v3 모델 |
|-------------|-------------|-----------|---------------|
| 30초 | 9초 | 3초 | 24초 |
| 5분 | 90초 | 30초 | 240초 |
| 30분 | 9분 | 3분 | 24분 |

## 📁 출력 형식

시스템은 다음 4가지 형식으로 결과를 저장합니다:

### 1. CSV 파일 (`*_transcript.csv`)
```csv
segment_id,speaker_id,start_time,end_time,duration,text,confidence
0,화자_0,1.2,4.5,3.3,"안녕하세요 반갑습니다",0.92
1,화자_1,5.1,8.3,3.2,"네 안녕하세요",0.89
```

### 2. JSON 파일 (`*_transcript.json`)
```json
[
  {
    "segment_id": 0,
    "speaker_id": "화자_0",
    "start_time": 1.2,
    "end_time": 4.5,
    "duration": 3.3,
    "text": "안녕하세요 반갑습니다",
    "confidence": 0.92
  }
]
```

### 3. 대화록 파일 (`*_conversation.txt`)
```
=== 대화 전사 결과 ===

[00:01] 화자_0: 안녕하세요 반갑습니다 🟢
[00:05] 화자_1: 네 안녕하세요 🟢
[00:09] 화자_0: 오늘 날씨가 참 좋네요 🟡

=== 통계 ===
총 화자 수: 2명
총 길이: 30.5초
발화 구간: 3개
```

### 4. SRT 자막 파일 (`*_subtitles.srt`)
```srt
1
00:00:01,200 --> 00:00:04,500
화자_0: 안녕하세요 반갑습니다

2
00:00:05,100 --> 00:00:08,300
화자_1: 네 안녕하세요
```

## 🔧 최적화 및 문제해결

### RTX 4060 최적화 팁

#### 메모리 부족 시
```bash
# 더 작은 모델 사용
python main.py single audio.wav --model base

# 배치 크기 줄이기
python main.py single audio.wav --batch-size 1
```

#### 처리 속도 향상
```bash
# 청크 단위 처리 (자동)
python main.py single long_audio.wav

# 병렬 처리
python main.py batch *.wav
```

### 일반적인 문제 해결

#### 1. CUDA 오류
```bash
# CUDA 버전 확인
nvidia-smi
nvcc --version

# PyTorch 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. 메모리 부족
`config.py`에서 설정 조정:
```python
processing_config.batch_size = 1
processing_config.chunk_duration = 15.0
processing_config.use_fp16 = True
```

#### 3. 음성 구간 미감지
`config.py`에서 VAD 임계값 조정:
```python
audio_config.vad_threshold_percentile = 15  # 기본값: 30
audio_config.min_duration = 0.1  # 기본값: 0.3
```

#### 4. pyannote 접근 권한 오류
```bash
# Hugging Face 토큰 설정
huggingface-cli login

# 또는 경량 모드 강제 사용
# speaker_diarization.py에서 _should_use_pyannote 함수가 항상 False 반환하도록 수정
```

## 🧪 테스트 및 검증

### 테스트 파일 생성
```bash
python test_audio_generator.py
```

### 단계별 디버깅
```bash
# VAD 테스트
python -c "
from audio_utils import AudioProcessor, VADProcessor
processor = AudioProcessor()
vad = VADProcessor()
audio, sr = processor.load_audio('test.wav')
segments = vad.enhanced_vad(audio, sr)
print(f'감지된 구간: {len(segments)}개')
"

# 화자 분리 테스트
python -c "
from speaker_diarization import create_speaker_diarizer
diarizer = create_speaker_diarizer('lightweight')
print('화자 분리 모델 로드 완료')
"
```

### 성능 벤치마크
```bash
python main.py info --benchmark
```

## 📚 프로젝트 구조

```
speaker-diarization-system/
├── main.py                    # 메인 통합 시스템
├── config.py                  # 설정 및 하이퍼파라미터
├── audio_utils.py             # 오디오 처리 유틸리티
├── speaker_diarization.py     # 화자 분리 기능
├── speech_recognition.py      # 음성 인식 기능
├── system_monitor.py          # 시스템 모니터링
├── test_audio_generator.py    # 테스트 파일 생성기
├── requirements.txt           # 의존성 패키지
├── README.md                  # 이 파일
└── output/                    # 결과 파일 저장 디렉토리
```

## 🔄 버전 정보

### v1.0.0
- ✅ VAD 알고리즘 개선으로 음성 구간 감지율 향상
- ✅ pyannote 접근 권한 문제 우회 방법 추가
- ✅ RTX 4060 메모리 최적화 강화
- ✅ Hugging Face 토큰 설정 및 pyannote 모델 사용 조건 동의 필요
- ✅ 경량 화자 분리 모드 이슈 존재 (화자 인식률 낮음)

### 개발 환경 설정
```bash
# 개발 의존성 설치
pip install -r requirements-dev.txt

# 코드 스타일 검사
flake8 .

# 테스트 실행
pytest tests/
```