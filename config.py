# config.py
"""
화자 분리 시스템 설정 파일
"""

import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """모델 관련 설정"""
    
    # Whisper 모델 설정
    whisper_model_name: str = "openai/whisper-medium"  # RTX 4060 최적화
    whisper_large_model: str = "openai/whisper-large-v3"
    whisper_base_model: str = "openai/whisper-base"
    
    # 화자 분리 모델 설정
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    
    # 언어 설정
    language: str = "korean"
    task: str = "transcribe"

@dataclass
class AudioConfig:
    """오디오 처리 설정"""
    
    # 기본 오디오 설정
    sample_rate: int = 16000
    frame_length: int = 2048
    hop_length: int = 512
    
    # VAD 설정
    min_duration: float = 0.1  # 0.3 → 0.1로 변경
    max_duration: float = 30.0  # 최대 음성 구간 (초)
    vad_threshold_percentile: int = 15  # 30 → 15로 변경
    spectral_threshold_percentile: int = 40
    
    # 품질 기준
    min_snr: float = 15.0  # 최소 SNR (dB)
    max_clipping_ratio: float = 0.001  # 최대 클리핑 비율
    min_rms: float = 0.005  # 0.01 → 0.005로 변경

@dataclass
class ProcessingConfig:
    """처리 관련 설정"""
    
    # 메모리 최적화
    chunk_duration: float = 30.0  # 청크 단위 (초)
    batch_size: int = 2  # 배치 크기 (RTX 4060 최적화)
    max_length: int = 100  # Whisper 최대 토큰 길이
    
    # 화자 분리 설정
    max_speakers: int = 6              # 10 → 6으로 현실적 조정
    min_cluster_size: int = 5          # 15 → 5로 완화
    clustering_threshold: float = 0.5  # 0.7 → 0.5로 완화
    
    # GPU 설정
    use_fp16: bool = True  # FP16 사용으로 메모리 절약
    gradient_checkpointing: bool = True
    memory_cleanup_interval: int = 3  # 몇 청크마다 메모리 정리

@dataclass
class SystemConfig:
    """시스템 설정"""
    
    # 하드웨어 감지
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_memory_threshold: float = 10.0  # GB, RTX 4060 감지용
    
    # 로깅 설정
    log_level: str = "INFO"
    enable_memory_monitoring: bool = True
    
    # 출력 설정
    output_format: str = "csv"  # csv, json, txt
    save_intermediate_results: bool = False

# GPU별 최적화 설정
GPU_CONFIGS = {
    "RTX_4060": {
        "max_batch_size": 2,
        "recommended_model": "openai/whisper-medium",
        "chunk_duration": 30.0,
        "use_fp16": True,
        "memory_limit_gb": 7.0
    },
    "RTX_3090": {
        "max_batch_size": 8,
        "recommended_model": "openai/whisper-large-v3", 
        "chunk_duration": 60.0,
        "use_fp16": True,
        "memory_limit_gb": 20.0
    },
    "A100": {
        "max_batch_size": 16,
        "recommended_model": "openai/whisper-large-v3",
        "chunk_duration": 120.0,
        "use_fp16": False,
        "memory_limit_gb": 35.0
    }
}

# 언어별 설정
LANGUAGE_CONFIGS = {
    "korean": {
        "whisper_language": "korean",
        "min_korean_ratio": 0.7,
        "special_char_max_ratio": 0.1
    },
    "english": {
        "whisper_language": "english", 
        "min_english_ratio": 0.8,
        "special_char_max_ratio": 0.05
    }
}

# 기본 설정 인스턴스
model_config = ModelConfig()
audio_config = AudioConfig()
processing_config = ProcessingConfig()
system_config = SystemConfig()

def get_gpu_config():
    """현재 GPU에 맞는 설정 반환"""
    if not torch.cuda.is_available():
        return None
    
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if "4060" in gpu_name:
        return GPU_CONFIGS["RTX_4060"]
    elif "3090" in gpu_name:
        return GPU_CONFIGS["RTX_3090"] 
    elif "A100" in gpu_name:
        return GPU_CONFIGS["A100"]
    else:
        # 메모리 기반 추정
        if total_memory < 10:
            return GPU_CONFIGS["RTX_4060"]
        elif total_memory < 25:
            return GPU_CONFIGS["RTX_3090"]
        else:
            return GPU_CONFIGS["A100"]

def update_config_for_gpu():
    """GPU에 맞게 설정 자동 조정"""
    gpu_config = get_gpu_config()
    
    if gpu_config:
        processing_config.batch_size = gpu_config["max_batch_size"]
        processing_config.chunk_duration = gpu_config["chunk_duration"]
        processing_config.use_fp16 = gpu_config["use_fp16"]
        model_config.whisper_model_name = gpu_config["recommended_model"]
        
        print(f"GPU 최적화 설정 적용: {gpu_config}")
    else:
        print("CPU 모드로 설정")
        processing_config.batch_size = 1
        processing_config.use_fp16 = False

# 초기화 시 자동 설정
if __name__ == "__main__":
    update_config_for_gpu()
