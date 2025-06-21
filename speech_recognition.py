# speech_recognition.py
"""
음성 인식 (Speech Recognition) 기능
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

from config import model_config, processing_config, system_config


class BaseASR:
    """음성 인식 기본 클래스"""

    def __init__(self, config=None):
        self.config = config or model_config
        self.device = system_config.device

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        """음성을 텍스트로 변환 (추상 메서드)"""
        raise NotImplementedError

    def transcribe_batch(self, audio_list: List[np.ndarray], sr: int) -> List[str]:
        """배치 음성 인식 (추상 메서드)"""
        raise NotImplementedError


class WhisperASR(BaseASR):
    """Whisper 기반 음성 인식"""

    def __init__(self, model_name: str = None, config=None):
        super().__init__(config)

        self.model_name = model_name or self.config.whisper_model_name
        self.processor = None
        self.model = None
        self.processing_config = processing_config

        self._load_model()

    def _load_model(self):
        """Whisper 모델 로드"""
        print(f"Whisper 모델 로딩: {self.model_name}")

        try:
            # 프로세서 로드
            self.processor = WhisperProcessor.from_pretrained(self.model_name)

            # 모델 로드 (메모리 최적화)
            if self.processing_config.use_fp16 and self.device == "cuda":
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
            else:
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True
                )

                if self.device == "cuda":
                    self.model = self.model.to(self.device)

            # Gradient checkpointing 활성화 (메모리 절약)
            if (self.processing_config.gradient_checkpointing and
                    hasattr(self.model, 'gradient_checkpointing_enable')):
                self.model.gradient_checkpointing_enable()

            # 한국어 설정
            self.processor.tokenizer.set_prefix_tokens(
                language=self.config.language,
                task=self.config.task
            )

            print("✅ Whisper 모델 로드 완료")
            self._print_model_info()

        except Exception as e:
            print(f"❌ Whisper 모델 로드 실패: {e}")
            raise

    def _print_model_info(self):
        """모델 정보 출력"""
        if self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
            print(f"GPU 메모리 사용량: {memory_allocated:.2f}GB")

        print(f"언어: {self.config.language}")
        print(f"작업: {self.config.task}")
        print(f"FP16 사용: {self.processing_config.use_fp16}")

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        """단일 오디오 음성 인식"""

        if self.model is None or self.processor is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        try:
            # 오디오 전처리
            input_features = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features

            # GPU로 이동 및 타입 맞추기
            if self.device == "cuda":
                input_features = input_features.to(self.device)
                # FP16 모델인 경우 입력도 FP16으로 변환
                if self.processing_config.use_fp16:
                    input_features = input_features.half()

            # 추론
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    language=self.config.language,
                    task=self.config.task,
                    max_length=self.processing_config.max_length,
                    forced_decoder_ids=None  # 충돌 방지
                )

            # 디코딩
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            return transcription.strip()

        except Exception as e:
            print(f"음성 인식 오류: {e}")
            return ""

    def transcribe_batch(self, audio_list: List[np.ndarray], sr: int) -> List[str]:
        """배치 음성 인식"""

        if not audio_list:
            return []

        # 배치 크기 제한
        batch_size = self.processing_config.batch_size
        results = []

        for i in range(0, len(audio_list), batch_size):
            batch = audio_list[i:i + batch_size]
            batch_results = self._process_batch(batch, sr)
            results.extend(batch_results)

            # 메모리 정리
            if self.device == "cuda" and i % (batch_size * 3) == 0:
                torch.cuda.empty_cache()

        return results

    def _process_batch(self, audio_batch: List[np.ndarray], sr: int) -> List[str]:
        """배치 처리"""

        if not audio_batch:
            return []

        try:
            # 길이 맞추기 (패딩)
            max_length = max(len(audio) for audio in audio_batch)
            padded_batch = []

            for audio in audio_batch:
                if len(audio) < max_length:
                    padded_audio = np.pad(audio, (0, max_length - len(audio)))
                else:
                    padded_audio = audio
                padded_batch.append(padded_audio)

            # 배치 처리
            input_features = self.processor(
                padded_batch,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features

            # GPU로 이동 및 타입 맞추기
            if self.device == "cuda":
                input_features = input_features.to(self.device)
                # FP16 모델인 경우 입력도 FP16으로 변환
                if self.processing_config.use_fp16:
                    input_features = input_features.half()

            # 배치 추론
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    language=self.config.language,
                    task=self.config.task,
                    max_length=self.processing_config.max_length,
                    forced_decoder_ids=None  # 충돌 방지
                )

            # 배치 디코딩
            transcriptions = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )

            return [text.strip() for text in transcriptions]

        except Exception as e:
            print(f"배치 처리 오류: {e}")
            # 개별 처리로 폴백
            return [self.transcribe(audio, sr) for audio in audio_batch]

    def transcribe_with_timestamps(self, audio: np.ndarray, sr: int) -> Dict:
        """타임스탬프와 함께 음성 인식"""

        try:
            input_features = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            ).input_features

            if self.device == "cuda":
                input_features = input_features.to(self.device)

            # 타임스탬프 포함 생성
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    language=self.config.language,
                    task=self.config.task,
                    return_timestamps=True,
                    max_length=self.processing_config.max_length
                )

            # 디코딩 (타임스탬프 포함)
            result = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
                decode_with_timestamps=True
            )[0]

            return {
                'text': result,
                'has_timestamps': True
            }

        except Exception as e:
            print(f"타임스탬프 인식 실패: {e}")
            # 일반 인식으로 폴백
            text = self.transcribe(audio, sr)
            return {
                'text': text,
                'has_timestamps': False
            }

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""

        info = {
            'model_name': self.model_name,
            'language': self.config.language,
            'task': self.config.task,
            'device': self.device,
            'fp16': self.processing_config.use_fp16
        }

        if self.device == "cuda":
            info['gpu_memory'] = torch.cuda.memory_allocated(0) / 1024 ** 3

        return info


class HuggingFacePipelineASR(BaseASR):
    """Hugging Face Pipeline 기반 음성 인식"""

    def __init__(self, model_name: str = None, config=None):
        super().__init__(config)

        self.model_name = model_name or self.config.whisper_model_name
        self.pipe = None

        self._load_pipeline()

    def _load_pipeline(self):
        """파이프라인 로드"""
        print(f"Pipeline 모델 로딩: {self.model_name}")

        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device=0 if self.device == "cuda" else -1,
                model_kwargs={
                    "language": self.config.language,
                    "task": self.config.task
                }
            )

            print("✅ Pipeline 로드 완료")

        except Exception as e:
            print(f"❌ Pipeline 로드 실패: {e}")
            raise

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        """파이프라인 기반 음성 인식"""

        if self.pipe is None:
            raise RuntimeError("파이프라인이 로드되지 않았습니다.")

        try:
            result = self.pipe(
                audio,
                sampling_rate=sr,
                return_timestamps=False
            )

            return result['text'].strip()

        except Exception as e:
            print(f"파이프라인 인식 오류: {e}")
            return ""

    def transcribe_batch(self, audio_list: List[np.ndarray], sr: int) -> List[str]:
        """파이프라인 배치 처리"""

        # 파이프라인은 기본적으로 배치를 지원하지 않으므로 순차 처리
        results = []
        for audio in audio_list:
            result = self.transcribe(audio, sr)
            results.append(result)

        return results


class OptimizedASR:
    """RTX 4060 최적화 음성 인식"""

    def __init__(self, config=None):
        self.config = config or model_config
        self.processing_config = processing_config

        # GPU 메모리에 따라 모델 선택
        self.asr_model = self._select_optimal_model()

    def _select_optimal_model(self) -> BaseASR:
        """최적 모델 선택"""

        if system_config.device == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

                print(f"GPU: {gpu_name}, 메모리: {total_memory:.1f}GB")

                if "4060" in gpu_name or total_memory < 10:
                    # RTX 4060용 최적화
                    print("RTX 4060 최적화 모드")
                    return WhisperASR(model_name="openai/whisper-medium")
                else:
                    # 고성능 GPU
                    return WhisperASR(model_name=self.config.whisper_model_name)

            except Exception as e:
                print(f"GPU 정보 확인 실패: {e}")
                return WhisperASR(model_name="openai/whisper-base")
        else:
            # CPU 모드
            print("CPU 모드 - 경량 모델 사용")
            return WhisperASR(model_name="openai/whisper-base")

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        """최적화된 음성 인식"""
        return self.asr_model.transcribe(audio, sr)

    def transcribe_batch(self, audio_list: List[np.ndarray], sr: int) -> List[str]:
        """최적화된 배치 인식"""
        return self.asr_model.transcribe_batch(audio_list, sr)

    def transcribe_segments(self, audio: np.ndarray,
                            segments: List[Tuple[float, float]],
                            sr: int) -> List[Dict]:
        """세그먼트별 음성 인식"""

        results = []
        batch_audio = []
        batch_info = []

        # 배치 준비
        for i, (start, end) in enumerate(segments):
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) > sr * 0.3:  # 최소 0.3초
                batch_audio.append(segment_audio)
                batch_info.append({
                    'segment_id': i,
                    'start': start,
                    'end': end,
                    'duration': end - start
                })

        if not batch_audio:
            return results

        # 배치 음성 인식
        transcriptions = self.transcribe_batch(batch_audio, sr)

        # 결과 조합
        for transcription, info in zip(transcriptions, batch_info):
            if transcription.strip():
                results.append({
                    'segment_id': info['segment_id'],
                    'start_time': round(info['start'], 2),
                    'end_time': round(info['end'], 2),
                    'duration': round(info['duration'], 2),
                    'text': transcription.strip()
                })

        return results


class ASRPostProcessor:
    """음성 인식 결과 후처리"""

    def __init__(self):
        pass

    def clean_transcription(self, text: str) -> str:
        """전사 결과 정리"""

        import re

        if not text:
            return ""

        # 1. 앞뒤 공백 제거
        text = text.strip()

        # 2. 연속 공백 정리
        text = re.sub(r'\s+', ' ', text)

        # 3. 불필요한 반복 제거 (예: "어어어" → "어")
        text = re.sub(r'(.)\1{2,}', r'\1', text)

        # 4. 한국어 맞춤법 기본 정리
        text = self._fix_korean_spacing(text)

        # 5. 숫자 정리 (필요시)
        text = self._normalize_numbers(text)

        return text

    def _fix_korean_spacing(self, text: str) -> str:
        """한국어 띄어쓰기 기본 정리"""

        import re

        # 조사 앞 띄어쓰기 제거
        particles = ['은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로', '의', '와', '과', '도', '만', '까지', '부터', '께서']
        for particle in particles:
            text = re.sub(f' +{particle}', particle, text)

        # 어미 앞 띄어쓰기 제거
        endings = ['입니다', '습니다', '였습니다', '했습니다', '되었습니다', '였어요', '했어요']
        for ending in endings:
            text = re.sub(f' +{ending}', ending, text)

        return text

    def _normalize_numbers(self, text: str) -> str:
        """숫자 표기 정규화"""

        import re

        # 한국어 숫자를 아라비아 숫자로 변환 (기본적인 것만)
        korean_numbers = {
            '영': '0', '공': '0',
            '일': '1', '하나': '1',
            '이': '2', '둘': '2',
            '삼': '3', '셋': '3',
            '사': '4', '넷': '4',
            '오': '5', '다섯': '5',
            '육': '6', '여섯': '6',
            '칠': '7', '일곱': '7',
            '팔': '8', '여덟': '8',
            '구': '9', '아홉': '9'
        }

        for korean, arabic in korean_numbers.items():
            # 단독으로 나타나는 경우만 변환
            text = re.sub(f'\\b{korean}\\b', arabic, text)

        return text

    def remove_filler_words(self, text: str) -> str:
        """채움말(음, 어, 그) 제거"""

        import re

        # 한국어 채움말 패턴
        filler_patterns = [
            r'\b음+\b',
            r'\b어+\b',
            r'\b그+\b',
            r'\b아+\b',
            r'\b에+\b(?=\s)',  # 단독 "에"
            r'\b뭐+\b'
        ]

        for pattern in filler_patterns:
            text = re.sub(pattern, '', text)

        # 정리된 공백 재정리
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def merge_short_segments(self, segments: List[Dict],
                             min_duration: float = 1.0) -> List[Dict]:
        """짧은 세그먼트 병합"""

        if not segments:
            return segments

        merged = []
        current_segment = segments[0].copy()

        for next_segment in segments[1:]:
            current_duration = current_segment['end_time'] - current_segment['start_time']

            # 현재 세그먼트가 너무 짧고, 다음 세그먼트와 가까우면 병합
            if (current_duration < min_duration and
                    next_segment['start_time'] - current_segment['end_time'] < 2.0):

                # 텍스트 병합
                current_segment['text'] += ' ' + next_segment['text']
                current_segment['end_time'] = next_segment['end_time']
                current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
            else:
                # 현재 세그먼트 완료
                merged.append(current_segment)
                current_segment = next_segment.copy()

        # 마지막 세그먼트 추가
        merged.append(current_segment)

        return merged

    def calculate_confidence_score(self, text: str, audio_quality: Dict) -> float:
        """신뢰도 점수 계산 (추정)"""

        # 기본 점수
        confidence = 0.8

        # 텍스트 길이 기반 조정
        text_length = len(text.strip())
        if text_length == 0:
            return 0.0
        elif text_length < 3:
            confidence *= 0.6
        elif text_length > 50:
            confidence *= 1.1

        # 오디오 품질 기반 조정
        if audio_quality:
            if audio_quality.get('snr', 0) > 20:
                confidence *= 1.1
            elif audio_quality.get('snr', 0) < 10:
                confidence *= 0.8

            if not audio_quality.get('clipping_ok', True):
                confidence *= 0.9

        # 한국어 비율 확인
        korean_chars = len([c for c in text if '가' <= c <= '힣'])
        total_chars = len([c for c in text if c.isalnum() or '가' <= c <= '힣'])

        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            if korean_ratio > 0.8:
                confidence *= 1.05
            elif korean_ratio < 0.5:
                confidence *= 0.85

        return min(confidence, 1.0)


# 팩토리 함수
def create_asr_model(model_type: str = "auto", model_name: str = None, config=None) -> BaseASR:
    """ASR 모델 생성"""

    if model_type == "auto":
        return OptimizedASR(config)
    elif model_type == "whisper":
        return WhisperASR(model_name, config)
    elif model_type == "pipeline":
        return HuggingFacePipelineASR(model_name, config)
    else:
        raise ValueError(f"알 수 없는 모델 타입: {model_type}")


# 유틸리티 함수들
def benchmark_asr_models(test_audio: np.ndarray, sr: int) -> Dict:
    """ASR 모델들 성능 벤치마크"""

    import time

    models_to_test = [
        ("whisper-base", "openai/whisper-base"),
        ("whisper-medium", "openai/whisper-medium"),
        ("whisper-large-v3", "openai/whisper-large-v3")
    ]

    results = {}

    for model_name, model_path in models_to_test:
        print(f"\n=== {model_name} 테스트 ===")

        try:
            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 모델 로드 시간 측정
            start_time = time.time()
            asr = WhisperASR(model_path)
            load_time = time.time() - start_time

            # 추론 시간 측정
            start_time = time.time()
            transcription = asr.transcribe(test_audio, sr)
            inference_time = time.time() - start_time

            # 메모리 사용량
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(0) / 1024 ** 3
            else:
                memory_used = 0

            results[model_name] = {
                'load_time': load_time,
                'inference_time': inference_time,
                'memory_used_gb': memory_used,
                'transcription': transcription,
                'success': True
            }

            print(f"✅ 로드 시간: {load_time:.2f}초")
            print(f"✅ 추론 시간: {inference_time:.2f}초")
            print(f"✅ 메모리 사용: {memory_used:.2f}GB")
            print(f"✅ 결과: {transcription[:50]}...")

            # 모델 언로드
            del asr

        except Exception as e:
            print(f"❌ 실패: {e}")
            results[model_name] = {
                'success': False,
                'error': str(e)
            }

        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def estimate_processing_time(audio_duration: float, model_name: str) -> Dict:
    """처리 시간 추정"""

    # 모델별 처리 속도 (RTX 4060 기준, 배수)
    speed_multipliers = {
        "openai/whisper-base": 0.1,  # 실시간의 10%
        "openai/whisper-medium": 0.3,  # 실시간의 30%
        "openai/whisper-large-v3": 0.8  # 실시간의 80%
    }

    multiplier = speed_multipliers.get(model_name, 0.5)
    estimated_time = audio_duration * multiplier

    return {
        'audio_duration': audio_duration,
        'estimated_processing_time': estimated_time,
        'speed_ratio': multiplier,
        'model': model_name
    }


if __name__ == "__main__":
    # 테스트 코드
    print("음성 인식 모듈 로드됨")

    # 최적화된 ASR 생성
    try:
        asr = create_asr_model("auto")
        print(f"✅ ASR 모델 생성 완료: {type(asr).__name__}")

        # 모델 정보 출력
        if hasattr(asr, 'asr_model'):
            info = asr.asr_model.get_model_info()
            print(f"모델 정보: {info}")

    except Exception as e:
        print(f"❌ ASR 모델 생성 실패: {e}")


# 팩토리 함수
def create_asr_model(model_type: str = "auto", model_name: str = None, config=None) -> BaseASR:
    """ASR 모델 생성"""

    if model_type == "auto":
        return OptimizedASR(config)
    elif model_type == "whisper":
        return WhisperASR(model_name, config)
    elif model_type == "pipeline":
        return HuggingFacePipelineASR(model_name, config)
    else:
        raise ValueError(f"알 수 없는 모델 타입: {model_type}")


# 유틸리티 함수들
def benchmark_asr_models(test_audio: np.ndarray, sr: int) -> Dict:
    """ASR 모델들 성능 벤치마크"""

    import time

    models_to_test = [
        ("whisper-base", "openai/whisper-base"),
        ("whisper-medium", "openai/whisper-medium"),
        ("whisper-large-v3", "openai/whisper-large-v3")
    ]

    results = {}

    for model_name, model_path in models_to_test:
        print(f"\n=== {model_name} 테스트 ===")

        try:
            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 모델 로드 시간 측정
            start_time = time.time()
            asr = WhisperASR(model_path)
            load_time = time.time() - start_time

            # 추론 시간 측정
            start_time = time.time()
            transcription = asr.transcribe(test_audio, sr)
            inference_time = time.time() - start_time

            # 메모리 사용량
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(0) / 1024 ** 3
            else:
                memory_used = 0

            results[model_name] = {
                'load_time': load_time,
                'inference_time': inference_time,
                'memory_used_gb': memory_used,
                'transcription': transcription,
                'success': True
            }

            print(f"✅ 로드 시간: {load_time:.2f}초")
            print(f"✅ 추론 시간: {inference_time:.2f}초")
            print(f"✅ 메모리 사용: {memory_used:.2f}GB")
            print(f"✅ 결과: {transcription[:50]}...")

            # 모델 언로드
            del asr

        except Exception as e:
            print(f"❌ 실패: {e}")
            results[model_name] = {
                'success': False,
                'error': str(e)
            }

        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def estimate_processing_time(audio_duration: float, model_name: str) -> Dict:
    """처리 시간 추정"""

    # 모델별 처리 속도 (RTX 4060 기준, 배수)
    speed_multipliers = {
        "openai/whisper-base": 0.1,  # 실시간의 10%
        "openai/whisper-medium": 0.3,  # 실시간의 30%
        "openai/whisper-large-v3": 0.8  # 실시간의 80%
    }

    multiplier = speed_multipliers.get(model_name, 0.5)
    estimated_time = audio_duration * multiplier

    return {
        'audio_duration': audio_duration,
        'estimated_processing_time': estimated_time,
        'speed_ratio': multiplier,
        'model': model_name
    }


if __name__ == "__main__":
    # 테스트 코드
    print("음성 인식 모듈 로드됨")

    # 최적화된 ASR 생성
    try:
        asr = create_asr_model("auto")
        print(f"✅ ASR 모델 생성 완료: {type(asr).__name__}")

        # 모델 정보 출력
        if hasattr(asr, 'asr_model'):
            info = asr.asr_model.get_model_info()
            print(f"모델 정보: {info}")

    except Exception as e:
        print(f"❌ ASR 모델 생성 실패: {e}")