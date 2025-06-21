# system_monitor.py
"""
시스템 리소스 모니터링 및 최적화
"""

import torch
import psutil
import time
import gc
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import warnings

warnings.filterwarnings("ignore")


@dataclass
class SystemStats:
    """시스템 통계 정보"""
    timestamp: datetime
    gpu_memory_allocated: float  # GB
    gpu_memory_reserved: float  # GB
    gpu_memory_total: float  # GB
    gpu_utilization: float  # %
    cpu_usage: float  # %
    ram_usage: float  # %
    ram_available: float  # GB


class SystemMonitor:
    """시스템 리소스 모니터링"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stats_history: List[SystemStats] = []
        self.monitoring = False

    def get_current_stats(self) -> SystemStats:
        """현재 시스템 상태 조회"""

        timestamp = datetime.now()

        # GPU 정보
        if self.device == "cuda":
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
                gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

                # GPU 사용률 (근사치)
                gpu_utilization = (gpu_memory_allocated / gpu_memory_total) * 100

            except Exception:
                gpu_memory_allocated = 0
                gpu_memory_reserved = 0
                gpu_memory_total = 0
                gpu_utilization = 0
        else:
            gpu_memory_allocated = 0
            gpu_memory_reserved = 0
            gpu_memory_total = 0
            gpu_utilization = 0

        # CPU 및 RAM 정보
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        ram_usage = memory_info.percent
        ram_available = memory_info.available / 1024 ** 3

        return SystemStats(
            timestamp=timestamp,
            gpu_memory_allocated=gpu_memory_allocated,
            gpu_memory_reserved=gpu_memory_reserved,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            cpu_usage=cpu_usage,
            ram_usage=ram_usage,
            ram_available=ram_available
        )

    def print_current_stats(self):
        """현재 상태 출력"""

        stats = self.get_current_stats()

        print(f"\n=== 시스템 상태 ({stats.timestamp.strftime('%H:%M:%S')}) ===")

        if self.device == "cuda":
            print(f"GPU 메모리:")
            print(f"  할당됨: {stats.gpu_memory_allocated:.2f}GB")
            print(f"  예약됨: {stats.gpu_memory_reserved:.2f}GB")
            print(f"  총 용량: {stats.gpu_memory_total:.2f}GB")
            print(f"  사용률: {stats.gpu_utilization:.1f}%")

            # 메모리 여유도 경고
            if stats.gpu_utilization > 90:
                print("⚠️ GPU 메모리 부족 위험")
            elif stats.gpu_utilization > 75:
                print("⚠️ GPU 메모리 사용량 높음")

        print(f"CPU 사용률: {stats.cpu_usage:.1f}%")
        print(f"RAM 사용률: {stats.ram_usage:.1f}%")
        print(f"RAM 여유: {stats.ram_available:.1f}GB")

        # CPU/RAM 경고
        if stats.cpu_usage > 90:
            print("⚠️ CPU 사용률 높음")
        if stats.ram_usage > 90:
            print("⚠️ RAM 사용률 높음")

    def start_monitoring(self, interval: float = 5.0):
        """모니터링 시작 (백그라운드 스레드)"""

        if self.monitoring:
            print("이미 모니터링이 실행 중입니다.")
            return

        self.monitoring = True
        print(f"시스템 모니터링 시작 (간격: {interval}초)")

        def monitor_loop():
            try:
                while self.monitoring:
                    stats = self.get_current_stats()
                    self.stats_history.append(stats)

                    # 최근 100개 기록만 유지
                    if len(self.stats_history) > 100:
                        self.stats_history = self.stats_history[-100:]

                    time.sleep(interval)

            except Exception as e:
                print(f"모니터링 오류: {e}")
                self.monitoring = False

        # 백그라운드 스레드로 실행
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def stop_monitoring(self):
        """모니터링 중단"""
        self.monitoring = False

    def get_memory_trend(self, window_size: int = 10) -> Dict:
        """메모리 사용량 트렌드 분석"""

        if len(self.stats_history) < window_size:
            return {'trend': 'insufficient_data'}

        recent_stats = self.stats_history[-window_size:]

        # GPU 메모리 트렌드
        gpu_memory_values = [s.gpu_memory_allocated for s in recent_stats]
        gpu_trend = self._calculate_trend(gpu_memory_values)

        # RAM 트렌드
        ram_values = [s.ram_usage for s in recent_stats]
        ram_trend = self._calculate_trend(ram_values)

        return {
            'gpu_memory_trend': gpu_trend,
            'ram_trend': ram_trend,
            'window_size': window_size,
            'latest_gpu_memory': gpu_memory_values[-1],
            'latest_ram_usage': ram_values[-1]
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """값들의 트렌드 계산"""

        if len(values) < 2:
            return 'stable'

        # 선형 회귀로 기울기 계산
        n = len(values)
        x = list(range(n))

        # 기울기 계산
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 'stable'

        slope = numerator / denominator

        # 트렌드 분류
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    def get_performance_summary(self) -> Dict:
        """성능 요약 정보"""

        if not self.stats_history:
            return {'error': 'no_data'}

        latest = self.stats_history[-1]

        # 평균값 계산 (최근 10개)
        recent_window = min(10, len(self.stats_history))
        recent_stats = self.stats_history[-recent_window:]

        avg_gpu_memory = sum(s.gpu_memory_allocated for s in recent_stats) / recent_window
        avg_cpu_usage = sum(s.cpu_usage for s in recent_stats) / recent_window
        avg_ram_usage = sum(s.ram_usage for s in recent_stats) / recent_window

        # 최대값
        max_gpu_memory = max(s.gpu_memory_allocated for s in recent_stats)
        max_cpu_usage = max(s.cpu_usage for s in recent_stats)
        max_ram_usage = max(s.ram_usage for s in recent_stats)

        return {
            'current': {
                'gpu_memory_gb': latest.gpu_memory_allocated,
                'cpu_usage_pct': latest.cpu_usage,
                'ram_usage_pct': latest.ram_usage
            },
            'average': {
                'gpu_memory_gb': avg_gpu_memory,
                'cpu_usage_pct': avg_cpu_usage,
                'ram_usage_pct': avg_ram_usage
            },
            'peak': {
                'gpu_memory_gb': max_gpu_memory,
                'cpu_usage_pct': max_cpu_usage,
                'ram_usage_pct': max_ram_usage
            },
            'window_size': recent_window
        }


class MemoryManager:
    """메모리 관리 및 최적화"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def cleanup_memory(self, verbose: bool = True):
        """메모리 정리"""

        if verbose:
            print("메모리 정리 중...")

        # Python 가비지 컬렉션
        gc.collect()

        # GPU 메모리 정리
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if verbose:
            print("✅ 메모리 정리 완료")

    def get_memory_info(self) -> Dict:
        """메모리 정보 조회"""

        info = {}

        if self.device == "cuda":
            try:
                # GPU 메모리 정보
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                total = torch.cuda.get_device_properties(0).total_memory

                info['gpu'] = {
                    'allocated_gb': allocated / 1024 ** 3,
                    'reserved_gb': reserved / 1024 ** 3,
                    'total_gb': total / 1024 ** 3,
                    'free_gb': (total - reserved) / 1024 ** 3,
                    'utilization_pct': (allocated / total) * 100
                }

                # GPU 모델 정보
                info['gpu']['name'] = torch.cuda.get_device_name(0)
                info['gpu']['capability'] = torch.cuda.get_device_capability(0)

            except Exception as e:
                info['gpu'] = {'error': str(e)}

        # CPU 및 RAM 정보
        memory = psutil.virtual_memory()
        info['ram'] = {
            'total_gb': memory.total / 1024 ** 3,
            'available_gb': memory.available / 1024 ** 3,
            'used_gb': memory.used / 1024 ** 3,
            'usage_pct': memory.percent
        }

        info['cpu'] = {
            'usage_pct': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }

        return info

    def check_memory_availability(self, required_gb: float) -> Dict:
        """필요한 메모리가 사용 가능한지 확인"""

        info = self.get_memory_info()

        result = {
            'required_gb': required_gb,
            'available': False,
            'recommendations': []
        }

        if self.device == "cuda" and 'gpu' in info:
            gpu_info = info['gpu']
            if 'free_gb' in gpu_info:
                gpu_free = gpu_info['free_gb']

                if gpu_free >= required_gb:
                    result['available'] = True
                    result['gpu_free_gb'] = gpu_free
                else:
                    result['gpu_free_gb'] = gpu_free
                    result['gpu_shortage_gb'] = required_gb - gpu_free

                    # 권장사항
                    if gpu_info['utilization_pct'] > 80:
                        result['recommendations'].append("메모리 정리 수행")
                    if required_gb > gpu_info['total_gb'] * 0.8:
                        result['recommendations'].append("더 작은 모델 사용")
                        result['recommendations'].append("배치 크기 줄이기")

        # RAM 확인
        ram_info = info['ram']
        ram_free = ram_info['available_gb']

        if not result['available']:  # GPU가 부족한 경우 RAM 사용 고려
            if ram_free >= required_gb:
                result['cpu_fallback_possible'] = True
                result['recommendations'].append("CPU 모드로 전환")
            else:
                result['cpu_fallback_possible'] = False
                result['recommendations'].append("시스템 메모리 부족")

        return result

    def optimize_for_model(self, model_name: str) -> Dict:
        """모델별 메모리 최적화"""

        # 모델별 예상 메모리 사용량 (GB)
        model_memory_requirements = {
            "openai/whisper-base": 1.0,
            "openai/whisper-medium": 2.5,
            "openai/whisper-large-v3": 6.0,
            "pyannote/speaker-diarization": 2.0
        }

        required_memory = model_memory_requirements.get(model_name, 3.0)

        # 현재 메모리 상태 확인
        availability = self.check_memory_availability(required_memory)

        optimization = {
            'model': model_name,
            'required_memory_gb': required_memory,
            'memory_available': availability['available'],
            'optimizations_applied': []
        }

        if not availability['available']:
            # 메모리 정리 시도
            self.cleanup_memory()
            optimization['optimizations_applied'].append("memory_cleanup")

            # 재확인
            availability = self.check_memory_availability(required_memory)
            optimization['memory_available_after_cleanup'] = availability['available']

            if not availability['available']:
                # 추가 최적화 제안
                optimization['suggestions'] = availability['recommendations']

        return optimization


class ProcessingTimer:
    """처리 시간 측정 및 분석"""

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}

    def start(self, operation_name: str):
        """타이밍 시작"""
        self.start_times[operation_name] = time.time()

    def end(self, operation_name: str) -> float:
        """타이밍 종료 및 기록"""

        if operation_name not in self.start_times:
            raise ValueError(f"타이밍이 시작되지 않음: {operation_name}")

        elapsed = time.time() - self.start_times[operation_name]

        if operation_name not in self.timings:
            self.timings[operation_name] = []

        self.timings[operation_name].append(elapsed)
        del self.start_times[operation_name]

        return elapsed

    def get_statistics(self, operation_name: str) -> Dict:
        """작업별 통계"""

        if operation_name not in self.timings:
            return {'error': 'no_data'}

        times = self.timings[operation_name]

        return {
            'count': len(times),
            'total_time': sum(times),
            'average_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'last_time': times[-1] if times else 0
        }

    def get_all_statistics(self) -> Dict:
        """모든 작업 통계"""

        all_stats = {}

        for operation in self.timings:
            all_stats[operation] = self.get_statistics(operation)

        return all_stats

    def print_summary(self):
        """통계 요약 출력"""

        print("\n=== 처리 시간 통계 ===")

        for operation, stats in self.get_all_statistics().items():
            if 'error' not in stats:
                print(f"\n{operation}:")
                print(f"  횟수: {stats['count']}")
                print(f"  평균: {stats['average_time']:.2f}초")
                print(f"  최소: {stats['min_time']:.2f}초")
                print(f"  최대: {stats['max_time']:.2f}초")
                print(f"  총 시간: {stats['total_time']:.2f}초")


class ResourceMonitor:
    """리소스 모니터링 컨텍스트 매니저"""

    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.monitor = SystemMonitor()
        self.timer = ProcessingTimer()
        self.memory_manager = MemoryManager()

        self.start_stats = None
        self.end_stats = None

    def __enter__(self):
        """컨텍스트 시작"""
        # 시작 시점 기록
        self.start_stats = self.monitor.get_current_stats()
        self.timer.start(self.operation_name)

        print(f"🚀 {self.operation_name} 시작")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료"""
        # 종료 시점 기록
        try:
            elapsed_time = self.timer.end(self.operation_name)
            self.end_stats = self.monitor.get_current_stats()

            # 결과 출력
            print(f"✅ {self.operation_name} 완료 ({elapsed_time:.2f}초)")

        except Exception as timer_error:
            print(f"⚠️ 타이머 종료 중 오류: {timer_error}")
            elapsed_time = 0
            self.end_stats = self.monitor.get_current_stats()

        # 메모리 사용량 변화 계산
        if self.start_stats and self.end_stats:
            gpu_memory_diff = self.end_stats.gpu_memory_allocated - self.start_stats.gpu_memory_allocated
            ram_diff = self.end_stats.ram_usage - self.start_stats.ram_usage

            print(f"메모리 변화:")
            print(f"  GPU: {gpu_memory_diff:+.2f}GB")
            print(f"  RAM: {ram_diff:+.1f}%")

            # 메모리 누수 경고
            if gpu_memory_diff > 1.0:
                print("⚠️ GPU 메모리 사용량 크게 증가")
            if ram_diff > 10:
                print("⚠️ RAM 사용량 크게 증가")

        # 예외가 발생한 경우에도 정상적으로 처리
        if exc_type is not None:
            print(f"❌ {self.operation_name} 중 오류 발생: {exc_val}")
            # 메모리 정리 시도
            try:
                self.memory_manager.cleanup_memory(verbose=False)
            except Exception as cleanup_error:
                print(f"⚠️ 메모리 정리 중 오류: {cleanup_error}")

            # 예외를 억제하지 않고 전파 (False 반환)
            return False

        # 정상 종료
        return True

    def get_resource_usage(self) -> Dict:
        """리소스 사용량 반환"""

        if not (self.start_stats and self.end_stats):
            return {'error': 'incomplete_monitoring'}

        # timer에서 통계 가져오기 시 안전하게 처리
        try:
            duration_stats = self.timer.get_statistics(self.operation_name)
            if 'error' in duration_stats:
                duration_stats = {'last_time': 0, 'average_time': 0}
        except Exception:
            duration_stats = {'last_time': 0, 'average_time': 0}

        return {
            'operation': self.operation_name,
            'duration': duration_stats,
            'memory_change': {
                'gpu_gb': self.end_stats.gpu_memory_allocated - self.start_stats.gpu_memory_allocated,
                'ram_pct': self.end_stats.ram_usage - self.start_stats.ram_usage
            },
            'final_usage': {
                'gpu_gb': self.end_stats.gpu_memory_allocated,
                'ram_pct': self.end_stats.ram_usage,
                'cpu_pct': self.end_stats.cpu_usage
            }
        }


class PerformanceAnalyzer:
    """성능 분석 및 최적화 제안"""

    def __init__(self):
        self.benchmark_results = {}

    def analyze_gpu_compatibility(self) -> Dict:
        """GPU 호환성 및 최적화 분석"""

        analysis = {
            'gpu_detected': torch.cuda.is_available(),
            'recommendations': []
        }

        if not torch.cuda.is_available():
            analysis['recommendations'] = [
                "CUDA가 설치되지 않았거나 GPU를 감지할 수 없습니다",
                "CPU 모드로 실행하거나 GPU 드라이버를 확인하세요"
            ]
            return analysis

        try:
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

            analysis['gpu_name'] = gpu_name
            analysis['total_memory_gb'] = total_memory

            # GPU별 최적화 제안
            if "4060" in gpu_name:
                analysis['gpu_tier'] = "mid_range"
                analysis['recommended_models'] = ["whisper-base", "whisper-medium"]
                analysis['max_batch_size'] = 2
                analysis['recommendations'] = [
                    "RTX 4060 감지됨 - medium 모델 권장",
                    "FP16 정밀도 사용으로 메모리 절약",
                    "배치 크기는 2 이하로 제한",
                    "청크 단위 처리로 긴 오디오 안전 처리"
                ]
            elif "3090" in gpu_name or "4090" in gpu_name:
                analysis['gpu_tier'] = "high_end"
                analysis['recommended_models'] = ["whisper-large-v3"]
                analysis['max_batch_size'] = 8
                analysis['recommendations'] = [
                    "고성능 GPU 감지됨 - large 모델 사용 가능",
                    "배치 크기 최대 8까지 사용 가능",
                    "실시간 처리 가능"
                ]
            elif "A100" in gpu_name:
                analysis['gpu_tier'] = "enterprise"
                analysis['recommended_models'] = ["whisper-large-v3"]
                analysis['max_batch_size'] = 16
                analysis['recommendations'] = [
                    "엔터프라이즈급 GPU - 최고 성능 모델 사용",
                    "대용량 배치 처리 가능",
                    "멀티 GPU 활용 고려"
                ]
            else:
                # 메모리 기반 추정
                if total_memory < 8:
                    analysis['gpu_tier'] = "low_end"
                    analysis['recommended_models'] = ["whisper-base"]
                    analysis['max_batch_size'] = 1
                elif total_memory < 16:
                    analysis['gpu_tier'] = "mid_range"
                    analysis['recommended_models'] = ["whisper-medium"]
                    analysis['max_batch_size'] = 4
                else:
                    analysis['gpu_tier'] = "high_end"
                    analysis['recommended_models'] = ["whisper-large-v3"]
                    analysis['max_batch_size'] = 8

            # 메모리 최적화 제안
            if total_memory < 12:
                analysis['recommendations'].extend([
                    "메모리 제한 GPU - 주기적 메모리 정리 필요",
                    "긴 오디오는 청크 단위로 처리"
                ])

        except Exception as e:
            analysis['error'] = str(e)
            analysis['recommendations'] = ["GPU 정보 확인 실패 - CPU 모드 사용"]

        return analysis

    def benchmark_model_loading(self, models: List[str]) -> Dict:
        """모델 로딩 성능 벤치마크"""

        results = {}

        for model_name in models:
            print(f"벤치마킹: {model_name}")

            try:
                # 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 로딩 시간 측정
                start_time = time.time()

                # 간단한 모델 로딩 테스트
                from transformers import WhisperProcessor
                processor = WhisperProcessor.from_pretrained(model_name)

                load_time = time.time() - start_time

                # 메모리 사용량
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated(0) / 1024 ** 3
                else:
                    memory_used = 0

                results[model_name] = {
                    'load_time_seconds': load_time,
                    'memory_usage_gb': memory_used,
                    'success': True
                }

                # 정리
                del processor

            except Exception as e:
                results[model_name] = {
                    'success': False,
                    'error': str(e)
                }

            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def suggest_optimal_config(self, audio_duration: float,
                               num_speakers: int = None) -> Dict:
        """최적 설정 제안"""

        gpu_analysis = self.analyze_gpu_compatibility()

        config = {
            'audio_duration': audio_duration,
            'estimated_speakers': num_speakers,
            'processing_strategy': 'sequential'  # 기본값
        }

        # GPU 기반 모델 선택
        if gpu_analysis['gpu_detected'] and 'recommended_models' in gpu_analysis:
            config['recommended_model'] = gpu_analysis['recommended_models'][0]
            config['max_batch_size'] = gpu_analysis.get('max_batch_size', 2)
        else:
            config['recommended_model'] = "openai/whisper-base"
            config['max_batch_size'] = 1
            config['use_cpu'] = True

        # 오디오 길이 기반 최적화
        if audio_duration > 300:  # 5분 이상
            config['processing_strategy'] = 'chunked'
            config['chunk_duration'] = 60
            config['recommendations'] = [
                "긴 오디오 감지됨 - 청크 단위 처리",
                "중간 결과 저장으로 안정성 향상"
            ]
        elif audio_duration > 60:  # 1분 이상
            config['processing_strategy'] = 'chunked'
            config['chunk_duration'] = 30
        else:
            config['processing_strategy'] = 'full'

        # 화자 수 기반 최적화
        if num_speakers and num_speakers > 3:
            config['diarization_method'] = 'pyannote'
            config['clustering_method'] = 'agglomerative'
        else:
            config['diarization_method'] = 'lightweight'
            config['clustering_method'] = 'kmeans'

        # 메모리 기반 조정
        memory_info = MemoryManager().get_memory_info()

        if torch.cuda.is_available() and 'gpu' in memory_info:
            available_memory = memory_info['gpu'].get('free_gb', 0)

            if available_memory < 2:
                config['force_cpu'] = True
                config['recommendations'] = config.get('recommendations', [])
                config['recommendations'].append("GPU 메모리 부족 - CPU 모드 권장")
            elif available_memory < 4:
                config['use_fp16'] = True
                config['recommendations'] = config.get('recommendations', [])
                config['recommendations'].append("FP16 정밀도로 메모리 절약")

        return config


def quick_system_check() -> Dict:
    """빠른 시스템 체크"""

    monitor = SystemMonitor()
    memory_manager = MemoryManager()
    analyzer = PerformanceAnalyzer()

    return {
        'system_stats': monitor.get_current_stats(),
        'memory_info': memory_manager.get_memory_info(),
        'gpu_analysis': analyzer.analyze_gpu_compatibility(),
        'timestamp': datetime.now()
    }


def print_system_info():
    """시스템 정보 출력"""

    info = quick_system_check()

    print("=== 시스템 정보 ===")

    # GPU 정보
    gpu_analysis = info['gpu_analysis']
    if gpu_analysis['gpu_detected']:
        print(f"GPU: {gpu_analysis.get('gpu_name', 'Unknown')}")
        print(f"메모리: {gpu_analysis.get('total_memory_gb', 0):.1f}GB")
        print(f"등급: {gpu_analysis.get('gpu_tier', 'unknown')}")
        print(f"권장 모델: {', '.join(gpu_analysis.get('recommended_models', []))}")
    else:
        print("GPU: 감지되지 않음 (CPU 모드)")

    # 메모리 정보
    memory_info = info['memory_info']
    print(f"\nRAM: {memory_info['ram']['total_gb']:.1f}GB")
    print(f"RAM 사용률: {memory_info['ram']['usage_pct']:.1f}%")
    print(f"CPU 사용률: {memory_info['cpu']['usage_pct']:.1f}%")

    # 권장사항
    if 'recommendations' in gpu_analysis:
        print(f"\n권장사항:")
        for rec in gpu_analysis['recommendations']:
            print(f"  • {rec}")


def monitor_processing(operation_name: str):
    """처리 모니터링 데코레이터"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with ResourceMonitor(operation_name) as monitor:
                result = func(*args, **kwargs)

                # 리소스 사용량 정보를 결과에 추가 (가능한 경우)
                if isinstance(result, dict):
                    result['_resource_usage'] = monitor.get_resource_usage()

                return result

        return wrapper

    return decorator


if __name__ == "__main__":
    # 테스트 코드
    print("시스템 모니터링 모듈 로드됨")

    # 시스템 정보 출력
    print_system_info()

    # 성능 분석 예시
    analyzer = PerformanceAnalyzer()

    # 1시간 오디오, 3명 화자 가정
    optimal_config = analyzer.suggest_optimal_config(
        audio_duration=3600,  # 1시간
        num_speakers=3
    )

    print(f"\n=== 최적 설정 제안 ===")
    print(f"처리 전략: {optimal_config['processing_strategy']}")
    print(f"권장 모델: {optimal_config['recommended_model']}")
    print(f"배치 크기: {optimal_config['max_batch_size']}")

    if 'recommendations' in optimal_config:
        print("추가 권장사항:")
        for rec in optimal_config['recommendations']:
            print(f"  • {rec}")