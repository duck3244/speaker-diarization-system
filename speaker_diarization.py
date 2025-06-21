# speaker_diarization.py
"""
화자 분리 (Speaker Diarization) 기능
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import warnings

warnings.filterwarnings("ignore")

try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("pyannote.audio를 사용할 수 없습니다. 경량 화자 분리 모드를 사용합니다.")

from config import processing_config, system_config
from audio_utils import AudioFeatureExtractor


class BaseSpeakerDiarization:
    """화자 분리 기본 클래스"""

    def __init__(self, config=None):
        self.config = config or processing_config
        self.device = system_config.device

    def diarize(self, audio: np.ndarray, sr: int, segments: List[Tuple[float, float]]) -> List[int]:
        """화자 분리 수행 (추상 메서드)"""
        raise NotImplementedError


class PyAnnoteDiarization(BaseSpeakerDiarization):
    """pyannote.audio 기반 화자 분리"""

    def __init__(self, model_name: str = None, config=None):
        super().__init__(config)

        if not PYANNOTE_AVAILABLE:
            raise ImportError("pyannote.audio가 설치되지 않았습니다.")

        self.model_name = model_name or "pyannote/speaker-diarization-3.1"
        self.pipeline = None
        self.embedding_model = None
        self.pipeline_available = False

        self._load_models()

    def _load_models(self):
        """모델 로드"""
        try:
            # 화자 분리 파이프라인 (토큰 없이 시도)
            try:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=False
                )
                self.pipeline_available = True
                print(f"pyannote 파이프라인 로드 성공")
            except Exception as pipeline_error:
                print(f"pyannote 파이프라인 로드 실패: {pipeline_error}")
                print("임베딩 모델만 사용합니다.")
                self.pipeline_available = False

            # 화자 임베딩 모델 (항상 로드 시도)
            self.embedding_model = PretrainedSpeakerEmbedding(
                "speechbrain/spkrec-ecapa-voxceleb",
                device=self.device
            )

            print(f"pyannote 모델 로드 완료: {self.model_name}")

        except Exception as e:
            print(f"pyannote 모델 로드 실패: {e}")
            raise

    def diarize_from_file(self, audio_path: str) -> List[Dict]:
        """파일에서 직접 화자 분리"""

        if not self.pipeline_available or self.pipeline is None:
            print("파이프라인을 사용할 수 없습니다. 임베딩 기반 처리로 전환합니다.")
            # 오디오 로드 후 임베딩 기반 처리
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)

            # 간단한 VAD로 세그먼트 생성
            from audio_utils import VADProcessor
            vad = VADProcessor()
            segments = vad.enhanced_vad(audio, sr)

            # 임베딩 기반 화자 분리
            speaker_labels = self.diarize(audio, sr, segments)

            # 결과 변환
            results = []
            for i, ((start, end), speaker_id) in enumerate(zip(segments, speaker_labels)):
                results.append({
                    'start': start,
                    'end': end,
                    'speaker': f"SPEAKER_{speaker_id}",
                    'duration': end - start
                })

            return results

        # 화자 분리 실행
        diarization = self.pipeline(audio_path)

        # 결과 변환 (안전한 처리) - 수정된 부분
        results = []
        try:
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                results.append({
                    'start': segment.start,
                    'end': segment.end,
                    'speaker': speaker,
                    'duration': segment.end - segment.start
                })
        except ValueError as ve:
            print(f"diarization 결과 unpacking 오류: {ve}")
            # 안전한 처리 방법으로 전환
            try:
                for item in diarization.itertracks(yield_label=True):
                    if len(item) == 3:
                        segment, track, speaker = item
                    elif len(item) == 2:
                        segment, speaker = item
                        track = None
                    else:
                        print(f"예상치 못한 item 길이: {len(item)}")
                        continue

                    results.append({
                        'start': segment.start,
                        'end': segment.end,
                        'speaker': speaker,
                        'duration': segment.end - segment.start
                    })
            except Exception as e:
                print(f"diarization 결과 처리 중 오류: {e}")
                print("임베딩 기반 처리로 폴백합니다.")
                # 완전 폴백
                import librosa
                audio, sr = librosa.load(audio_path, sr=16000)
                from audio_utils import VADProcessor
                vad = VADProcessor()
                segments = vad.enhanced_vad(audio, sr)
                speaker_labels = self.diarize(audio, sr, segments)

                results = []
                for i, ((start, end), speaker_id) in enumerate(zip(segments, speaker_labels)):
                    results.append({
                        'start': start,
                        'end': end,
                        'speaker': f"SPEAKER_{speaker_id}",
                        'duration': end - start
                    })

        return results

    def extract_embeddings(self, audio: np.ndarray, segments: List[Tuple[float, float]],
                           sr: int) -> np.ndarray:
        """세그먼트별 화자 임베딩 추출"""

        if self.embedding_model is None:
            raise RuntimeError("임베딩 모델이 로드되지 않았습니다.")

        embeddings = []

        for start, end in segments:
            # 세그먼트 추출
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) > sr * 0.5:  # 최소 0.5초
                # 임베딩 추출
                with torch.no_grad():
                    embedding = self.embedding_model(
                        torch.tensor(segment_audio).unsqueeze(0).to(self.device)
                    )
                embeddings.append(embedding.cpu().numpy().flatten())
            else:
                # 짧은 구간은 제로 벡터
                embeddings.append(np.zeros(192))  # ECAPA-TDNN 차원

        return np.array(embeddings)

    def diarize(self, audio: np.ndarray, sr: int, segments: List[Tuple[float, float]]) -> List[int]:
        """세그먼트 기반 화자 분리"""

        # 임베딩 추출
        embeddings = self.extract_embeddings(audio, segments, sr)

        # 클러스터링
        speaker_labels = self._cluster_speakers(embeddings)

        return speaker_labels.tolist()

    def _cluster_speakers(self, embeddings: np.ndarray, n_speakers: int = None) -> np.ndarray:
        """화자 클러스터링"""

        if len(embeddings) <= 1:
            return np.array([0])

        if n_speakers is None:
            n_speakers = self._estimate_num_speakers(embeddings)

        # Agglomerative Clustering 사용
        clustering = AgglomerativeClustering(
            n_clusters=n_speakers,
            linkage='cosine',
            metric='cosine'
        )

        speaker_labels = clustering.fit_predict(embeddings)
        return speaker_labels

    def _estimate_num_speakers(self, embeddings: np.ndarray) -> int:
        """화자 수 자동 추정"""

        max_speakers = min(self.config.max_speakers, len(embeddings))
        best_score = -1
        best_n = 2

        for n in range(2, max_speakers + 1):
            try:
                clustering = AgglomerativeClustering(n_clusters=n, linkage='cosine')
                labels = clustering.fit_predict(embeddings)

                # 실루엣 스코어로 평가
                score = silhouette_score(embeddings, labels, metric='cosine')

                if score > best_score:
                    best_score = score
                    best_n = n

            except Exception:
                continue

        return best_n


class LightweightSpeakerDiarization(BaseSpeakerDiarization):
    """경량 화자 분리 (MFCC + 클러스터링 기반)"""

    def __init__(self, config=None):
        super().__init__(config)
        self.feature_extractor = AudioFeatureExtractor()

    def diarize(self, audio: np.ndarray, sr: int, segments: List[Tuple[float, float]]) -> List[int]:
        """경량 화자 분리 수행"""

        # 특성 추출
        features = self._extract_segment_features(audio, segments, sr)

        if len(features) <= 1:
            return [0] * len(segments)

        # 클러스터링
        speaker_labels = self._cluster_speakers(features)

        return speaker_labels.tolist()

    def _extract_segment_features(self, audio: np.ndarray,
                                  segments: List[Tuple[float, float]],
                                  sr: int) -> np.ndarray:
        """세그먼트별 특성 추출"""

        features_list = []

        for start, end in segments:
            # 세그먼트 추출
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) > sr * 0.3:  # 최소 0.3초
                # 종합 특성 추출
                features = self.feature_extractor.extract_comprehensive_features(segment_audio, sr)
                features_list.append(features)
            else:
                # 짧은 구간은 평균 특성으로 대체
                if features_list:
                    features_list.append(np.mean(features_list, axis=0))
                else:
                    # 첫 번째가 짧으면 제로 벡터
                    dummy_features = np.zeros(52)  # 예상 특성 차원
                    features_list.append(dummy_features)

        return np.array(features_list)

    def _cluster_speakers(self, features: np.ndarray, n_speakers: int = None) -> np.ndarray:
        """특성 기반 화자 클러스터링"""

        if len(features) <= 1:
            return np.array([0])

        if n_speakers is None:
            n_speakers = self._estimate_num_speakers_lightweight(features)

        # K-means 클러스터링 (빠르고 효율적)
        clustering = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        speaker_labels = clustering.fit_predict(features)

        return speaker_labels

    def _estimate_num_speakers_lightweight(self, features: np.ndarray) -> int:
        """경량 화자 수 추정"""

        max_speakers = min(5, len(features))  # 최대 5명으로 제한

        if len(features) <= 2:
            return len(features)

        # 엘보우 방법으로 최적 클러스터 수 찾기
        inertias = []
        K_range = range(1, max_speakers + 1)

        for k in K_range:
            if k <= len(features):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features)
                inertias.append(kmeans.inertia_)
            else:
                break

        # 엘보우 포인트 찾기
        if len(inertias) >= 3:
            # 2차 차분으로 엘보우 포인트 찾기
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)

            if len(second_diffs) > 0:
                elbow_point = np.argmax(second_diffs) + 2  # 인덱스 보정
                return min(elbow_point, max_speakers)

        # 기본값: 2명
        return min(3, max_speakers)  # 기존: 2 → 3


class HybridSpeakerDiarization(BaseSpeakerDiarization):
    """하이브리드 화자 분리 (상황에 따라 적응)"""

    def __init__(self, config=None):
        super().__init__(config)

        # 사용 가능한 방법들 초기화
        self.pyannote_diarizer = None
        self.lightweight_diarizer = LightweightSpeakerDiarization(config)

        # pyannote 사용 가능 여부 확인
        if PYANNOTE_AVAILABLE:
            try:
                self.pyannote_diarizer = PyAnnoteDiarization(config=config)
                self.use_pyannote = True
                print("PyAnnote 모델 사용 가능")
            except Exception as e:
                print(f"PyAnnote 사용 불가: {e}")
                self.use_pyannote = False
        else:
            self.use_pyannote = False

    def diarize(self, audio: np.ndarray, sr: int, segments: List[Tuple[float, float]]) -> List[int]:
        """상황에 맞는 최적 화자 분리"""

        # 조건에 따라 방법 선택
        if self._should_use_pyannote(audio, sr, segments):
            return self.pyannote_diarizer.diarize(audio, sr, segments)
        else:
            return self.lightweight_diarizer.diarize(audio, sr, segments)

    def _should_use_pyannote(self, audio: np.ndarray, sr: int,
                             segments: List[Tuple[float, float]]) -> bool:
        """PyAnnote 사용 여부 결정 - 임시로 항상 False 반환"""

        # 임시 수정: pyannote 파이프라인 오류 해결 전까지 경량 모드만 사용
        return False

        # 원래 코드 (문제 해결 후 복원용)
        """
        # PyAnnote 파이프라인이 사용 불가능하면 항상 False
        if not self.use_pyannote or not hasattr(self.pyannote_diarizer,
                                                'pipeline_available') or not self.pyannote_diarizer.pipeline_available:
            return False

        # 조건들
        audio_duration = len(audio) / sr
        num_segments = len(segments)
        avg_segment_length = np.mean([end - start for start, end in segments]) if segments else 0

        # 복잡한 경우에만 PyAnnote 사용
        complex_audio = (
                audio_duration > 60 or  # 1분 이상
                num_segments > 20 or  # 세그먼트 많음
                avg_segment_length < 2  # 짧은 세그먼트들
        )

        # GPU 메모리 여유 확인
        if self.device == "cuda":
            try:
                available_memory = torch.cuda.get_device_properties(0).total_memory
                used_memory = torch.cuda.memory_allocated(0)
                free_memory = (available_memory - used_memory) / 1024 ** 3

                memory_sufficient = free_memory > 2.0  # 2GB 이상 여유
            except:
                memory_sufficient = False
        else:
            memory_sufficient = False

        return complex_audio and memory_sufficient
        """


class SpeakerDiarizationEvaluator:
    """화자 분리 성능 평가"""

    def __init__(self):
        pass

    def calculate_der(self, predicted_labels: List[int],
                      true_labels: List[int],
                      segments: List[Tuple[float, float]]) -> float:
        """Diarization Error Rate (DER) 계산"""

        if len(predicted_labels) != len(true_labels):
            raise ValueError("예측과 실제 라벨 길이가 다릅니다.")

        total_duration = sum(end - start for start, end in segments)
        error_duration = 0

        for pred, true, (start, end) in zip(predicted_labels, true_labels, segments):
            if pred != true:
                error_duration += end - start

        der = error_duration / total_duration if total_duration > 0 else 0
        return der

    def calculate_speaker_accuracy(self, predicted_labels: List[int],
                                   true_labels: List[int]) -> float:
        """화자 분류 정확도 계산"""

        if len(predicted_labels) != len(true_labels):
            return 0.0

        correct = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)
        accuracy = correct / len(predicted_labels)

        return accuracy

    def evaluate_clustering_quality(self, features: np.ndarray,
                                    labels: np.ndarray) -> Dict:
        """클러스터링 품질 평가"""

        metrics = {}

        try:
            # 실루엣 스코어
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(features, labels)
                metrics['silhouette_score'] = silhouette
            else:
                metrics['silhouette_score'] = 0

            # 클러스터 내 응집도
            intra_cluster_distances = []
            for label in np.unique(labels):
                cluster_features = features[labels == label]
                if len(cluster_features) > 1:
                    distances = pdist(cluster_features)
                    intra_cluster_distances.extend(distances)

            if intra_cluster_distances:
                metrics['avg_intra_cluster_distance'] = np.mean(intra_cluster_distances)
            else:
                metrics['avg_intra_cluster_distance'] = 0

            # 클러스터 간 분리도
            centroids = []
            for label in np.unique(labels):
                cluster_features = features[labels == label]
                centroid = np.mean(cluster_features, axis=0)
                centroids.append(centroid)

            if len(centroids) > 1:
                inter_cluster_distances = pdist(centroids)
                metrics['avg_inter_cluster_distance'] = np.mean(inter_cluster_distances)
            else:
                metrics['avg_inter_cluster_distance'] = 0

        except Exception as e:
            print(f"평가 중 오류: {e}")
            metrics = {
                'silhouette_score': 0,
                'avg_intra_cluster_distance': 0,
                'avg_inter_cluster_distance': 0
            }

        return metrics


# 팩토리 함수
def create_speaker_diarizer(method: str = "auto", config=None) -> BaseSpeakerDiarization:
    """화자 분리기 생성"""

    if method == "auto":
        return HybridSpeakerDiarization(config)
    elif method == "pyannote":
        if PYANNOTE_AVAILABLE:
            return PyAnnoteDiarization(config=config)
        else:
            print("PyAnnote 사용 불가. 경량 모드로 전환")
            return LightweightSpeakerDiarization(config)
    elif method == "lightweight":
        return LightweightSpeakerDiarization(config)
    else:
        raise ValueError(f"알 수 없는 방법: {method}")


# 유틸리티 함수들
def post_process_speaker_labels(labels: List[int],
                                segments: List[Tuple[float, float]],
                                min_speaker_duration: float = 2.0) -> List[int]:
    """화자 라벨 후처리"""

    # 너무 짧은 화자 세그먼트 제거
    speaker_durations = {}
    for label, (start, end) in zip(labels, segments):
        if label not in speaker_durations:
            speaker_durations[label] = 0
        speaker_durations[label] += end - start

    # 짧은 화자들을 가장 가까운 긴 화자로 병합
    valid_speakers = [s for s, d in speaker_durations.items() if d >= min_speaker_duration]

    if not valid_speakers:
        return labels  # 모든 화자가 너무 짧으면 그대로 반환

    processed_labels = []
    for label in labels:
        if label in valid_speakers:
            processed_labels.append(label)
        else:
            # 가장 흔한 유효 화자로 대체
            most_common_speaker = max(valid_speakers,
                                      key=lambda x: speaker_durations[x])
            processed_labels.append(most_common_speaker)

    return processed_labels


def smooth_speaker_transitions(labels: List[int],
                               segments: List[Tuple[float, float]],
                               min_segment_duration: float = 0.5) -> List[int]:
    """화자 전환 스무딩"""

    if len(labels) <= 2:
        return labels

    smoothed_labels = labels.copy()

    for i in range(1, len(labels) - 1):
        current_duration = segments[i][1] - segments[i][0]

        # 짧은 세그먼트가 다른 화자로 분류된 경우
        if (current_duration < min_segment_duration and
                labels[i - 1] == labels[i + 1] and
                labels[i] != labels[i - 1]):
            smoothed_labels[i] = labels[i - 1]

    return smoothed_labels


if __name__ == "__main__":
    # 테스트 코드
    print("화자 분리 모듈 로드됨")

    # 사용 가능한 방법 확인
    if PYANNOTE_AVAILABLE:
        print("✅ PyAnnote 사용 가능")
    else:
        print("⚠️ PyAnnote 사용 불가 - 경량 모드만 사용")

    # 기본 화자 분리기 생성
    diarizer = create_speaker_diarizer("auto")
    print(f"화자 분리기 생성 완료: {type(diarizer).__name__}")