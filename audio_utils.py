# audio_utils.py
"""
오디오 처리 관련 유틸리티 함수들
"""

import librosa
import numpy as np
import soundfile as sf
from typing import List, Tuple, Dict, Optional
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

from config import audio_config

class AudioProcessor:
    """오디오 처리 클래스"""
    
    def __init__(self, config=None):
        self.config = config or audio_config
    
    def load_audio(self, audio_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """오디오 파일 로드"""
        sr = sr or self.config.sample_rate
        
        try:
            audio, actual_sr = librosa.load(audio_path, sr=sr)
            print(f"오디오 로드됨: {audio_path}")
            print(f"길이: {len(audio)/actual_sr:.2f}초, 샘플레이트: {actual_sr}Hz")
            return audio, actual_sr
        
        except Exception as e:
            raise Exception(f"오디오 로드 실패: {e}")
    
    def save_audio(self, audio: np.ndarray, output_path: str, sr: int = None) -> str:
        """오디오 파일 저장"""
        sr = sr or self.config.sample_rate
        
        try:
            sf.write(output_path, audio, sr)
            print(f"오디오 저장됨: {output_path}")
            return output_path
        
        except Exception as e:
            raise Exception(f"오디오 저장 실패: {e}")
    
    def normalize_audio(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """오디오 정규화"""
        
        # RMS 정규화
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)
        
        # 클리핑 방지
        audio = np.clip(audio, -0.95, 0.95)
        
        # 무음 구간 제거 (앞뒤)
        audio, _ = librosa.effects.trim(audio, top_db=30)
        
        return audio
    
    def check_audio_quality(self, audio: np.ndarray, sr: int) -> Dict:
        """오디오 품질 검사"""
        
        duration = len(audio) / sr
        
        # 1. 길이 체크
        duration_ok = self.config.min_duration <= duration <= self.config.max_duration
        
        # 2. 신호 강도 체크
        rms = np.sqrt(np.mean(audio**2))
        signal_ok = rms > self.config.min_rms
        
        # 3. 클리핑 체크
        clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
        clipping_ok = clipping_ratio < self.config.max_clipping_ratio
        
        # 4. SNR 추정
        noise_samples = np.concatenate([
            audio[:int(len(audio)*0.1)], 
            audio[int(len(audio)*0.9):]
        ])
        noise_power = np.mean(noise_samples**2)
        signal_power = np.mean(audio**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 50
            
        snr_ok = snr >= self.config.min_snr
        
        return {
            'duration': duration,
            'duration_ok': duration_ok,
            'rms': rms,
            'signal_ok': signal_ok,
            'clipping_ratio': clipping_ratio,
            'clipping_ok': clipping_ok,
            'snr': snr,
            'snr_ok': snr_ok,
            'overall_ok': all([duration_ok, signal_ok, clipping_ok, snr_ok])
        }
    
    def split_audio_chunks(self, audio: np.ndarray, sr: int, 
                          chunk_duration: float) -> List[Tuple[np.ndarray, float]]:
        """오디오를 청크로 분할"""
        
        chunk_samples = int(chunk_duration * sr)
        chunks = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk_audio = audio[i:i+chunk_samples]
            start_time = i / sr
            chunks.append((chunk_audio, start_time))
        
        return chunks
    
    def extract_segment(self, audio: np.ndarray, start_time: float, 
                       end_time: float, sr: int) -> np.ndarray:
        """특정 시간 구간 추출"""
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # 범위 체크
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        return audio[start_sample:end_sample]

class VADProcessor:
    """음성 활동 감지 (Voice Activity Detection) 프로세서"""
    
    def __init__(self, config=None):
        self.config = config or audio_config
    
    def simple_vad(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """에너지 기반 간단한 VAD"""
        
        # RMS 에너지 계산
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=self.config.frame_length,
            hop_length=self.config.hop_length
        )[0]
        
        # 임계값 설정
        threshold = np.percentile(rms, self.config.vad_threshold_percentile)
        
        # 시간 축 생성
        times = librosa.frames_to_time(
            np.arange(len(rms)), 
            sr=sr, 
            hop_length=self.config.hop_length
        )
        
        # 음성 활동 감지
        voice_activity = rms > threshold
        
        return self._find_voice_segments(times, voice_activity)
    
    def enhanced_vad(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """향상된 VAD (스펙트럴 특성 포함)"""
        
        hop_length = self.config.hop_length
        
        # 여러 특성 계산
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
        
        # 임계값 설정
        rms_threshold = np.percentile(rms, self.config.vad_threshold_percentile)
        centroid_threshold = np.percentile(spectral_centroids, self.config.spectral_threshold_percentile)
        zcr_threshold = np.percentile(zero_crossing_rate, 70)
        
        # 시간 축
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # 복합 음성 활동 감지
        voice_activity = (
            (rms > rms_threshold) & 
            (spectral_centroids > centroid_threshold) &
            (zero_crossing_rate < zcr_threshold)  # 음성은 ZCR이 낮음
        )
        
        return self._find_voice_segments(times, voice_activity)
    
    def _find_voice_segments(self, times: np.ndarray, 
                           voice_activity: np.ndarray) -> List[Tuple[float, float]]:
        """연속 음성 구간 찾기"""
        
        segments = []
        start = None
        
        for i, (time, is_voice) in enumerate(zip(times, voice_activity)):
            if is_voice and start is None:
                start = time
            elif not is_voice and start is not None:
                if time - start >= self.config.min_duration:
                    segments.append((start, time))
                start = None
        
        # 마지막 구간 처리
        if start is not None and times[-1] - start >= self.config.min_duration:
            segments.append((start, times[-1]))
        
        return segments
    
    def merge_close_segments(self, segments: List[Tuple[float, float]], 
                           gap_threshold: float = 0.5) -> List[Tuple[float, float]]:
        """가까운 구간들 병합"""
        
        if not segments:
            return segments
        
        merged = []
        current_start, current_end = segments[0]
        
        for start, end in segments[1:]:
            if start - current_end <= gap_threshold:
                # 구간 병합
                current_end = end
            else:
                # 새로운 구간 시작
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        return merged

class AudioFeatureExtractor:
    """오디오 특성 추출기"""
    
    def __init__(self, config=None):
        self.config = config or audio_config
    
    def extract_mfcc_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """MFCC 특성 추출 (화자 인식용)"""
        
        # MFCC 계산
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # 통계적 특성
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)
        
        # 특성 벡터 결합
        features = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta])
        
        return features
    
    def extract_pitch_features(self, audio: np.ndarray, sr: int) -> Dict:
        """피치 관련 특성 추출"""
        
        # 피치 추출
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        
        # 유효한 피치만 추출
        valid_pitches = pitches[pitches > 0]
        
        if len(valid_pitches) > 0:
            pitch_mean = np.mean(valid_pitches)
            pitch_std = np.std(valid_pitches)
            pitch_min = np.min(valid_pitches)
            pitch_max = np.max(valid_pitches)
        else:
            pitch_mean = pitch_std = pitch_min = pitch_max = 0
        
        return {
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_min': pitch_min,
            'pitch_max': pitch_max,
            'pitch_range': pitch_max - pitch_min
        }
    
    def extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict:
        """스펙트럴 특성 추출"""
        
        # 스펙트럴 특성들
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        return {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate)
        }
    
    def extract_comprehensive_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """종합적인 화자 특성 추출"""
        
        # MFCC 특성
        mfcc_features = self.extract_mfcc_features(audio, sr)
        
        # 피치 특성
        pitch_features = self.extract_pitch_features(audio, sr)
        pitch_vector = np.array(list(pitch_features.values()))
        
        # 스펙트럴 특성
        spectral_features = self.extract_spectral_features(audio, sr)
        spectral_vector = np.array(list(spectral_features.values()))
        
        # 모든 특성 결합
        comprehensive_features = np.concatenate([
            mfcc_features,
            pitch_vector,
            spectral_vector
        ])
        
        return comprehensive_features

# 유틸리티 함수들
def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """오디오 리샘플링"""
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    return audio

def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """프리엠퍼시스 필터 적용"""
    return librosa.effects.preemphasis(audio, coef=coeff)

def remove_silence(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """무음 구간 제거"""
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return audio_trimmed

def calculate_audio_stats(audio: np.ndarray, sr: int) -> Dict:
    """오디오 통계 정보 계산"""
    
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio**2))
    max_amplitude = np.max(np.abs(audio))
    
    # 동적 범위
    dynamic_range = 20 * np.log10(max_amplitude / (rms + 1e-10))
    
    return {
        'duration': duration,
        'sample_rate': sr,
        'rms': rms,
        'max_amplitude': max_amplitude,
        'dynamic_range': dynamic_range,
        'num_samples': len(audio)
    }

if __name__ == "__main__":
    # 테스트 코드
    processor = AudioProcessor()
    vad = VADProcessor()
    extractor = AudioFeatureExtractor()
    
    print("오디오 처리 유틸리티 모듈 로드됨")
