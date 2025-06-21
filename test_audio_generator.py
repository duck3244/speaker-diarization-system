# test_audio_generator.py
"""
테스트용 음성 파일 생성기
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import random


def generate_test_conversation(
        output_path: str = "test_conversation.wav",
        duration: float = 30.0,
        num_speakers: int = 2,
        sample_rate: int = 16000
):
    """
    테스트용 대화 음성 생성 (음성 합성 없이 패턴 기반)
    실제로는 TTS나 녹음이 필요하지만, 구조적 테스트용
    """

    # 전체 샘플 수
    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples)

    # 화자별 특성 (주파수, 진폭 패턴)
    speaker_configs = [
        {"freq": 150 + i * 50, "amplitude": 0.3 + i * 0.1}
        for i in range(num_speakers)
    ]

    # 발화 구간 생성
    segments = []
    current_time = 1.0  # 1초부터 시작

    while current_time < duration - 2:
        speaker_id = random.randint(0, num_speakers - 1)
        segment_duration = random.uniform(2.0, 5.0)  # 2-5초 발화

        if current_time + segment_duration > duration - 1:
            segment_duration = duration - current_time - 1

        segments.append({
            'speaker': speaker_id,
            'start': current_time,
            'end': current_time + segment_duration,
            'duration': segment_duration
        })

        current_time += segment_duration + random.uniform(0.5, 1.5)  # 간격

    # 음성 신호 생성
    for segment in segments:
        start_sample = int(segment['start'] * sample_rate)
        end_sample = int(segment['end'] * sample_rate)

        config = speaker_configs[segment['speaker']]

        # 기본 톤 생성
        t = np.linspace(0, segment['duration'], end_sample - start_sample)

        # 복합 주파수 (더 자연스러운 음성 패턴)
        signal = (
                config['amplitude'] * np.sin(2 * np.pi * config['freq'] * t) +
                config['amplitude'] * 0.3 * np.sin(2 * np.pi * config['freq'] * 2 * t) +
                config['amplitude'] * 0.1 * np.sin(2 * np.pi * config['freq'] * 3 * t)
        )

        # 엔벨로프 적용 (자연스러운 페이드)
        envelope = np.ones_like(signal)
        fade_samples = int(0.1 * sample_rate)  # 0.1초 페이드

        if len(signal) > 2 * fade_samples:
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        signal *= envelope

        # 약간의 노이즈 추가 (현실적인 음성)
        noise = np.random.normal(0, 0.02, len(signal))
        signal += noise

        # 오디오에 추가
        audio[start_sample:end_sample] = signal

    # 정규화
    audio = audio / np.max(np.abs(audio)) * 0.8

    # 저장
    sf.write(output_path, audio, sample_rate)

    # 세그먼트 정보 저장
    info_path = output_path.replace('.wav', '_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"테스트 대화 - {num_speakers}명 화자\n")
        f.write(f"총 길이: {duration:.1f}초\n\n")
        f.write("세그먼트 정보:\n")
        for i, seg in enumerate(segments):
            f.write(f"{i + 1}. 화자_{seg['speaker']} ({seg['start']:.1f}s - {seg['end']:.1f}s)\n")

    print(f"✅ 테스트 파일 생성: {output_path}")
    print(f"📄 세그먼트 정보: {info_path}")

    return segments


def download_sample_audio():
    """
    샘플 오디오 다운로드 가이드 출력
    """

    samples = [
        {
            "name": "짧은 대화 (2명)",
            "description": "2명이 간단한 대화",
            "duration": "1-3분",
            "difficulty": "쉬움",
            "example": "친구끼리 안부 인사"
        },
        {
            "name": "뉴스 인터뷰 (2-3명)",
            "description": "앵커 + 게스트 인터뷰",
            "duration": "5-10분",
            "difficulty": "보통",
            "example": "KBS, MBC 뉴스 인터뷰"
        },
        {
            "name": "토론 프로그램 (3-4명)",
            "description": "여러 명이 참여하는 토론",
            "duration": "10-30분",
            "difficulty": "어려움",
            "example": "100분 토론, 심야토론"
        },
        {
            "name": "팟캐스트 (2-3명)",
            "description": "진행자 + 게스트 대화",
            "duration": "20-60분",
            "difficulty": "보통",
            "example": "김어준의 뉴스공장"
        }
    ]

    print("=== 추천 테스트 오디오 ===\n")

    for i, sample in enumerate(samples, 1):
        print(f"{i}. {sample['name']}")
        print(f"   설명: {sample['description']}")
        print(f"   길이: {sample['duration']}")
        print(f"   난이도: {sample['difficulty']}")
        print(f"   예시: {sample['example']}")
        print()


def create_test_suite():
    """테스트 스위트 생성"""

    output_dir = Path("test_audio")
    output_dir.mkdir(exist_ok=True)

    test_cases = [
        {"name": "short_2speakers", "duration": 15, "speakers": 2},
        {"name": "medium_3speakers", "duration": 60, "speakers": 3},
        {"name": "long_2speakers", "duration": 180, "speakers": 2},
        {"name": "complex_4speakers", "duration": 120, "speakers": 4},
    ]

    print("🎵 테스트 스위트 생성 중...\n")

    for case in test_cases:
        output_path = output_dir / f"{case['name']}.wav"
        print(f"생성 중: {case['name']}")

        segments = generate_test_conversation(
            output_path=str(output_path),
            duration=case['duration'],
            num_speakers=case['speakers']
        )

        print(f"  - {case['speakers']}명 화자")
        print(f"  - {case['duration']}초 길이")
        print(f"  - {len(segments)}개 발화 구간\n")

    print("✅ 테스트 스위트 생성 완료!")
    print(f"📁 파일 위치: {output_dir.absolute()}")


def generate_test_conversation(
        output_path: str = "test_conversation.wav",
        duration: float = 30.0,
        num_speakers: int = 2,
        sample_rate: int = 16000
):
    """
    테스트용 대화 음성 생성 (음성 합성 없이 패턴 기반)
    실제로는 TTS나 녹음이 필요하지만, 구조적 테스트용
    """

    # 전체 샘플 수
    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples)

    # 화자별 특성 (주파수, 진폭 패턴)
    speaker_configs = [
        {"freq": 150 + i * 50, "amplitude": 0.3 + i * 0.1}
        for i in range(num_speakers)
    ]

    # 발화 구간 생성
    segments = []
    current_time = 1.0  # 1초부터 시작

    while current_time < duration - 2:
        speaker_id = random.randint(0, num_speakers - 1)
        segment_duration = random.uniform(2.0, 5.0)  # 2-5초 발화

        if current_time + segment_duration > duration - 1:
            segment_duration = duration - current_time - 1

        segments.append({
            'speaker': speaker_id,
            'start': current_time,
            'end': current_time + segment_duration,
            'duration': segment_duration
        })

        current_time += segment_duration + random.uniform(0.5, 1.5)  # 간격

    # 음성 신호 생성
    for segment in segments:
        start_sample = int(segment['start'] * sample_rate)
        end_sample = int(segment['end'] * sample_rate)

        config = speaker_configs[segment['speaker']]

        # 기본 톤 생성
        t = np.linspace(0, segment['duration'], end_sample - start_sample)

        # 복합 주파수 (더 자연스러운 음성 패턴)
        signal = (
                config['amplitude'] * np.sin(2 * np.pi * config['freq'] * t) +
                config['amplitude'] * 0.3 * np.sin(2 * np.pi * config['freq'] * 2 * t) +
                config['amplitude'] * 0.1 * np.sin(2 * np.pi * config['freq'] * 3 * t)
        )

        # 엔벨로프 적용 (자연스러운 페이드)
        envelope = np.ones_like(signal)
        fade_samples = int(0.1 * sample_rate)  # 0.1초 페이드

        if len(signal) > 2 * fade_samples:
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        signal *= envelope

        # 약간의 노이즈 추가 (현실적인 음성)
        noise = np.random.normal(0, 0.02, len(signal))
        signal += noise

        # 오디오에 추가
        audio[start_sample:end_sample] = signal

    # 정규화
    audio = audio / np.max(np.abs(audio)) * 0.8

    # 저장
    sf.write(output_path, audio, sample_rate)

    # 세그먼트 정보 저장
    info_path = output_path.replace('.wav', '_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"테스트 대화 - {num_speakers}명 화자\n")
        f.write(f"총 길이: {duration:.1f}초\n\n")
        f.write("세그먼트 정보:\n")
        for i, seg in enumerate(segments):
            f.write(f"{i + 1}. 화자_{seg['speaker']} ({seg['start']:.1f}s - {seg['end']:.1f}s)\n")

    print(f"✅ 테스트 파일 생성: {output_path}")
    print(f"📄 세그먼트 정보: {info_path}")

    return segments


def generate_realistic_test_conversation(
        output_path: str = "realistic_test.wav",
        duration: float = 30.0,
        num_speakers: int = 2,
        sample_rate: int = 16000
):
    """
    더 현실적인 테스트용 대화 음성 생성
    """

    # 전체 샘플 수
    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples)

    # 화자별 특성 (더 구분되는 특성)
    speaker_configs = [
        {"base_freq": 120, "formant1": 800, "formant2": 1200, "amplitude": 0.4},  # 남성 목소리
        {"base_freq": 180, "formant1": 1000, "formant2": 1800, "amplitude": 0.35},  # 여성 목소리
        {"base_freq": 150, "formant1": 900, "formant2": 1500, "amplitude": 0.38},  # 중간 목소리
        {"base_freq": 100, "formant1": 750, "formant2": 1100, "amplitude": 0.42}  # 낮은 목소리
    ]

    # 더 현실적인 발화 패턴
    segments = []
    current_time = 1.0

    # 대화 패턴: 번갈아가며 말하기
    current_speaker = 0

    while current_time < duration - 2:
        # 발화 길이를 더 다양하게
        if random.random() < 0.3:  # 30% 확률로 짧은 발화
            segment_duration = random.uniform(1.0, 2.5)
        else:  # 70% 확률로 긴 발화
            segment_duration = random.uniform(3.0, 8.0)

        if current_time + segment_duration > duration - 1:
            segment_duration = duration - current_time - 1

        segments.append({
            'speaker': current_speaker,
            'start': current_time,
            'end': current_time + segment_duration,
            'duration': segment_duration
        })

        # 다음 화자로 전환 (가끔 동일 화자 연속)
        if random.random() < 0.8:  # 80% 확률로 화자 변경
            current_speaker = (current_speaker + 1) % num_speakers

        # 자연스러운 대화 간격
        current_time += segment_duration + random.uniform(0.2, 1.0)

    # 더 자연스러운 음성 신호 생성
    for segment in segments:
        start_sample = int(segment['start'] * sample_rate)
        end_sample = int(segment['end'] * sample_rate)

        config = speaker_configs[segment['speaker'] % len(speaker_configs)]

        # 시간 벡터
        t = np.linspace(0, segment['duration'], end_sample - start_sample)

        # 기본 주파수 변화 (억양)
        pitch_variation = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)  # 천천히 변하는 억양
        base_freq = config['base_freq'] * pitch_variation

        # 포먼트가 포함된 복합 신호
        signal = (
            # 기본 주파수
                config['amplitude'] * np.sin(2 * np.pi * base_freq * t) * 0.6 +
                # 첫 번째 포먼트
                config['amplitude'] * 0.3 * np.sin(2 * np.pi * config['formant1'] * t) +
                # 두 번째 포먼트
                config['amplitude'] * 0.2 * np.sin(2 * np.pi * config['formant2'] * t) +
                # 하모닉스
                config['amplitude'] * 0.1 * np.sin(2 * np.pi * base_freq * 2 * t) +
                config['amplitude'] * 0.05 * np.sin(2 * np.pi * base_freq * 3 * t)
        )

        # AM 변조 (음성의 자연스러운 변화)
        am_freq = random.uniform(5, 15)  # 5-15Hz 변조
        am_mod = 0.8 + 0.2 * np.sin(2 * np.pi * am_freq * t)
        signal *= am_mod

        # 자연스러운 엔벨로프
        envelope = np.ones_like(signal)
        fade_samples = int(0.05 * sample_rate)  # 50ms 페이드

        if len(signal) > 2 * fade_samples:
            # 부드러운 페이드 인/아웃
            envelope[:fade_samples] = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
            envelope[-fade_samples:] = 0.5 * (1 + np.cos(np.pi * np.arange(fade_samples) / fade_samples))

        signal *= envelope

        # 배경 노이즈 추가 (더 현실적)
        noise_level = 0.015
        noise = np.random.normal(0, noise_level, len(signal))
        signal += noise

        # 간헐적인 브레스 사운드 (숨소리)
        if random.random() < 0.3 and len(signal) > sample_rate:
            breath_pos = random.randint(sample_rate // 4, len(signal) - sample_rate // 4)
            breath_length = int(0.1 * sample_rate)  # 100ms 브레스
            breath_sound = np.random.normal(0, 0.02, breath_length) * np.exp(
                -np.arange(breath_length) / (0.05 * sample_rate))
            signal[breath_pos:breath_pos + breath_length] += breath_sound[:min(breath_length, len(signal) - breath_pos)]

        # 오디오에 추가
        audio[start_sample:end_sample] = signal

    # 전체적인 배경 노이즈 (매우 약하게)
    background_noise = np.random.normal(0, 0.005, len(audio))
    audio += background_noise

    # 정규화 (피크가 0.9를 넘지 않도록)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    # 저장
    sf.write(output_path, audio, sample_rate)

    # 세그먼트 정보 저장
    info_path = output_path.replace('.wav', '_segments.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"현실적인 테스트 대화 - {num_speakers}명 화자\n")
        f.write(f"총 길이: {duration:.1f}초\n")
        f.write(f"총 세그먼트: {len(segments)}개\n\n")
        f.write("세그먼트 정보:\n")
        for i, seg in enumerate(segments):
            f.write(
                f"{i + 1:2d}. 화자_{seg['speaker']} ({seg['start']:5.1f}s - {seg['end']:5.1f}s, {seg['duration']:4.1f}s)\n")

        # 화자별 통계
        f.write(f"\n화자별 발화 시간:\n")
        for speaker_id in range(num_speakers):
            speaker_segments = [s for s in segments if s['speaker'] == speaker_id]
            total_time = sum(s['duration'] for s in speaker_segments)
            f.write(f"화자_{speaker_id}: {total_time:5.1f}초 ({len(speaker_segments)}개 구간)\n")

    print(f"✅ 현실적인 테스트 파일 생성: {output_path}")
    print(f"📄 세그먼트 정보: {info_path}")
    print(f"📊 {len(segments)}개 발화 구간, {num_speakers}명 화자")

    return segments
    """
    샘플 오디오 다운로드 가이드 출력
    """

    samples = [
        {
            "name": "짧은 대화 (2명)",
            "description": "2명이 간단한 대화",
            "duration": "1-3분",
            "difficulty": "쉬움",
            "example": "친구끼리 안부 인사"
        },
        {
            "name": "뉴스 인터뷰 (2-3명)",
            "description": "앵커 + 게스트 인터뷰",
            "duration": "5-10분",
            "difficulty": "보통",
            "example": "KBS, MBC 뉴스 인터뷰"
        },
        {
            "name": "토론 프로그램 (3-4명)",
            "description": "여러 명이 참여하는 토론",
            "duration": "10-30분",
            "difficulty": "어려움",
            "example": "100분 토론, 심야토론"
        },
        {
            "name": "팟캐스트 (2-3명)",
            "description": "진행자 + 게스트 대화",
            "duration": "20-60분",
            "difficulty": "보통",
            "example": "김어준의 뉴스공장"
        }
    ]

    print("=== 추천 테스트 오디오 ===\n")

    for i, sample in enumerate(samples, 1):
        print(f"{i}. {sample['name']}")
        print(f"   설명: {sample['description']}")
        print(f"   길이: {sample['duration']}")
        print(f"   난이도: {sample['difficulty']}")
        print(f"   예시: {sample['example']}")
        print()


def create_test_suite():
    """테스트 스위트 생성"""

    output_dir = Path("test_audio")
    output_dir.mkdir(exist_ok=True)

    test_cases = [
        {"name": "short_2speakers", "duration": 15, "speakers": 2},
        {"name": "medium_3speakers", "duration": 60, "speakers": 3},
        {"name": "long_2speakers", "duration": 180, "speakers": 2},
        {"name": "complex_4speakers", "duration": 120, "speakers": 4},
    ]

    print("🎵 테스트 스위트 생성 중...\n")

    for case in test_cases:
        output_path = output_dir / f"{case['name']}.wav"
        print(f"생성 중: {case['name']}")

        segments = generate_test_conversation(
            output_path=str(output_path),
            duration=case['duration'],
            num_speakers=case['speakers']
        )

        print(f"  - {case['speakers']}명 화자")
        print(f"  - {case['duration']}초 길이")
        print(f"  - {len(segments)}개 발화 구간\n")

    print("✅ 테스트 스위트 생성 완료!")
    print(f"📁 파일 위치: {output_dir.absolute()}")


if __name__ == "__main__":
    print("🎤 테스트 음성 파일 생성기")
    print("=" * 50)

    # 샘플 추천
    download_sample_audio()

    # 더 현실적인 테스트 파일 생성
    print("\n현실적인 테스트 파일 생성...")

    # 간단한 2명 대화 (30초)
    generate_realistic_test_conversation(
        output_path="realistic_2speakers_30s.wav",
        duration=30,
        num_speakers=2
    )

    # 복잡한 3명 대화 (60초)
    generate_realistic_test_conversation(
        output_path="realistic_3speakers_60s.wav",
        duration=60,
        num_speakers=3
    )

    print("\n✅ 모든 테스트 파일 생성 완료!")
    print("\n테스트 명령어:")
    print("python main.py single realistic_2speakers_30s.wav")
    print("python main.py single realistic_3speakers_60s.wav")