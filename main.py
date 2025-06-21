# main_complete.py
"""
화자 분리 음성 인식 통합 시스템 (완전 버전)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import json
from datetime import datetime
import traceback

# 로컬 모듈 import
from config import (
    model_config, audio_config, processing_config, system_config,
    update_config_for_gpu
)
from audio_utils import AudioProcessor, VADProcessor
from speaker_diarization import create_speaker_diarizer, post_process_speaker_labels
from speech_recognition import create_asr_model, ASRPostProcessor
from system_monitor import ResourceMonitor, print_system_info, PerformanceAnalyzer


class SpeakerDiarizationSystem:
    """통합 화자 분리 음성 인식 시스템"""

    def __init__(self, config_override: Dict = None):
        """시스템 초기화"""

        print("🚀 화자 분리 음성 인식 시스템 초기화")

        # GPU 최적화 설정 적용
        update_config_for_gpu()

        # 설정 오버라이드 적용
        if config_override:
            self._apply_config_override(config_override)

        # 모듈 초기화
        self.audio_processor = AudioProcessor()
        self.vad_processor = VADProcessor()
        self.asr_post_processor = ASRPostProcessor()

        # 모델들 (지연 로딩)
        self.speaker_diarizer = None
        self.asr_model = None

        print("✅ 시스템 초기화 완료")

    def _apply_config_override(self, override: Dict):
        """설정 오버라이드 적용"""

        if 'whisper_model' in override:
            model_config.whisper_model_name = override['whisper_model']

        if 'batch_size' in override:
            processing_config.batch_size = override['batch_size']

        if 'chunk_duration' in override:
            processing_config.chunk_duration = override['chunk_duration']

    def load_models(self, force_reload: bool = False):
        """모델 로드"""

        if self.speaker_diarizer is None or force_reload:
            print("📡 화자 분리 모델 로딩...")
            with ResourceMonitor("speaker_diarizer_loading"):
                self.speaker_diarizer = create_speaker_diarizer("auto")

        if self.asr_model is None or force_reload:
            print("🎤 음성 인식 모델 로딩...")
            with ResourceMonitor("asr_model_loading"):
                self.asr_model = create_asr_model("auto")

    def process_audio_file(self, audio_path: str,
                           output_dir: str = None,
                           n_speakers: int = None) -> Dict:
        """오디오 파일 전체 처리"""

        print(f"📁 처리 시작: {audio_path}")

        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = Path(audio_path).parent / "output"

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        try:
            # 모델 로드
            self.load_models()

            with ResourceMonitor("full_audio_processing") as monitor:

                # 1. 오디오 로드 및 전처리
                print("🔊 오디오 로드 및 전처리")
                audio, sr = self.audio_processor.load_audio(audio_path)

                # 오디오 품질 체크
                quality = self.audio_processor.check_audio_quality(audio, sr)
                print(f"오디오 품질: {'✅ 양호' if quality['overall_ok'] else '⚠️ 주의'}")

                # 오디오 정규화
                audio = self.audio_processor.normalize_audio(audio)

                # 2. 음성 활동 감지 (VAD)
                print("🎯 음성 활동 감지")
                segments = self.vad_processor.enhanced_vad(audio, sr)
                segments = self.vad_processor.merge_close_segments(segments)

                print(f"감지된 음성 구간: {len(segments)}개")

                if not segments:
                    return {
                        'success': False,
                        'error': '음성 구간이 감지되지 않았습니다.',
                        'audio_quality': quality
                    }

                # 3. 화자 분리
                print("👥 화자 분리")
                speaker_labels = self.speaker_diarizer.diarize(audio, sr, segments)

                # 후처리
                speaker_labels = post_process_speaker_labels(speaker_labels, segments)

                unique_speakers = len(set(speaker_labels))
                print(f"식별된 화자 수: {unique_speakers}명")

                # 4. 음성 인식
                print("🗣️ 음성 인식")
                transcription_results = self.asr_model.transcribe_segments(audio, segments, sr)

                # 5. 결과 통합
                print("🔄 결과 통합 및 후처리")
                final_results = self._integrate_results(
                    transcription_results, speaker_labels, segments, quality
                )

                # 6. 결과 저장
                print("💾 결과 저장")
                output_files = self._save_results(final_results, output_dir, audio_path)

                # 처리 완료
                processing_info = monitor.get_resource_usage()

                # 안전한 통계 계산
                total_duration = len(audio) / sr if audio is not None and sr > 0 else 0

                return {
                    'success': True,
                    'num_segments': len(final_results),
                    'num_speakers': unique_speakers,
                    'audio_duration': total_duration,
                    'output_files': output_files,
                    'audio_quality': quality,
                    'processing_info': processing_info
                }

        except Exception as e:
            print(f"❌ 처리 중 오류 발생: {e}")
            return {
                'success': False,
                'error': str(e),
                'audio_quality': quality if 'quality' in locals() else {}
            }

    def process_batch(self, audio_files: List[str],
                      output_dir: str = None) -> List[Dict]:
        """배치 파일 처리"""

        print(f"📦 배치 처리 시작: {len(audio_files)}개 파일")

        if output_dir is None:
            output_dir = Path("./batch_output")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 모델 한 번만 로드
        try:
            self.load_models()
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return [{
                'success': False,
                'error': f'모델 로드 실패: {e}',
                'file_path': 'ALL'
            }]

        results = []

        for i, audio_file in enumerate(audio_files):
            print(f"\n[{i + 1}/{len(audio_files)}] {audio_file}")

            try:
                # 파일 존재 여부 확인
                if not os.path.exists(audio_file):
                    print(f"❌ 파일을 찾을 수 없음: {audio_file}")
                    results.append({
                        'success': False,
                        'file_path': audio_file,
                        'error': '파일을 찾을 수 없음'
                    })
                    continue

                # 개별 파일 처리
                file_output_dir = output_dir / Path(audio_file).stem
                result = self.process_audio_file(audio_file, file_output_dir)
                result['file_path'] = audio_file
                results.append(result)

            except Exception as e:
                print(f"❌ 파일 처리 실패: {e}")
                results.append({
                    'success': False,
                    'file_path': audio_file,
                    'error': str(e)
                })

                # 메모리 정리 (오류 후)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

        # 배치 결과 요약 저장
        try:
            self._save_batch_summary(results, output_dir)
        except Exception as e:
            print(f"⚠️ 배치 요약 저장 실패: {e}")

        return results

    def _integrate_results(self, transcriptions: List[Dict],
                           speaker_labels: List[int],
                           segments: List[tuple],
                           audio_quality: Dict) -> List[Dict]:
        """결과 통합 및 후처리"""

        integrated_results = []

        # 전사 결과와 화자 라벨 매칭
        for i, transcription in enumerate(transcriptions):
            segment_id = transcription.get('segment_id', i)

            # 안전한 인덱스 접근
            if segment_id < len(speaker_labels) and segment_id < len(segments):
                speaker_label = speaker_labels[segment_id]

                # 텍스트 후처리
                cleaned_text = self.asr_post_processor.clean_transcription(
                    transcription['text']
                )

                if cleaned_text:  # 빈 텍스트가 아닌 경우만

                    # 신뢰도 점수 계산
                    confidence = self.asr_post_processor.calculate_confidence_score(
                        cleaned_text, audio_quality
                    )

                    integrated_results.append({
                        'segment_id': segment_id,
                        'speaker_id': f"화자_{speaker_label}",
                        'start_time': transcription.get('start_time', 0),
                        'end_time': transcription.get('end_time', 0),
                        'duration': transcription.get('duration', 0),
                        'text': cleaned_text,
                        'original_text': transcription['text'],
                        'confidence': confidence
                    })

        # 짧은 세그먼트 병합 (안전하게 처리)
        try:
            integrated_results = self.asr_post_processor.merge_short_segments(integrated_results)
        except Exception as e:
            print(f"⚠️ 세그먼트 병합 중 오류: {e}")

        return integrated_results

    def _save_results(self, results: List[Dict],
                      output_dir: Path,
                      audio_path: str) -> Dict[str, str]:
        """결과를 다양한 형식으로 저장"""

        output_files = {}
        base_name = Path(audio_path).stem

        try:
            # 1. CSV 형식
            csv_path = output_dir / f"{base_name}_transcript.csv"
            if results:  # 결과가 있는 경우만 저장
                df = pd.DataFrame(results)
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                output_files['csv'] = str(csv_path)

            # 2. JSON 형식
            json_path = output_dir / f"{base_name}_transcript.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            output_files['json'] = str(json_path)

            # 3. 텍스트 형식 (읽기 쉬운 대화록)
            txt_path = output_dir / f"{base_name}_conversation.txt"
            formatted_text = self._format_conversation_text(results)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            output_files['txt'] = str(txt_path)

            # 4. SRT 자막 형식
            if results:  # 결과가 있는 경우만 SRT 생성
                srt_path = output_dir / f"{base_name}_subtitles.srt"
                srt_content = self._format_srt_subtitles(results)
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
                output_files['srt'] = str(srt_path)

        except Exception as e:
            print(f"⚠️ 파일 저장 중 오류: {e}")
            # 최소한 JSON은 저장 시도
            try:
                json_path = output_dir / f"{base_name}_transcript.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                output_files['json'] = str(json_path)
            except:
                pass

        return output_files

    def _format_conversation_text(self, results: List[Dict]) -> str:
        """대화 형식으로 포맷팅"""

        formatted_lines = []
        formatted_lines.append("=== 대화 전사 결과 ===\n")

        if not results:
            formatted_lines.append("인식된 음성이 없습니다.\n")
        else:
            for result in results:
                speaker = result.get('speaker_id', '알 수 없음')
                start_time = result.get('start_time', 0)
                text = result.get('text', '')
                confidence = result.get('confidence', 0)

                # 시간 포맷팅 (MM:SS)
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"

                # 신뢰도 표시
                confidence_icon = "🟢" if confidence > 0.9 else "🟡" if confidence > 0.7 else "🔴"

                formatted_lines.append(f"[{time_str}] {speaker}: {text} {confidence_icon}")

        # 통계 정보 추가
        if results:
            unique_speakers = len(set(r.get('speaker_id', '') for r in results))
            total_duration = max((r.get('end_time', 0) for r in results), default=0)
        else:
            unique_speakers = 0
            total_duration = 0

        formatted_lines.append(f"\n=== 통계 ===")
        formatted_lines.append(f"총 화자 수: {unique_speakers}명")
        formatted_lines.append(f"총 길이: {total_duration:.1f}초")
        formatted_lines.append(f"발화 구간: {len(results)}개")
        formatted_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(formatted_lines)

    def _format_srt_subtitles(self, results: List[Dict]) -> str:
        """SRT 자막 형식으로 포맷팅"""

        if not results:
            return "1\n00:00:00,000 --> 00:00:01,000\n인식된 음성이 없습니다.\n\n"

        srt_lines = []

        for i, result in enumerate(results, 1):
            start_time = result.get('start_time', 0)
            end_time = result.get('end_time', start_time + 1)
            speaker = result.get('speaker_id', '알 수 없음')
            text = result.get('text', '')

            # SRT 시간 형식: HH:MM:SS,mmm
            start_srt = self._seconds_to_srt_time(start_time)
            end_srt = self._seconds_to_srt_time(end_time)

            srt_lines.append(str(i))
            srt_lines.append(f"{start_srt} --> {end_srt}")
            srt_lines.append(f"{speaker}: {text}")
            srt_lines.append("")  # 빈 줄

        return "\n".join(srt_lines)

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """초를 SRT 시간 형식으로 변환"""

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def _save_batch_summary(self, results: List[Dict], output_dir: Path):
        """배치 처리 결과 요약 저장"""

        try:
            summary = {
                'processed_files': len(results),
                'successful': len([r for r in results if r.get('success', False)]),
                'failed': len([r for r in results if not r.get('success', False)]),
                'total_duration': sum(r.get('audio_duration', 0) for r in results if r.get('success', False)),
                'total_speakers': sum(r.get('num_speakers', 0) for r in results if r.get('success', False)),
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'details': results
            }

            summary_path = output_dir / "batch_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

            print(f"📊 배치 요약 저장: {summary_path}")
            print(f"성공: {summary['successful']}/{summary['processed_files']}")

        except Exception as e:
            print(f"⚠️ 배치 요약 저장 중 오류: {e}")
            # 최소한의 요약이라도 출력
            successful = len([r for r in results if r.get('success', False)])
            total = len(results)
            print(f"배치 처리 완료: {successful}/{total} 성공")


def create_cli_parser():
    """CLI 인터페이스 생성"""

    parser = argparse.ArgumentParser(
        description="화자 분리 음성 인식 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 파일 처리
  python main.py single audio.wav

  # 배치 처리
  python main.py batch *.wav

  # 출력 디렉토리 지정
  python main.py single audio.wav --output ./results

  # 모델 지정
  python main.py single audio.wav --model whisper-medium

  # 시스템 정보 확인
  python main.py info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')

    # 단일 파일 처리
    single_parser = subparsers.add_parser('single', help='단일 오디오 파일 처리')
    single_parser.add_argument('audio_file', help='처리할 오디오 파일')
    single_parser.add_argument('--output', '-o', help='출력 디렉토리')
    single_parser.add_argument('--speakers', '-s', type=int, help='예상 화자 수')
    single_parser.add_argument('--model', '-m', help='사용할 Whisper 모델')
    single_parser.add_argument('--batch-size', type=int, help='배치 크기')

    # 배치 처리
    batch_parser = subparsers.add_parser('batch', help='여러 오디오 파일 배치 처리')
    batch_parser.add_argument('audio_files', nargs='+', help='처리할 오디오 파일들')
    batch_parser.add_argument('--output', '-o', help='출력 디렉토리')
    batch_parser.add_argument('--model', '-m', help='사용할 Whisper 모델')
    batch_parser.add_argument('--batch-size', type=int, help='배치 크기')

    # 시스템 정보
    info_parser = subparsers.add_parser('info', help='시스템 정보 및 권장 설정 확인')
    info_parser.add_argument('--benchmark', action='store_true', help='성능 벤치마크 실행')

    # 대화형 모드
    interactive_parser = subparsers.add_parser('interactive', help='대화형 모드')

    return parser


def main():
    """메인 함수"""

    parser = create_cli_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # 시스템 정보 명령
    if args.command == 'info':
        print_system_info()

        if hasattr(args, 'benchmark') and args.benchmark:
            print("\n🔬 성능 벤치마크 실행...")
            try:
                analyzer = PerformanceAnalyzer()

                # 모델 로딩 벤치마크
                models = ['openai/whisper-base', 'openai/whisper-medium']
                benchmark_results = analyzer.benchmark_model_loading(models)

                print("\n=== 모델 로딩 성능 ===")
                for model, result in benchmark_results.items():
                    if result['success']:
                        print(f"{model}:")
                        print(f"  로딩 시간: {result['load_time_seconds']:.2f}초")
                        print(f"  메모리 사용: {result['memory_usage_gb']:.2f}GB")
                    else:
                        print(f"{model}: ❌ 실패 - {result['error']}")
            except Exception as e:
                print(f"⚠️ 벤치마크 실행 중 오류: {e}")

        return 0

    # 설정 오버라이드 준비
    config_override = {}

    if hasattr(args, 'model') and args.model:
        # 모델 이름 정규화
        model_name = args.model.lower()
        if not model_name.startswith('openai/whisper-'):
            model_name = f"openai/whisper-{model_name}"
        config_override['whisper_model'] = model_name

    if hasattr(args, 'batch_size') and args.batch_size:
        config_override['batch_size'] = args.batch_size

    # 시스템 초기화
    try:
        system = SpeakerDiarizationSystem(config_override)
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        print("의존성 패키지가 모두 설치되어 있는지 확인하세요.")
        return 1

    # 명령별 처리
    try:
        if args.command == 'single':
            # 단일 파일 처리
            if not os.path.exists(args.audio_file):
                print(f"❌ 파일을 찾을 수 없습니다: {args.audio_file}")
                return 1

            result = system.process_audio_file(
                args.audio_file,
                getattr(args, 'output', None),
                getattr(args, 'speakers', None)
            )

            if result['success']:
                print(f"\n✅ 처리 완료!")
                print(f"  🎯 음성 구간: {result['num_segments']}개")
                print(f"  👥 화자 수: {result['num_speakers']}명")
                print(f"  ⏱️ 오디오 길이: {result['audio_duration']:.1f}초")
                print(f"  📁 출력 파일:")
                for format_type, file_path in result['output_files'].items():
                    print(f"    {format_type.upper()}: {file_path}")
            else:
                print(f"❌ 처리 실패: {result.get('error', '알 수 없는 오류')}")
                return 1

        elif args.command == 'batch':
            # 배치 처리
            # 파일 존재 확인
            valid_files = []
            for file_path in args.audio_files:
                if os.path.exists(file_path):
                    valid_files.append(file_path)
                else:
                    print(f"⚠️ 파일을 찾을 수 없음: {file_path}")

            if not valid_files:
                print("❌ 처리할 유효한 파일이 없습니다.")
                return 1

            results = system.process_batch(valid_files, getattr(args, 'output', None))

            successful = len([r for r in results if r.get('success', False)])
            total = len(results)

            print(f"\n📦 배치 처리 완료: {successful}/{total} 성공")

            # 실패한 파일들 표시
            failed_files = [r for r in results if not r.get('success', False)]
            if failed_files:
                print("❌ 실패한 파일들:")
                for result in failed_files:
                    print(f"  {result['file_path']}: {result.get('error', '알 수 없는 오류')}")

        elif args.command == 'interactive':
            # 대화형 모드
            interactive_mode(system)

    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
        return 1
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        traceback.print_exc()
        return 1

    return 0


def interactive_mode(system: SpeakerDiarizationSystem):
    """대화형 모드"""

    print("\n🎯 대화형 모드 시작")
    print("명령어:")
    print("  process <파일경로>  - 파일 처리")
    print("  info               - 시스템 정보")
    print("  config             - 현재 설정 확인")
    print("  help               - 도움말")
    print("  exit               - 종료")

    while True:
        try:
            command = input("\n> ").strip().split()

            if not command:
                continue

            cmd = command[0].lower()

            if cmd in ['exit', 'quit']:
                print("👋 종료합니다.")
                break

            elif cmd == 'help':
                print("사용 가능한 명령어:")
                print("  process <파일경로>  - 오디오 파일 처리")
                print("  info               - 시스템 정보 출력")
                print("  config             - 현재 설정 확인")
                print("  help               - 이 도움말")
                print("  exit               - 프로그램 종료")

            elif cmd == 'info':
                print_system_info()

            elif cmd == 'config':
                print("=== 현재 설정 ===")
                print(f"Whisper 모델: {model_config.whisper_model_name}")
                print(f"배치 크기: {processing_config.batch_size}")
                print(f"청크 길이: {processing_config.chunk_duration}초")
                print(f"디바이스: {system_config.device}")
                print(f"FP16 사용: {processing_config.use_fp16}")

            elif cmd == 'process':
                if len(command) < 2:
                    print("❌ 파일 경로를 입력하세요. 예: process audio.wav")
                    continue

                file_path = command[1]

                if not os.path.exists(file_path):
                    print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
                    continue

                print(f"🚀 처리 시작: {file_path}")

                try:
                    result = system.process_audio_file(file_path)

                    if result['success']:
                        print(f"✅ 처리 완료!")
                        print(f"  화자 수: {result['num_speakers']}명")
                        print(f"  구간 수: {result['num_segments']}개")
                        print(f"  결과 파일: {result['output_files'].get('txt', 'N/A')}")
                    else:
                        print(f"❌ 처리 실패: {result.get('error', '알 수 없는 오류')}")

                except Exception as e:
                    print(f"❌ 처리 중 오류: {e}")

            else:
                print(f"❌ 알 수 없는 명령어: {cmd}")
                print("help를 입력하여 사용 가능한 명령어를 확인하세요.")

        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


# 유틸리티 함수들
def check_dependencies():
    """필수 의존성 확인"""

    required_packages = [
        'torch', 'transformers', 'librosa', 'soundfile',
        'pandas', 'numpy', 'sklearn'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ 다음 패키지들이 설치되지 않았습니다:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n설치 명령어:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def setup_logging():
    """로깅 설정"""

    import logging

    logging.basicConfig(
        level=getattr(logging, system_config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('speaker_diarization.log'),
            logging.StreamHandler()
        ]
    )


def show_welcome_message():
    """환영 메시지 출력"""

    print("=" * 60)
    print("🎤 화자 분리 음성 인식 시스템")
    print("RTX 4060 최적화 한국어 AI 음성 분석")
    print("=" * 60)
    print()


if __name__ == "__main__":
    # 환영 메시지
    show_welcome_message()

    # 의존성 확인
    if not check_dependencies():
        sys.exit(1)

    # 로깅 설정
    setup_logging()

    # 메인 실행
    sys.exit(main())