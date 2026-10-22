from app.services.audio_extractor import extract_audio, cleanup_temp_file, get_wav_duration
from app.services.text_normalizer import (
    remove_accents,
    clean_punctuation,
    replace_spanish_number_words,
    normalize_text_for_matching,
    build_normalized_token_stream
)
from app.services.marker_detector import (
    detect_markers_from_tokens,
    detect_markers_from_plain_text
)
from app.services.transcriber import (
    get_whisper_model,
    transcribe_audio_sync,
    transcribe_audio_async
)
from app.services.clap_detector import (
    detect_claps_for_markers,
    detect_clap_in_signal
)

__all__ = [
    "extract_audio",
    "cleanup_temp_file",
    "get_wav_duration",
    "remove_accents",
    "clean_punctuation",
    "replace_spanish_number_words",
    "normalize_text_for_matching",
    "build_normalized_token_stream",
    "detect_markers_from_tokens",
    "detect_markers_from_plain_text",
    "get_whisper_model",
    "transcribe_audio_sync",
    "transcribe_audio_async",
    "detect_claps_for_markers",
    "detect_clap_in_signal",
]
