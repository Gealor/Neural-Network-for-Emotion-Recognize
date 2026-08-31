from prepare_data.pipelines.audio.pipeline import AudioPipeline

# Расширения файлов, которые умеет обрабатывать аудио-пайплайн. Контракт модальности:
# должно соответствовать CorpusReader.file_glob у аудио-корпусов.
AUDIO_EXTENSIONS: tuple[str, ...] = (".wav",)

__all__ = ["AudioPipeline", "AUDIO_EXTENSIONS"]
