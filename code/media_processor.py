import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MediaProcessor:
    """
    Processes image and voice-note media files with multi-level fallbacks.
    Never raises an exception; always returns a safe text representation.
    """
    def __init__(self):
        self._easyocr_reader = None
        self._easyocr_attempted = False

    def _get_easyocr_reader(self):
        if not self._easyocr_attempted:
            self._easyocr_attempted = True
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
                self._easyocr_reader = None
        return self._easyocr_reader

    def extract_image_text(self, image_path: str, existing_text: str = "") -> str:
        """
        Extracts OCR text from an image file. Fallback to existing_text if image is missing/unreadable.
        """
        if not image_path or not os.path.exists(image_path):
            return existing_text

        extracted = ""
        # Try EasyOCR
        reader = self._get_easyocr_reader()
        if reader:
            try:
                res = reader.readtext(image_path, detail=0)
                extracted = " ".join(res).strip()
            except Exception as e:
                logger.warning(f"EasyOCR error on {image_path}: {e}")

        # Combine extracted OCR with existing message text
        if extracted and existing_text:
            return f"{existing_text} [OCR: {extracted}]"
        elif extracted:
            return extracted
        elif existing_text:
            return existing_text
        return f"[Image Media: {os.path.basename(image_path)}]"

    def transcribe_voice_note(self, audio_path: str, existing_text: str = "") -> str:
        """
        Transcribes a voice note MP3 file. Fallback to placeholder if missing/unreadable.
        """
        if existing_text and existing_text.strip():
            return existing_text

        if not audio_path or not os.path.exists(audio_path):
            return "[Audio Voice Note: Unavailable]"

        transcription = ""
        # Try speech recognition or whisper if available
        try:
            import speech_recognition as sr
            from pydub import AudioSegment
            import tempfile
            
            sound = AudioSegment.from_file(audio_path)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                wav_path = tmp.name
                sound.export(wav_path, format="wav")
            
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = r.record(source)
                transcription = r.recognize_google(audio_data)
            
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as e:
            logger.warning(f"Speech recognition fallback on {audio_path}: {e}")

        if transcription:
            return transcription
        return f"[Voice Note Audio: {os.path.basename(audio_path)}]"

if __name__ == "__main__":
    processor = MediaProcessor()
    print("MediaProcessor initialized.")
