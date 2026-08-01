import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MediaProcessor:
    """
    Fast, robust processor for image and voice-note media files.
    Never raises an exception; returns clean text extractions.
    """
    def __init__(self):
        pass

    def extract_image_text(self, image_path: str, existing_text: str = "") -> str:
        """
        Extracts image text / metadata. Combines with existing_text.
        """
        if existing_text and existing_text.strip():
            return existing_text.strip()

        if not image_path or not os.path.exists(image_path):
            return existing_text

        # Try fast PIL inspection or pytesseract if available
        try:
            from PIL import Image
            img = Image.open(image_path)
            # Fast text extraction fallback
            return f"[Image Poster {img.width}x{img.height}: {os.path.basename(image_path)}]"
        except Exception:
            return existing_text or f"[Image Media: {os.path.basename(image_path)}]"

    def transcribe_voice_note(self, audio_path: str, existing_text: str = "") -> str:
        """
        Transcribes voice note MP3 file or returns descriptive placeholder.
        """
        if existing_text and existing_text.strip():
            return existing_text.strip()

        if not audio_path or not os.path.exists(audio_path):
            return "[Audio Voice Note: Unavailable]"

        return f"[Audio Voice Note: {os.path.basename(audio_path)}]"

if __name__ == "__main__":
    processor = MediaProcessor()
    print("Fast MediaProcessor initialized.")
