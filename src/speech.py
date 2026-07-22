import os
import subprocess
import urllib.request
from faster_whisper import WhisperModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EgyptianASR:
    """
    ASR module using faster-whisper.
    Handles Egyptian dialect (and general Arabic) + English code-switching.
    """
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        logger.info(f"Loading Whisper model '{model_size}' on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes the given audio file.
        Forces the language to Arabic to better handle the dialect.
        """
        logger.info(f"Transcribing audio file: {audio_path}")
        segments, info = self.model.transcribe(audio_path, language="ar", beam_size=5)
        
        logger.info(f"Detected language '{info.language}' with probability {info.language_probability}")
        
        transcript = ""
        for segment in segments:
            transcript += segment.text + " "
        
        return transcript.strip()


class OfflineTTS:
    """
    Offline Text-to-Speech module using piper-tts.
    Downloads and uses an Arabic voice model by default.
    """
    def __init__(self, model_dir="models", voice="ar_JO-kareem-low"):
        self.model_dir = model_dir
        self.voice = voice
        self.model_path = os.path.join(model_dir, f"{voice}.onnx")
        self.config_path = os.path.join(model_dir, f"{voice}.onnx.json")
        
        os.makedirs(model_dir, exist_ok=True)
        self._ensure_model_exists()

    def _ensure_model_exists(self):
        """Downloads the piper model and config if they don't exist."""
        # Piper voices URL structure depends on the voice name, e.g., ar_JO/kareem/low/ar_JO-kareem-low
        # Assuming format: language_region-name-quality
        parts = self.voice.split('-')
        if len(parts) == 3:
            lang_region, name, quality = parts
            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_region.split('_')[0]}/{lang_region}/{name}/{quality}/{self.voice}"
        else:
            # Fallback to kareem if parsing fails
            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/ar/ar_JO/kareem/low/{self.voice}"
        
        if not os.path.exists(self.model_path):
            logger.info(f"Downloading Piper TTS model: {self.voice}.onnx")
            urllib.request.urlretrieve(f"{base_url}.onnx", self.model_path)
        
        if not os.path.exists(self.config_path):
            logger.info(f"Downloading Piper TTS config: {self.voice}.onnx.json")
            urllib.request.urlretrieve(f"{base_url}.onnx.json", self.config_path)

    def synthesize(self, text: str, output_path: str, speed: float = 0.75):
        """
        Synthesizes speech from text and saves to output_path.
        Uses the piper command line tool which is installed via pip.
        'speed' parameter adjusts voice speed (lower is faster).
        """
        import sys
        import shutil
        logger.info(f"Synthesizing text to {output_path} with speed={speed}")
        
        # Find piper executable in the same dir as the current python interpreter (i.e. inside venv/bin)
        piper_exe = os.path.join(os.path.dirname(sys.executable), "piper")
        if not os.path.exists(piper_exe):
            # Fallback to searching in PATH
            piper_exe = shutil.which("piper")
            
        if not piper_exe:
            raise FileNotFoundError("Could not find 'piper' executable. Make sure piper-tts is installed.")
            
        command = [
            piper_exe,
            "--model", self.model_path,
            "--output_file", output_path,
            "--length_scale", str(speed)
        ]
        
        try:
            # Piper takes input text from standard input
            process = subprocess.run(
                command,
                input=text,
                text=True,
                capture_output=True,
                check=True
            )
            logger.info(f"Audio successfully saved to {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Piper TTS failed: {e.stderr}")
            raise
