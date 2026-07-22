import os
from src.speech import EgyptianASR, OfflineTTS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_tests():
    logger.info("--- Starting Speech Processing Tests ---")
    
    # 1. Test Offline TTS
    tts = OfflineTTS()
    
    #test_text = "أهلاً بك في شركة المصرية للاتصالات. كيف يمكنني مساعدتك اليوم؟"
    test_text = "ضحى وصفاو أحمد أحلى أخوات في الدنيا "

    output_audio = "test_output.wav"
    
    logger.info("Synthesizing test audio...")
    tts.synthesize(test_text, output_audio)
    
    if os.path.exists(output_audio):
        logger.info(f"Success! Audio saved to {output_audio}")
    else:
        logger.error("Failed to generate audio file!")
        return

    # 2. Test ASR
    asr = EgyptianASR(model_size="small")
    
    logger.info(f"Transcribing generated audio file '{output_audio}'...")
    transcript = asr.transcribe(output_audio)
    
    logger.info("--- ASR Result ---")
    logger.info(f"Original Text: {test_text}")
    logger.info(f"Transcribed Text: {transcript}")
    logger.info("------------------")

if __name__ == "__main__":
    run_tests()
