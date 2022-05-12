import configuration
from hf.infer import Wav2vecHF

class Transcribe:
    def __init__(self):
        self.model_type = configuration.MODEL
        self.vad = configuration.USE_VAD
        self.wav_path = configuration.WAV_PATH
        self.lm = configuration.USE_LM
        if self.model_type == 'hf':
            self.model = self.hf_model(configuration.HF_MODEL_PATH)
            
    def hf_model(self, model_path):
        if self.lm:
            asr_model = Wav2vecHF(model_path, 'kenlm')
        else:
            asr_model = Wav2vecHF(model_path, 'viterbi')

        return asr_model

    def speech_to_text(self, audio_path, hot_words=[]):
        if self.model_type == 'hf':
            return self.model.transcribe(audio_path, hotwords=hot_words)
        
if __name__ == '__main__':
    print(Transcribe().speech_to_text(configuration.WAV_PATH))