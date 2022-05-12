import configuration
from hf.infer import Wav2vecHF

class Transcribe:
    def __init__(self, model_type, vad, wav_path, lm):
        self.model_type = model_type
        self.vad = vad
        self.wav_path = wav_path
        self.lm = lm
        if self.model_type == 'hf':
            self.model = self.hf_model(configuration.HF_MODEL_PATH)
            
    def hf_model(self, model_path):
        if self.lm:
            asr_model = Wav2vecHF(model_path, 'kenlm')
        else:
            asr_model = Wav2vecHF(model_path, 'viterbi')

        return asr_model

    def speech_to_text(self, hot_words=[]):
        if self.model_type == 'hf':
            return self.model.transcribe(self.wav_path, hotwords=hot_words)
        
if __name__ == '__main__':
    print(Transcribe(configuration.MODEL, configuration.USE_VAD, configuration.WAV_PATH, configuration.USE_LM).speech_to_text())