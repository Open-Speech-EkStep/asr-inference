import configuration
from hf.infer import Wav2vecHF
from utils import wav_to_pcm16
import soundfile as sf
import vad.timestamp_generator as tg
from tqdm import tqdm


class Transcribe:
    def __init__(self, model_type, vad, wav_path, lm):
        self.model_type = model_type
        self.vad = vad
        self.wav_path = wav_path
        self.lm = lm
        if self.model_type == "hf":
            self.model = self.hf_model(configuration.HF_MODEL_PATH)

    def hf_model(self, model_path):
        if self.lm:
            asr_model = Wav2vecHF(model_path, "kenlm")
        else:
            asr_model = Wav2vecHF(model_path, "viterbi")

        return asr_model

    def speech_to_text(self, hot_words=[]):
        text = ""
        if self.model_type == "hf":
            if self.vad:

                audio, _ = sf.read(self.wav_path)
                audio_bytes = wav_to_pcm16(audio)
                start_times, end_times = tg.extract_time_stamps(audio_bytes)
                for i, _ in tqdm(enumerate(start_times), total=len(start_times)):
                    samples = audio[
                        int(start_times[i] * 16000) : int(end_times[i] * 16000)
                    ]
                    text = text + self.model.transcribe(
                        samples, mode="numpy", hotwords=hot_words
                    )

            else:

                text = self.model.transcribe(self.wav_path, hotwords=hot_words)

        return text


if __name__ == "__main__":
    print(
        Transcribe(
            configuration.MODEL,
            configuration.USE_VAD,
            configuration.WAV_PATH,
            configuration.USE_LM,
        ).speech_to_text()
    )
