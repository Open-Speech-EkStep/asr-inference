import configuration
from hf.infer import Wav2vecHF
from utils import wav_to_pcm16
import soundfile as sf
import vad.timestamp_generator as tg
from tqdm import tqdm
from rich.console import Console
from rich.traceback import install

install()
console = Console()


class Transcribe:
    def __init__(self, model_type, vad, wav_path, lm):
        self.model_type = model_type
        self.vad = vad
        self.wav_path = wav_path
        self.lm = lm
        console.log(f"Transcribing audio file {self.wav_path}")
        console.log(f"Using model type [green underline]{self.model_type}[/]")
        console.log(f"Use vad is set to {self.vad}")
        console.log(f"Use LM is set to {self.lm}")
        if self.model_type == "hf":
            console.log(f"Model loaded from {configuration.HF_MODEL_PATH}")
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

        console.log(f"Transcription: {text}", style="bold")

        return text


if __name__ == "__main__":

    Transcribe(
        model_type=configuration.MODEL,
        vad=configuration.USE_VAD,
        wav_path=configuration.WAV_PATH,
        lm=configuration.USE_LM,
    ).speech_to_text()
