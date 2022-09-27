from pyexpat import model
import configuration
from hf.infer import Wav2vecHF
from fseq.infer import load_model_and_generator, get_results
from utils import wav_to_pcm16
import soundfile as sf
import vad.timestamp_generator as tg
from tqdm import tqdm
from rich.console import Console
from rich.traceback import install
import torch
from model_item import ModelItem
from fseq.infer import Wav2VecCtc
from glob import glob
from conformer.infer import Conformer

install()
console = Console()


class Transcribe:
    def __init__(self, model_type, vad, lm):
        self.model_type = model_type
        self.vad = vad
        self.lm = lm
        self.model_items = {}

        if torch.cuda.is_available():
            self.cuda = False
            self.half = False
        else:
            self.cuda = False
            self.half = False

        console.log(f"Using model type [green underline]{self.model_type}[/]")
        console.log(f"Use vad is set to {self.vad}")
        console.log(f"Use LM is set to {self.lm}")

        if self.model_type == "hf":
            console.log(f"Model loaded from {configuration.HF_MODEL_PATH}")
            self.model = self.hf_model(configuration.HF_MODEL_PATH)

        if self.model_type == "conformer":
            console.log(f"Model loaded from {configuration.CONFORMER_PATH}")
            self.model = self.conformer_model(configuration.CONFORMER_PATH)

        if self.model_type == "fairseq":
            model_item = ModelItem(
                configuration.FAIRSEQ_MODEL_PATH, configuration.FAIRSEQ_CHECKPOINT_NAME
            )
            console.log(
                f"Model artifacts loaded from {configuration.FAIRSEQ_MODEL_PATH}"
            )
            decoder_type = "kenlm" if self.lm else "viterbi"
            console.log(f"Decoder type is {decoder_type}")
            model, generator = load_model_and_generator(
                model_item, self.cuda, decoder=decoder_type, half=self.half
            )
            model_item.set_generator(generator)
            model_item.set_model(model)
            console.log(f":thumbs_up: Model and generator loaded successfully")
            self.model_items["fseq"] = model_item

    def hf_model(self, model_path):
        if self.lm:
            asr_model = Wav2vecHF(model_path, "kenlm")
        else:
            asr_model = Wav2vecHF(model_path, "viterbi")

        return asr_model

    def conformer_model(self, model_path):

        if self.lm:
            return Conformer(model_path, "kenlm")
        else:
            return Conformer(model_path, "viterbi")

    def speech_to_text(self, wav_path, hot_words=[]):
        text = ""

        if self.model_type == "conformer":
            text = self.model.transcribe(wav_path)


        if self.model_type == "hf":
            if self.vad:

                audio, _ = sf.read(wav_path)
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

                text = self.model.transcribe(wav_path, hotwords=hot_words)

        if self.model_type == "fairseq":
            model_item = self.model_items["fseq"]
            text = get_results(
                wav_path=wav_path,
                dict_path=model_item.get_dict_file_path(),
                generator=model_item.get_generator(),
                use_cuda=self.cuda,
                model=model_item.get_model(),
                half=self.half,
            )

        console.log(f"Transcription: {text}", style="bold")

        return text


if __name__ == "__main__":

    m = Transcribe(
        model_type=configuration.MODEL,
        vad=configuration.USE_VAD,
        lm=configuration.USE_LM,
    )

    #wav_files = glob("/home/anirudhgupta/hindi_data/yourquote_hf/*/clean/*.wav")
    wav_files = ['/home/anirudhgupta/test_hindi.wav']

    for wav in tqdm(wav_files):
        text = m.speech_to_text(wav)
        print(text)
        #with open(wav.replace(".wav", ".txt"), "w+") as f:
        #    f.write(text)
