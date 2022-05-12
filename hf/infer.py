import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
import glob
import soundfile as sf
from tqdm import tqdm
from rich.console import Console
from rich.traceback import install

install()
console = Console()


class Wav2vecHF:
    def __init__(self, model_path, lm):
        self.model_path = model_path
        self.lm = lm
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor, self.model = self.load_model()
        console.log(f":thumbs_up: Model loaded successfully to device {self.device}")

    def load_model(self):
        model = Wav2Vec2ForCTC.from_pretrained(self.model_path)
        if self.lm == "viterbi":
            processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        elif self.lm == "kenlm":
            processor = Wav2Vec2ProcessorWithLM.from_pretrained(self.model_path)
        else:
            print("Chose either viterbi or kenlm for decoding")
            exit()

        return processor, model.to(self.device)

    def transcribe(self, wav_path, hotwords=[], return_timestamps=False, mode="file"):

        audio_input = self.audio_features(mode, wav_path)

        inputs = self.processor(
            audio_input, sampling_rate=16_000, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            logits = self.model(**inputs.to(self.device)).logits

        if self.lm == "viterbi":
            text = self.processor.batch_decode(
                torch.argmax(logits, dim=-1), skip_special_tokens=True
            )

        elif self.lm == "kenlm":
            result = self.processor.batch_decode(
                logits.cpu().numpy(),
                hotwords=hotwords,
                output_word_offsets=return_timestamps,
            )
            text = result.text

            if return_timestamps:
                time_offset = (
                    self.model.config.inputs_to_logits_ratio
                    / self.processor.feature_extractor.sampling_rate
                )
                word_offsets = [
                    {
                        "word": d["word"],
                        "start_time": round(d["start_offset"] * time_offset, 2),
                        "end_time": round(d["end_offset"] * time_offset, 2),
                    }
                    for d in result.word_offsets[0]
                ]

                return text[0], word_offsets

        return text[0]

    def audio_features(self, mode, wav_path):
        if mode == "file":
            audio_input, _ = sf.read(wav_path)

        elif mode == "numpy":
            audio_input = wav_path
        return audio_input

    def transcribe_dir(self, audio_dir, hotwords=[], return_timestamps=False):
        wav_files = glob.glob(audio_dir + "/*.wav")
        for wav_file in tqdm(wav_files):
            text = self.transcribe(wav_file, hotwords, return_timestamps)
            filename = wav_file.split("/")[-1].replace(".wav", ".txt")
            with open(audio_dir + "/" + filename, mode="w+", encoding="utf-8") as ofile:
                ofile.write(text)


if __name__ == "__main__":
    model_lm = Wav2vecHF("wav2vec2_to_hf/hindi_hf", "kenlm")
    model_v = Wav2vecHF("wav2vec2_to_hf/hindi_hf", "viterbi")
    print(model_lm.transcribe("ahd_00026_long_00144_hin_0011.wav"))
    print(model_v.transcribe("ahd_00026_long_00144_hin_0011.wav"))
