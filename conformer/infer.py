
from nemo.utils import model_utils
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
import os
from glob import glob
import torch
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm
from rich.console import Console
from rich.traceback import install
from glob import glob

install()
console = Console()

class Conformer:
    def __init__(self, model_path, lm):
        self.model_path = model_path
        self.lm = lm
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.decoder = self.build_decoder()

    def load_model(self):
        model = glob(self.model_path + '/*.nemo')[0]
        model_cfg = ASRModel.restore_from(restore_path=model, return_config=True)
        classpath = model_cfg.target
        imported_class = model_utils.import_class_by_path(classpath)
        asr_model = imported_class.restore_from(restore_path=model, map_location=self.device)
        return asr_model

    def build_decoder(self):
        if self.lm == "viterbi":
            decoder = build_ctcdecoder(self.model.decoder.vocabulary)
        else:
            lm_path = glob(self.model_path + '/*.binary')[0]
            unigram_path = glob(self.model_path + '/*.txt')[0]
            with open(unigram_path, encoding='utf-8') as f:
                unigram_list = [t for t in f.read().strip().split('\n')]
            decoder = build_ctcdecoder(self.model.decoder.vocabulary, lm_path, unigram_list)
        return decoder
        

    def transcribe(self, wav_path):
        logits = self.model.transcribe([wav_path], logprobs=True)[0]
        return self.decoder.decode(logits)
    

if __name__ == '__main__':

    m = Conformer('/home/anirudhgupta/asr-inference/models/conformer/hindi','viterbi').transcribe('/home/anirudhgupta/test_hindi.wav')
    print(m)
