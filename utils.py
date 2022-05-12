import numpy as np

def wav_to_pcm16(wav):
    ints = (wav * 32768).astype(np.int16)
    little_endian = ints.astype('<u2')
    wav_bytes = little_endian.tobytes()
    return wav_bytes
