import os


def convert_audio_to_wav(file_path, output_path):
    cmd = (
        "ffmpeg -i "
        + str(file_path)
        + " -ar 16000 -ac 1 -bits_per_raw_sample 16 -vn "
        + str(output_path)
    )
    os.system(cmd)


if __name__ == "__main__":
    convert_audio_to_wav(
        "/Users/anirudhgupta/Downloads/market_mantra_4_may.mp4",
        "/Users/anirudhgupta/ekstepspeechrecognition/vakyansh-hf/market_mantra_4_may.wav",
    )
