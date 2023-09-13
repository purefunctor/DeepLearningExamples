from waveglow.data_function import MelAudioLoader

class args:
    max_wav_value = 32768.0
    # FIXME: figure out what to change when increasing the sample rate
    sampling_rate = 44100
    segment_length = 4000
    filter_length = 1024
    hop_length = 256
    win_length = 1024
    n_mel_channels = 80
    mel_fmin = 0.0
    mel_fmax = 8000.0

audio_loader = MelAudioLoader("./", "sr/manifest.txt", args)
print(audio_loader[0])
