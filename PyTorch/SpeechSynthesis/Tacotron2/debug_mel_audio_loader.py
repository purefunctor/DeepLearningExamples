from waveglow.data_function import MelAudioLoader
import soundfile as sf

class args:
    max_wav_value = 32768.0
    sampling_rate = 22050
    segment_length = 4000
    filter_length = 1024
    hop_length = 256
    win_length = 1024
    n_mel_channels = 80
    mel_fmin = 0.0
    mel_fmax = 8000.0

audio_loader = MelAudioLoader("./", "sr/manifest.txt", args)
i_audio, i_mel, t_audio, t_mel = audio_loader.take_at(
    "./sr/day1_wavs/nt1_middle_far_mid_48_8.wav", 
    "./sr/day1_wavs/67_near_far_close_30_8.wav",
    offset=0,
    length=22050 * 10,
)

with sf.SoundFile("./output/input_audio.wav", "w", samplerate=22050, channels=1) as f:
    f.write(i_audio.unsqueeze(1).numpy())

with sf.SoundFile("./output/target_audio.wav", "w", samplerate=22050, channels=1) as f:
    f.write(t_audio.unsqueeze(1).numpy())
