import argparse
# from inference import load_and_setup_model
from waveglow.data_function import MelAudioLoader
from scipy.io.wavfile import write
import soundfile as sf
import torch


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument(
        "--tacotron2", type=str, help="full path to the Tacotron2 model checkpoint file"
    )
    parser.add_argument(
        "--waveglow", type=str, help="full path to the WaveGlow model checkpoint file"
    )
    parser.add_argument("-s", "--sigma-infer", default=0.6, type=float)
    parser.add_argument("-d", "--denoising-strength", default=0.01, type=float)
    parser.add_argument(
        "-sr", "--sampling-rate", default=22050, type=int, help="Sampling rate"
    )

    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--fp16", action="store_true", help="Run inference with FP16")
    run_mode.add_argument("--cpu", action="store_true", help="Run inference on CPU")

    parser.add_argument(
        "--log-file", type=str, default="nvlog.json", help="Filename for logging"
    )
    parser.add_argument(
        "--stft-hop-length",
        type=int,
        default=256,
        help="STFT hop length for estimating audio length from mel size",
    )
    parser.add_argument(
        "--num-iters", type=int, default=10, help="Number of iterations"
    )
    parser.add_argument(
        "-il", "--input-length", type=int, default=64, help="Input length"
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Batch size")

    return parser


parser = argparse.ArgumentParser(description="PyTorch WaveGlow Inference")
parser = parse_args(parser)

print("Setting up model...")
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
model = model.remove_weightnorm(model)
model.forward = model.infer
# model = load_and_setup_model(
#     "WaveGlow",
#     parser,
#     checkpoint="./output/checkpoint_WaveGlow_1500.pt",
#     cpu_run=False,
#     fp16_run=False,
#     forward_is_infer=True,
#     jittable=False,
# )


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

print("Loading data...")
i_audio, i_mel, t_audio, t_mel = audio_loader.take_at(
    "./sr/day1_wavs/nt1_middle_far_mid_48_8.wav",
    "./sr/day1_wavs/67_near_far_close_30_8.wav",
    offset=0,
    length=22050 * 10,
)

print("Inferring output...")
o_audio = model(i_mel.unsqueeze(0)).squeeze()

print("Saving files...")

with sf.SoundFile("./output/input_audio.wav", "w", samplerate=22050, channels=1) as f:
    f.write(i_audio.numpy())

with sf.SoundFile("./output/target_audio.wav", "w", samplerate=22050, channels=1) as f:
    f.write(t_audio.numpy())

with sf.SoundFile("./output/prediction_audio.wav", "w", samplerate=22050, channels=1) as f:
    f.write(o_audio.numpy())
