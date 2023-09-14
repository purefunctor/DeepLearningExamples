from pathlib import Path
from librosa import load
import soundfile as sf


DAY_1_ORIGINAL = Path("sr/day1_original")
DAY_1_WAVS = Path("sr/day1_wavs")

DAY_2_ORIGINAL = Path("sr/day2_original")
DAY_2_WAVS = Path("sr/day2_wavs")


def downsample_files(input_path: Path, target_path: Path):
    if not target_path.exists():
        target_path.mkdir(parents=True)
    for input_file in input_path.iterdir():
        print(f"Processing {input_file}")
        target_file = target_path / input_file.name
        resampled_audio, _ = load(input_file, sr=22050)
        sf.write(target_file, resampled_audio, 22050)


if __name__ == "__main__":
    downsample_files(DAY_1_ORIGINAL, DAY_1_WAVS)
    downsample_files(DAY_2_ORIGINAL, DAY_2_WAVS)
