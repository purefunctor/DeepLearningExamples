from pathlib import Path
import re

DAY_1_WAVS = Path("sr/day1_wavs")
DAY_2_WAVS = Path("sr/day2_wavs")
PATTERN = re.compile(r"^(.+)_(\w+)_\w+_\w+_\d+_(\d+).wav$")


def generate_pairs(data_path: Path) -> dict:
    nt1 = {}
    u67 = {}
    for wav_file in data_path.iterdir():
        match = PATTERN.match(wav_file.name)
        if match is None:
            raise Exception(f"Invalid file {wav_file.name}")
        microphone, _, offset = match.groups()
        if microphone == "67":
            u67[offset] = wav_file
        elif microphone == "nt1":
            nt1[offset] = wav_file
        else:
            raise Exception(f"Invalid microphone {microphone}")

    pairs = []
    for key, value in nt1.items():
        pairs.append((value, u67[key]))

    return pairs


if __name__ == "__main__":
    with open("sr/manifest.txt", "w") as f:
        pairs = []
        pairs.extend(generate_pairs(DAY_1_WAVS))
        pairs.extend(generate_pairs(DAY_2_WAVS))
        for (nt1, u67) in pairs:
            print(nt1, u67, sep=",", file=f)
