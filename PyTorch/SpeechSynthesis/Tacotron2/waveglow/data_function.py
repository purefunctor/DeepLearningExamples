# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\

from pathlib import Path
import torch
import tacotron2_common.layers as layers
from tacotron2_common.utils import load_wav_to_torch, to_gpu


def load_manifest(manifest_file):
    pairs = []
    with open(manifest_file, "r") as f:
        for line in f.readlines():
            i, t = line.split(",")
            i = i.strip()
            t = t.strip()
            pairs.append((Path(i), Path(t)))
    return pairs


class MelAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) computes mel-spectrograms from audio files.
    """

    def __init__(self, dataset_path, audiopaths_and_text, args):
        self.manifest_pairs = load_manifest(audiopaths_and_text)
        self.max_wav_value = args.max_wav_value
        self.sampling_rate = args.sampling_rate
        self.stft = layers.TacotronSTFT(
            args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.sampling_rate, args.mel_fmin,
            args.mel_fmax)
        self.segment_length = args.segment_length

    def _take_start(self, audio):
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = torch.randint(0, max_audio_start + 1, size=(1,)).item()
            return audio_start
        else:
            return None

    def _take_mel(self, audio):
        audio_norm = audio.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = melspec.squeeze(0)
        return melspec

    def _take_segment(self, audio, audio_start):
        if audio_start is not None:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = torch.randint(0, max_audio_start + 1, size=(1,)).item()
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(
                audio, (0, self.segment_length - audio.size(0)), 'constant').data    

        audio = audio / self.max_wav_value
        melspec = self._take_mel(audio)

        return (melspec, audio, len(audio))

    def get_input_mel_target_audio(self, i, t):
        i_audio, i_sampling_rate = load_wav_to_torch(i)
        t_audio, t_sampling_rate = load_wav_to_torch(t)

        if i_sampling_rate != self.stft.sampling_rate or t_sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(i_sampling_rate, self.stft.sampling_rate))

        audio_start = self._take_start(i_audio)

        (i_mel, i_audio, i_len) = self._take_segment(i_audio, audio_start)
        (_, t_audio, t_len) = self._take_segment(t_audio, audio_start)

        assert i_len == t_len

        return (i_mel, t_audio, i_len)

    def take_at(self, i, t, *, offset, length):
        i_audio, _ = load_wav_to_torch(i)
        t_audio, _ = load_wav_to_torch(t)

        i_audio = i_audio[offset:offset+length] / self.max_wav_value
        t_audio = t_audio[offset:offset+length] / self.max_wav_value

        i_mel = self._take_mel(i_audio)
        t_mel = self._take_mel(t_audio)

        return (i_audio, i_mel, t_audio, t_mel)

    def __getitem__(self, index):
        index = index // 10
        return self.get_input_mel_target_audio(*self.manifest_pairs[index])

    def __len__(self):
        return len(self.manifest_pairs * 10)


def batch_to_gpu(batch):
    x, y, len_y = batch
    x = to_gpu(x).float()
    y = to_gpu(y).float()
    len_y = to_gpu(torch.sum(len_y))
    return ((x, y), y, len_y)
