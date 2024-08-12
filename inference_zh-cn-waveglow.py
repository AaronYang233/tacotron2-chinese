import os
from os.path import exists, join, basename, splitext

import sys

sys.path.append("./")
sys.path.append("waveglow")

import time
import matplotlib
import matplotlib.pylab as plt
from scipy.io.wavfile import write

# @title 加载预训练模型
force_download_TT2 = True
tacotron2_pretrained_model = "./model/tacotron2_statedict.pt"  # @param {type:"string"}
waveglow_pretrained_model = (
    "./model/waveglow_256channels_universal_v5.pt"  # @param {type:"string"}
)

# import IPython.display as ipd
import numpy as np
import torch

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
device = "cpu"

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
from denoiser import Denoiser

graph_width = 900
graph_height = 360


def plot_data(data, figsize=(int(graph_width / 100), int(graph_height / 100))):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(
            data[i],
            aspect="auto",
            origin="bottom",
            interpolation="none",
            cmap="inferno",
        )
    fig.canvas.draw()
    plt.show()


thisdict = {}
for line in reversed((open("./model/merged.dict.txt", "r").read()).splitlines()):
    thisdict[(line.split(" ", 1))[0]] = (line.split(" ", 1))[1].strip()


def ARPA(text):
    out = ""
    for word_ in text.split(" "):
        word = word_
        end_chars = ""
        while any(elem in word for elem in r"!?,.;") and len(word) > 1:
            if word[-1] == "!":
                end_chars = "!" + end_chars
                word = word[:-1]
            if word[-1] == "?":
                end_chars = "?" + end_chars
                word = word[:-1]
            if word[-1] == ",":
                end_chars = "," + end_chars
                word = word[:-1]
            if word[-1] == ".":
                end_chars = "." + end_chars
                word = word[:-1]
            if word[-1] == ";":
                end_chars = ";" + end_chars
                word = word[:-1]
            else:
                break
        try:
            word_arpa = thisdict[word.upper()]
        except:
            word_arpa = ""
        if len(word_arpa) != 0:
            word = "{" + str(word_arpa) + "}"
        out = (out + " " + word + end_chars).strip()
    if out[-1] != ";":
        out = out + ";"
    return out


# initialize Tacotron2 with the pretrained model
hparams = create_hparams()

hparams.sampling_rate = 22050  # Don't change this
hparams.max_decoder_steps = (
    1000  # How long the audio will be before it cuts off (1000 is about 11 seconds)
)
hparams.gate_threshold = 0.1  # Model must be 90% sure the clip is over before ending generation (the higher this number is, the more likely that the AI will keep generating until it reaches the Max Decoder Steps)
model = Tacotron2(hparams)
model.load_state_dict(
    torch.load(tacotron2_pretrained_model, map_location=device, weights_only=False)[
        "state_dict"
    ]
)
_ = model.cuda().eval().half()

# Load WaveGlow
waveglow = torch.load(
    waveglow_pretrained_model, map_location=device, weights_only=True)["model"]
waveglow.cuda().eval().half()

for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)


text = "Your Text Here"  # @param {type:"string"}
sigma = 0.8
denoise_strength = 0.324
raw_input = True  # disables automatic ARPAbet conversion, useful for inputting your own ARPAbet pronounciations or just for testing.
# should be True if synthesizing a non-English language

for i in text.split("\n"):
    if len(i) < 1:
        continue
    print(i)
    if raw_input:
        if i[-1] != ";":
            i = i + ";"
    else:
        i = ARPA(i)
    print(i)
    with torch.no_grad():  # save VRAM by not including gradients
        sequence = np.array(text_to_sequence(i, ["english_cleaners"]))[None, :]
        # sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        plot_data(
            (
                mel_outputs_postnet.float().data.cpu().numpy()[0],
                alignments.float().data.cpu().numpy()[0].T,
            )
        )
        audio = waveglow.infer(mel_outputs_postnet, sigma=sigma)
        print("")
        # ipd.display(ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate))
        audio_path = "output_waveglow.wav"
        audio_np = write(audio_path, hparams.sampling_rate, audio[0].data.cpu().numpy())
