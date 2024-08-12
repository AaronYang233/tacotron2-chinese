#@markdown 配置：

#@markdown 重新运行即可应用配置的更改

#国际 HiFi-GAN 模型(有点机器音): 1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW
#@markdown 你训练好的tacotron2模型的路径填在`Tacotron2_Model`这里
# Tacotron2_Model = './model/tacotron2_statedict.pt'#@param {type:"string"}
# Tacotron2_Model = './sys_outdir/IM'
# Tacotron2_Model = './model/IM'
Tacotron2_Model = './model/cpop_model/checkpoint_Tacotron2_1500.pt'
TACOTRON2_ID = Tacotron2_Model
# HIFIGAN_ID = "1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW"
#@markdown 选择预处理文本的cleaner
text_cleaner = 'chinese_english_phrase_cleaners'#@param {type:"string"}

from tqdm import tqdm

with tqdm(total=5, leave=False) as pbar:
    import os
    from os.path import exists, join, basename, splitext
    import sys
    sys.path.append('./')
    sys.path.append('hifi-gan')
    import time
    import matplotlib
    import matplotlib.pylab as plt
    # import IPython.display as ipd
    import numpy as np
    import torch
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    import json
    from scipy.io.wavfile import write
    pbar.update(1)
    
    from hparams import create_hparams
    from model import Tacotron2
    from layers import TacotronSTFT
    from audio_processing import griffin_lim
    from text import text_to_sequence
    from env import AttrDict
    from meldataset import MAX_WAV_VALUE
    from models import Generator

    pbar.update(1) # initialized Dependancies

    graph_width = 900
    graph_height = 360
    def plot_data(data, figsize=(int(graph_width/100), int(graph_height/100))):
        fig, axes = plt.subplots(1, len(data), figsize=figsize)
        for i in range(len(data)):
            axes[i].imshow(data[i], aspect='auto', origin='lower', 
                        interpolation='none', cmap='inferno')
        fig.canvas.draw()
        plt.show()

    # Setup Pronounciation Dictionary
    thisdict = {}
    for line in reversed((open('./model/merged.dict.txt', "r").read()).splitlines()):
        thisdict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()

    pbar.update(1) # Downloaded and Set up Pronounciation Dictionary

    def ARPA(text, punctuation=r"!?,.;", EOS_Token=True):
        out = ''
        for word_ in text.split(" "):
            word=word_; end_chars = ''
            while any(elem in word for elem in punctuation) and len(word) > 1:
                if word[-1] in punctuation: end_chars = word[-1] + end_chars; word = word[:-1]
                else: break
            try:
                word_arpa = thisdict[word.upper()]
                word = "{" + str(word_arpa) + "}"
            except KeyError: pass
            out = (out + " " + word + end_chars).strip()
        if EOS_Token and out[-1] != ";": out += ";"
        return out

    def get_hifigan():
        # Download HiFi-GAN
        hifigan_pretrained_model = './model/hifimodel'
        if not exists(hifigan_pretrained_model):
            raise Exception("HiFI-GAN model failed to download!")

        # Load HiFi-GAN
        conf = os.path.join("hifi-gan", "config_v1.json")
        with open(conf) as f:
            json_config = json.loads(f.read())
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        
        hifigan = Generator(h).to(device)
        state_dict_g = torch.load(hifigan_pretrained_model, map_location=device, weights_only=True)
        
        
        hifigan.load_state_dict(state_dict_g["generator"])
        hifigan.eval()
        hifigan.remove_weight_norm()
        return hifigan, h

    hifigan, h = get_hifigan()
    pbar.update(1) # Downloaded and Set up HiFi-GAN

    def has_MMI(STATE_DICT):
        return any(True for x in STATE_DICT.keys() if "mi." in x)

    def get_Tactron2(MODEL_ID):
        # Download Tacotron2
        tacotron2_pretrained_model = TACOTRON2_ID
        if not exists(tacotron2_pretrained_model):
            raise Exception("Tacotron2 model failed to download!")
        # Load Tacotron2 and Config
        hparams = create_hparams()
        hparams.sampling_rate = 22050
        hparams.max_decoder_steps = 3000 # Max Duration
        hparams.gate_threshold = 0.25 # Model must be 25% sure the clip is over before ending generation
        model = Tacotron2(hparams)
        state_dict = torch.load(tacotron2_pretrained_model, map_location=device, weights_only=True)['state_dict']
        if has_MMI(state_dict):
            raise Exception("ERROR: This notebook does not currently support MMI models.")
        model.load_state_dict(state_dict) # <<< GPU
        # _ = model.cuda().eval().half()
        _ = model.cpu().eval().half()
        return model, hparams

    model, hparams = get_Tactron2(TACOTRON2_ID)
    previous_tt2_id = TACOTRON2_ID

    pbar.update(1) # Downloaded and Set up Tacotron2

    # Extra Info
    def end_to_end_infer(text, pronounciation_dictionary, show_graphs):
        for i in [x for x in text.split("\n") if len(x)]:
            if not pronounciation_dictionary:
                if i[-1] != ";": i=i+";" 
            else: i = ARPA(i)
            with torch.no_grad(): # save VRAM by not including gradients
                sequence = np.array(text_to_sequence(i, [text_cleaner]))[None, :]
                # sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()　＃　<<<===GPU
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()
                mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
                if show_graphs:
                    plot_data((mel_outputs_postnet.float().data.cpu().numpy()[0],
                            alignments.float().data.cpu().numpy()[0].T))
                y_g_hat = hifigan(mel_outputs_postnet.float())
                audio = y_g_hat.squeeze()
                audio = audio * MAX_WAV_VALUE
                print("")
                # ipd.display(ipd.Audio(audio.cpu().numpy().astype("int16"), rate=hparams.sampling_rate))
                audio_np = audio.cpu().numpy().astype("int16")
                sampling_rate = hparams.sampling_rate

                # Define the path to save the audio file
                output_path = "output_audio.wav"

                # Save the audio file
                write(output_path, sampling_rate, audio_np)

from IPython.display import clear_output
clear_output()
initilized = "Ready"

if previous_tt2_id != TACOTRON2_ID:
    print("Updating Models")
    model, hparams = get_Tactron2(TACOTRON2_ID)
    hifigan, h = get_hifigan()
    previous_tt2_id = TACOTRON2_ID

pronounciation_dictionary = False #@param {type:"boolean"}
# disables automatic ARPAbet conversion, useful for inputting your own ARPAbet pronounciations or just for testing
show_graphs = True #@param {type:"boolean"}
max_duration = 25 #this does nothing
model.decoder.max_decoder_steps = 2000 #@param {type:"integer"}
stop_threshold = 0.324 #@param {type:"number"}
model.decoder.gate_threshold = stop_threshold

# #@markdown ---

print(f"Current Config:\npronounciation_dictionary: {pronounciation_dictionary}\nshow_graphs: {show_graphs}\nmax_duration (in seconds): {max_duration}\nstop_threshold: {stop_threshold}\n\n")

time.sleep(1)
# print("输入要转换成语音的文本.")
contents = []
# while True:
#     try:
#         print("-"*50)
#         line = input()
#         if line == "":
#             continue
#         end_to_end_infer(line, pronounciation_dictionary, show_graphs)
#     except EOFError:
#         break
#     except KeyboardInterrupt:
#         print("程序终止...")
#         break

line = "跟我天天向上看世间繁华左右手快慢抱逍遥"
# line = "la la la la la la"
end_to_end_infer(line, pronounciation_dictionary, show_graphs)