# Tacotron2-Chinese

- Tacotron2 implementation of Chinese

## Links

- Reference: [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)

- Reference: [CjangCjengh/tacotron2-japanese](https://github.com/CjangCjengh/tacotron2-japanese)

- [Pre-training tacotron2 models](https://github.com/CjangCjengh/TTSModels)

- [Latest JP changes can be viewed in this repository](https://github.com/StarxSky/tacotron2-JP)

- [Opencpop dataset](https://wenet.org.cn/opencpop/)

- [Code-Pre-training](https://pan.baidu.com/s/13cl40S3YN4g9wMjd6vfpTA?pwd=hm7g)

## How to use

1. Put raw Chinese + English or Japanese texts in ./filelists
2. Put WAV files in ./wav Or manually specify dir path
3. (Optional) Download NVIDIA's [pretrained model](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing)
4. pip install -r requirements.txt
5. Open ./train_zh-cn.py to start training
6. Use hifi-gan or waveglow to generate wav
7. Open ./inference_zh-cn-hifigan.py or ./inference_zh-cn-waveglow.py to generate voice

## Choose a cleaner for preprocessing text

### 1. 'chinese_english_phrase_cleaners'

Map the specified fields through ./conf/security_mapping.json

```
Before

感受 我发端的指尖

After

gan3 shou4 wo3 fa1 duan1 de zhi3 jian1.

Special Treatment

gan3 shou4 [SPACE] wo3 fa1 duan1 de zhi3 jian1.

```

### 2. 'japanese_cleaners'

```
Before

何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも

After

nanikaacltaraitsudemohanashItekudasai.gakuiNnokotojanaku,shijinikaNsurukotodemonanidemo.
```

### 3. 'japanese_tokenization_cleaners'

```
Before

何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも

After

nani ka acl tara itsu demo hanashi te kudasai. gakuiN no koto ja naku, shiji nikaNsuru koto de mo naNdemo.
```

### 4. 'japanese_accent_cleaners'

```
Before

何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも

After

:na)nika a)cltara i)tsudemo ha(na)shIte ku(dasa)i.:ga(kuiNno ko(to)janaku,:shi)jini ka(Nsu)ru ko(to)demo na)nidemo.
```

### 4. 'japanese_phrase_cleaners'
```
Before

何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも

After

nanika acltara itsudemo hanashIte kudasai. gakuiNno kotojanaku, shijini kaNsuru kotodemo nanidemo.
```
# Install FFmpeg

Download FFmpeg

## Method 1

```shell
sudo apt-get install ffmpeg
```

## Method 2

[FFmpeg official website](https://ffmpeg.org/) Find the corresponding system version to download.

Extract the compressed file to any directory.

Add the bin folder in the installation directory to the Path environment variable of the system.

If the environment variables are successfully added, open cmd and type ffmpeg - version to see the version information.