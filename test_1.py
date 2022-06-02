import imp
import logging
import os

import numpy as np
import torch
import time
import shutil
import traceback
import subprocess

from threading import Thread

import gdown

import logging

from datetime import datetime

from mozilla_voice_tts.tts.utils.generic_utils import setup_model
from mozilla_voice_tts.tts.utils.synthesis import synthesis
from mozilla_voice_tts.tts.utils.text.symbols import symbols, phonemes, make_symbols
from mozilla_voice_tts.utils.io import load_config
from mozilla_voice_tts.utils.audio import AudioProcessor
from mozilla_voice_tts.tts.utils.io import load_checkpoint
from mozilla_voice_tts.vocoder.utils.generic_utils import setup_generator
from mozilla_voice_tts.vocoder.tf.utils.io import load_checkpoint as load_vocoder_checkpoint

from moviepy.editor import *
from moviepy.audio.AudioClip import *

from scipy.io.wavfile import write

from text_clean import vi_cleaners

time_start = time.time()


def tts(model, sentences, CONFIG, use_cuda, ap, vocoder_model, use_gl, folder=None, number_file=0, figures=True, speaker_id=None, wav=True, times_denoise=0):
    t_1 = time.time()
    waveform_data = np.zeros(1)

    for idx, sentence in enumerate(sentences):
        waveform, alignment, mel_spec, \
        mel_postnet_spec, stop_tokens, inputs = \
            synthesis(model, sentence,
                      CONFIG, use_cuda, ap,
                      speaker_id, style_wav=None,
                      truncated=False,
                      enable_eos_bos_chars=CONFIG.enable_eos_bos_chars)

        if not use_gl:
            waveform = vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
            waveform = waveform.flatten()
        if use_cuda:
            waveform = waveform.cpu()
        waveform = waveform.numpy()
        if len(sentences) > 1:
            _silence = np.zeros(int(ap.sample_rate/4), dtype=np.uint8)
            waveform = np.append(waveform, _silence)
            waveform_data = np.append(waveform_data, waveform)
        else:
            waveform_data = waveform

    if not folder:
        _name = str(datetime.now().time())
        folder = os.getcwd()
        name_voice = f'{_name}.wav'
        path = os.path.join(folder, name_voice)
        
    else:
        _name = number_file
        name_voice = f'{_name}.wav'
        path = os.path.join(folder, name_voice)
    
    # Save wav file to folder
    try:
        write(path, CONFIG.audio['sample_rate'], waveform_data)
    except Exception as e:
        logging.error(f"Can not save wav file to {path} \n {e}")
        
    print(f"TTS success in {time.time()- t_1} s")
    
    

def text_to_speech(sentences=['You need to insert text']):
    """
    Detail : to create audio from text
    :param language: language need to create
    :param voice: human reading voice
    :param sentence: text need to voice
    :return:
    """


    use_cuda = True
    if not use_cuda:
        device = 'cpu'
    else:
        device = 'cuda'

    # model paths
    # model paths
    # TTS_MODEL = "./Tien_Thanh/male/tts_model.pth.tar"
    # TTS_CONFIG = "./Tien_Thanh/male/model_config.json"
    # VOCODER_MODEL = "./Tien_Thanh/male/vocoder_model.pth.tar"
    # VOCODER_CONFIG = "./Tien_Thanh/male/vocoder_config.json"

    TTS_MODEL = "./new_model/tts.pth.tar"
    TTS_CONFIG = "./new_model/tts_config.json"
    VOCODER_MODEL = "./new_model/vocoder.pth.tar"
    VOCODER_CONFIG = "./new_model/vocoder_config.json"

    time_start_load = time.time()

    # load configs
    try:
        TTS_CONFIG = load_config(TTS_CONFIG)
        VOCODER_CONFIG = load_config(VOCODER_CONFIG)
    except Exception as e:
        traceback.print_exc()


    # load the audio processor
    try:
        ap = AudioProcessor(**TTS_CONFIG.audio)
    except:
        traceback.print_exc()
    # ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])

    # LOAD TTS MODEL
    # multi speaker
    speaker_id = None
    speakers = []


    # load the model
    if 'characters' in TTS_CONFIG.keys():
        symbols, phonemes = make_symbols(**TTS_CONFIG.characters)
    else:
        raise "You must config characters in config file"

    num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)

    model = setup_model(num_chars, len(speakers), TTS_CONFIG)

    # load model state
    cp = torch.load(TTS_MODEL, map_location=torch.device(device))

    # load the model
    try:
        model.load_state_dict(cp['model'])
    except:
        traceback.print_exc()

    if use_cuda:
        model.cuda()
    model.eval()

    # set model stepsize
    if 'r' in cp:
        model.decoder.set_r(cp['r'])

    # LOAD VOCODER MODEL
    vocoder_model = setup_generator(VOCODER_CONFIG)
    vocoder_model.load_state_dict(torch.load(VOCODER_MODEL, map_location=device)["model"])
    vocoder_model.remove_weight_norm()
    vocoder_model.inference_padding = 0

    # ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])

    if use_cuda:
        vocoder_model.cuda()
    vocoder_model.eval()
    print(f"time need to load data is {time.time() - time_start_load}")
    try:
        path_tmp = tts(model, sentences, TTS_CONFIG, use_cuda,
                                ap, vocoder_model,
                                use_gl=False, figures=True)
    except Exception as e:
        print(e)


def text_to_speech_en(sentences=['You need to insert text']):
    """
    Detail : to create audio from text
    :param language: language need to create
    :param voice: human reading voice
    :param sentence: text need to voice
    :return:
    """


    use_cuda = True
    if not use_cuda:
        device = 'cpu'
    else:
        device = 'cuda'

    # model paths
    # model paths
    TTS_MODEL = "tts_model.pth.tar"
    TTS_CONFIG = "config.json"
    VOCODER_MODEL = "vocoder_model.pth.tar"
    VOCODER_CONFIG = "config_vocoder.json"
    time_start_load = time.time()

    # load configs
    try:
        TTS_CONFIG = load_config(TTS_CONFIG)
        VOCODER_CONFIG = load_config(VOCODER_CONFIG)
    except Exception as e:
        traceback.print_exc()


    # load the audio processor
    try:
        ap = AudioProcessor(**TTS_CONFIG.audio)
    except:
        traceback.print_exc()
    # ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])

    # LOAD TTS MODEL
    # multi speaker
    speaker_id = None
    speakers = []


    # load the model
    if 'characters' in TTS_CONFIG.keys():
        symbols, phonemes = make_symbols(**TTS_CONFIG.characters)
    else:
        raise "You must config characters in config file"

    num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)

    model = setup_model(num_chars, len(speakers), TTS_CONFIG)

    # load model state
    cp = torch.load(TTS_MODEL, map_location=torch.device(device))

    # load the model
    try:
        model.load_state_dict(cp['model'])
    except:
        traceback.print_exc()

    if use_cuda:
        model.cuda()
    model.eval()

    # set model stepsize
    if 'r' in cp:
        model.decoder.set_r(cp['r'])

    # LOAD VOCODER MODEL
    vocoder_model = setup_generator(VOCODER_CONFIG)
    vocoder_model.load_state_dict(torch.load(VOCODER_MODEL, map_location=device)["model"])
    vocoder_model.remove_weight_norm()
    vocoder_model.inference_padding = 0

    # ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])

    if use_cuda:
        vocoder_model.cuda()
    vocoder_model.eval()
    print(f"time need to load data is {time.time() - time_start_load}")
    try:
        path_tmp = tts(model, sentences, TTS_CONFIG, use_cuda,
                                ap, vocoder_model,
                                use_gl=False, figures=True)
    except Exception as e:
        print(e)



sentences_1 =  [
        "Bill got in the habit of asking himself “Is that thought true?” and if he wasn’t absolutely certain it was, he just let it go.\
        By Tuesday morning in Moscow, more than 1 million signatures had been added to a Russian-language Change.org petition against the war in Ukraine.",\
        "Putin has no reason to publicize the anger at his rule and every reason to snuff it out.\
        It is a sobering assessment that when Putin puts his finger in the wind of public opinion he can be reasonably sure it is blowing in the direction he instructed his state organs to set it.",\
        "A top view of their hiding place in the ceiling of the plane.\
        After the aircraft carrying Yohannes and Gebremeskel landed in Brussels, the two waited for their chance to reach the terminal building.",
        "But with each rehearsal Putin tweaked the play. In Georgia in 2008, Putin's soldiers had dirty boots and rusty tanks, but he first tested his now infamous cyber attacks. He got away with it.\
        It is too late for Dima, who was killed in eastern Ukraine in fighting a year after we spoke and long before Europeans finally recognized that it was them he was fighting for.",\
        "A major drug bust was made at sea, Molly Ringwald's mom forgot her birthday, and an Olympic gold medalist showed off his moves off the ice. These are the must-watch videos of the week.\
        Three tons of cocaine were seized by Mexico's army as a helicopter captured the high-speed chase at sea, about 68 nautical miles from the resort city of Cabo San Lucas.",\
        "A training accident involving two Utah National Guard UH-60 Black Hawk helicopters was caught on video from a nearby ski resort. No crew members or skiers at the Snowbird ski resort were injured.\
        A textbook wolf howl was captured on camera. CNN's Jeanne Moos reports it put the wolf to sleep.",\
        "If the compound came to the market, the divorcing pair would have to find a buyer looking for a Seattle home with a trampoline room and 66,000 square feet to dust.\
        But the suburban Seattle property, should it ever be up for grabs, could be a very tough sell.",\
        "Such a degree of customization within a private residence doesn’t make a home more valuable, though it does garner attention, according to Mr. Carswell."
]


sentences_2 =  [
        "Bill got in the habit of asking himself “Is that thought true?” and if he wasn’t absolutely certain it was, he just let it go.\
        By Tuesday morning in Moscow, more than 1 million signatures had been added to a Russian-language Change.org petition against the war in Ukraine.",\
        "Putin has no reason to publicize the anger at his rule and every reason to snuff it out.\
        It is a sobering assessment that when Putin puts his finger in the wind of public opinion he can be reasonably sure it is blowing in the direction he instructed his state organs to set it.",\
        "A top view of their hiding place in the ceiling of the plane.\
        After the aircraft carrying Yohannes and Gebremeskel landed in Brussels, the two waited for their chance to reach the terminal building.",
        "But with each rehearsal Putin tweaked the play. In Georgia in 2008, Putin's soldiers had dirty boots and rusty tanks, but he first tested his now infamous cyber attacks. He got away with it.\
        It is too late for Dima, who was killed in eastern Ukraine in fighting a year after we spoke and long before Europeans finally recognized that it was them he was fighting for.",\
        "A major drug bust was made at sea, Molly Ringwald's mom forgot her birthday, and an Olympic gold medalist showed off his moves off the ice. These are the must-watch videos of the week.\
        Three tons of cocaine were seized by Mexico's army as a helicopter captured the high-speed chase at sea, about 68 nautical miles from the resort city of Cabo San Lucas.",\
        "A training accident involving two Utah National Guard UH-60 Black Hawk helicopters was caught on video from a nearby ski resort. No crew members or skiers at the Snowbird ski resort were injured.\
        A textbook wolf howl was captured on camera. CNN's Jeanne Moos reports it put the wolf to sleep.",\
        "If the compound came to the market, the divorcing pair would have to find a buyer looking for a Seattle home with a trampoline room and 66,000 square feet to dust.\
        But the suburban Seattle property, should it ever be up for grabs, could be a very tough sell.",\
        "Such a degree of customization within a private residence doesn’t make a home more valuable, though it does garner attention, according to Mr. Carswell."
]

sentences_3 =  [
    "Tôi đang chạy thử chương trình này. Đang tét với trường hợp chạy nhiều thờ rét.",
    "Chào mừng bạn đến vơi công ty của chúng tôi.",
    "Chạy thử chương trình mới.",
    "Tôi đang chạy thử chương trình này. Đang tét với trường hợp chạy nhiều thờ rét.",
    "Chào mừng bạn đến vơi công ty của chúng tôi.",
    "Chạy thử chương trình mới.",
    "Tôi đang chạy thử chương trình này. Đang tét với trường hợp chạy nhiều thờ rét.",
    "Chào mừng bạn đến vơi công ty của chúng tôi.",
    "Chạy thử chương trình mới."

]

sentences_4 =  [
    "Tôi đang chạy thử chương trình này. Đang tét với trường hợp chạy nhiều thờ rét.",
    "Chào mừng bạn đến vơi công ty của chúng tôi.",
    "Chạy thử chương trình mới."
]

# text_to_speech(sentences_3)

# thread_1 = Thread(target=text_to_speech, args=(sentences_1, ))
# thread_2 = Thread(target=text_to_speech, args=(sentences_2, ))
thread_3 = Thread(target=text_to_speech, args=(sentences_3, ))


# thread_1.start()
# thread_2.start()
thread_3.start()

# thread_1.join()
# thread_2.join()
# thread_3.join()

print(f"Sum of time is {time.time() - time_start}")
