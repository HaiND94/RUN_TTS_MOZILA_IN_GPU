import os
import torch
import time
import numpy as np

from threading import Thread

from datetime import datetime

from TTS.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.utils.synthesis import synthesis
from TTS.vocoder.utils.generic_utils import setup_generator

from scipy.io.wavfile import write


time_start_load = time.time()

# To implement model
def tts(model, text_list, CONFIG, use_cuda, ap, vocoder_model, use_gl=False, folder=None, number_file=0, figures=True, speaker_id=None, wav=False, times_denoise=0):
    waveform_data = np.zeros(1)
    # runtime settings
    if not use_cuda:
        device = 'cpu'
    else:
        device = 'cuda'

    start_time = time.time()

    for idx, sentence in enumerate(text_list):
        _start_time = time.time()

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
        
        print(f"Time of process {device} is {time.time() - _start_time}", end="\n\n")
        # print(f"Time of process GPU in all process {time.time() - start_time}", end="\n\n")
        __start_time_cpu = time.time()
        
        if use_cuda:
            waveform = waveform.cpu()
        waveform = waveform.numpy()
        if len(text_list) > 1:
            _silence = np.zeros(int(ap.sample_rate/4), dtype=np.uint8)
            waveform = np.append(waveform, _silence)
            waveform_data = np.append(waveform_data, waveform)
        else:
            waveform_data = waveform

        # print(f"Time process in CPU is {time.time() - __start_time_cpu}", end="\n\n")
        
    # print(f"Time in all process is {time.time() - start_time}")
    _start_time = time.time()
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
        print(f"Can not save wav file to {path} \n {e}")
        if not wav:
            return None
        else:
            return None
    print(f"Time need to save data is {time.time() - _start_time}")
    print(f"Time of process is {time.time() - start_time}")
    print("success!")
    return True

def test_multi(sentences):

    # runtime settings
    use_cuda = True
    if not use_cuda:
        device = 'cpu'
    else:
        device = 'cuda'

    # model paths
    TTS_MODEL = "tts_model.pth.tar"
    TTS_CONFIG = "config.json"
    VOCODER_MODEL = "vocoder_model.pth.tar"
    VOCODER_CONFIG = "config_vocoder.json"

    # load configs
    TTS_CONFIG = load_config(TTS_CONFIG)
    VOCODER_CONFIG = load_config(VOCODER_CONFIG)

    # load the audio processor
    ap = AudioProcessor(**TTS_CONFIG.audio)         

    # LOAD TTS MODEL
    # multi speaker 
    speaker_id = None
    speakers = []

    # load the model
    num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
    model = setup_model(num_chars, len(speakers), TTS_CONFIG)

    # load model state
    cp =  torch.load(TTS_MODEL, map_location=device)

    # load the model
    model.load_state_dict(cp['model'])
    if use_cuda:
        model.cuda()
    model.eval()

    # set model stepsize
    if 'r' in cp:
        model.decoder.set_r(cp['r'])

    # LOAD VOCODER MODEL
    vocoder_model = setup_generator(VOCODER_CONFIG)
    vocoder_model.load_state_dict(torch.load(VOCODER_MODEL, map_location=torch.device(device))["model"])
    vocoder_model.remove_weight_norm()
    vocoder_model.inference_padding = 0

    ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])    
    if use_cuda:
        vocoder_model.cuda()
    vocoder_model.eval()

    print(f"Time need to load data is {time.time() - time_start_load}")

    result = tts(model, sentences, TTS_CONFIG, use_cuda, ap, vocoder_model)

    return result



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

sentences_4 =  [
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

thread_1 = Thread(target=test_multi, args=(sentences_1, ))
thread_2 = Thread(target=test_multi, args=(sentences_2, ))
# thread_3 = Thread(target=test_multi, args=(sentences_3, ))


thread_1.start()
thread_2.start()
# thread_3.start()

thread_1.join()
thread_2.join()
# thread_3.join()

print(f"Sum of time is {time.time() - time_start_load}")
