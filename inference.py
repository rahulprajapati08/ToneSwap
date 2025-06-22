import os
import torch
import librosa
from scipy.io.wavfile import write
import time

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder

# Load hyperparams, model, encoder once
HP_PATH = "configs/freevc.json"
PT_PATH = "checkpoints/freevc.pth"
SPK_PATH = "speaker_encoder/ckpt/pretrained_bak_5805000.pt"

hps = utils.get_hparams_from_file(HP_PATH)

net_g = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint(PT_PATH, net_g, None, True)
cmodel = utils.get_cmodel(0)

smodel = None
if hps.model.use_spk:
    smodel = SpeakerEncoder(SPK_PATH)

def voice_convert(src_path, tgt_path, output_path="output/freevc/output.wav"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with torch.no_grad():
        # Target
        wav_tgt, _ = librosa.load(tgt_path, sr=hps.data.sampling_rate)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

        if hps.model.use_spk:
            g_tgt = smodel.embed_utterance(wav_tgt)
            g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
        else:
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
            mel_tgt = mel_spectrogram_torch(
                wav_tgt,
                hps.data.filter_length,
                hps.data.n_mels_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

        # Source
        wav_src, _ = librosa.load(src_path, sr=hps.data.sampling_rate)
        wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
        c = utils.get_content(cmodel, wav_src)

        if hps.model.use_spk:
            audio = net_g.infer(c, g=g_tgt)
        else:
            audio = net_g.infer(c, mel=mel_tgt)

        audio = audio[0][0].data.cpu().float().numpy()
        write(output_path, hps.data.sampling_rate, audio)
        return output_path
