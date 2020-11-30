from distributed import apply_gradient_allreduce
import time

import IPython.display as ipd

from numpy import finfo
import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from text import hangul_to_sequence
from waveglow.denoiser import Denoiser
from scipy.io import wavfile

hparams=create_hparams()


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
    return model



def plot_data(data, figsize=(16, 4)):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')
    return plt

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

if __name__ =='__main__':
    hparams = create_hparams()
    hparams.distributed_run=False
    # 경상 "/home/ubuntu/Workspaces/thien/nvidia-tacotron-je/outdir/male/gyeongsang/checkpoint_58000"
    # 제주 "/home/ubuntu/Workspaces/thien/nvidia-tacotron-je/outdir/checkpoint_70056"
    # 전라 "/home/ubuntu/Workspaces/thien/nvidia-tacotron-jeonla/outdir/checkpoint_155000"
    checkpoint_path = "/home/ubuntu/Workspaces/thien/nvidia-tacotron-je/outdir/male/gyeongsang/checkpoint_58000"
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()

    # 경상 "/home/ubuntu/Workspaces/thien/nvidia-tacotron-je/outdir/male/gyeongsang/waveglow_gyeongsang_266000"
    # 제주 "/home/ubuntu/Workspaces/thien/nvidia-tacotron-je/outdir/waveglow_jeju_146000" 
    # 전라 "/home/ubuntu/Workspaces/thien/nvidia-tacotron-jeonla/outdir/waveglow_240000"
    waveglow_path = "/home/ubuntu/Workspaces/thien/nvidia-tacotron-je/outdir/male/gyeongsang/waveglow_gyeongsang_266000"
    taco = checkpoint_path.split('_')[-1]
    wave = waveglow_path.split('_')[-1]


    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    # 텍스트 넣기
    txt_list = ['여러분이 있었기 땜시 즈희가 잘할 수 있었십니다',
                '니는 어떤 제목 드라마를 좋아하노?',
                '내 친구들은 다 휴가 갔습니더',
                '어. 그라모 다덜 이메일 주소들 좀 도.',
                '내는 이 책으로 열심히 공부하고 싶어예',
                '이 둘은 같은 디자인인데 사이즈가 다릅니더',
                '온라인상에서도 마찬가지입니더',
                '고객분들에 한해 무료로 배포하는거 아닙니꺼?',
                '애들이 묵기에는 쪼매 그렇네예.',
                '당신은 기차역에서 열차를 잘못 탔습니더',
                '그녀는 매사에 정확한 사람입니더',
                '갈비탕을 맛있게 하는 곳이 있으믄 거 가고 싶데이.',
                '건물 중에 어데 갈라꼬 하시는건가예?',
                '훨씬 나아지긴 했는데 지금은 너무 밝아서 파이다.',
                '예, 문제가 있으신가예?',
                '당신은 내랑 꼭 같이 가지 않아도 됩니더',
                '당신 마이 아파 보이는데 병원에 가보는 게 어떻습니꺼?',
                '저는 제가 결혼하게 되어가 기쁩니더',
                '오늘이 물리치료 몇 번째 받으시는 긴가예?',
                '영어보다 중국어로 말씀을 더 잘하시네예']
    for i,text in enumerate(txt_list):
        # text = "야. 도로모깡도 왜정시대나 낫주. 도로모깡도 엇일 땐양 허벅에."
        sequence = np.array(hangul_to_sequence(text))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

        start = time.time()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

        print(text)
        print('mel output shape: {}'.format(mel_outputs.shape))
        print('Tacotron synthesize time: {}'.format(time.time() - start))
        start = time.time()
        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

        print('Wavenet synthesize time: {}'.format(time.time() - start))
        audio = denoiser(audio, strength=0.01)[:, 0]
        start = time.time()
        save_wav(audio[0].data.cpu().numpy(), f'gyeong_{i}.wav', sr=hparams.sampling_rate)  # 파일명 변경
        print('Audio --> .wav file saving time: {}'.format(time.time() - start))
    #     audio = audio[0].data.cpu().numpy()
    #
    # start = time.time()
    # save_wav(audio, 'output.wav', sr=hparams.sampling_rate)
    # print('audio saving time: {}'.format(time.time() - start))

    # start = time.time()
    # audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    # ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)
    # print('audio denoising time: {}'.format(time.time() - start))
    #
    # plt=plot_data((mel_outputs.float().data.cpu().numpy()[0],
    #            mel_outputs_postnet.float().data.cpu().numpy()[0],
    #            alignments.float().data.cpu().numpy()[0].T))
    # plt.savefig('output_{}_{}.png'.format(taco, wave))

#
# def mel_to_audio(hparams, mel, sigma=0.1):
#
#     upsample = torch.nn.ConvTranspose1d(hparams.n_mel_channels,hparams.n_mel_channels,1024, stride=256)
#     spect = upsample(mel)