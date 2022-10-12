# Copyright (c) 2022 Horizon Inc. (author: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import librosa
import torch
import torchaudio
import yaml

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torchaudio.compliance.kaldi as kaldi

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.mask import make_pad_mask
from wenet.utils.common import replace_duplicates_with_blank


def get_args():
    parser = argparse.ArgumentParser(
        description='Analyze latency and plot CTC-Spike.')
    parser.add_argument('--config', required=True,
                        type=str, help='configration')
    parser.add_argument('--ckpt', required=True,
                        type=str, help='model checkpoint')
    parser.add_argument('--tag', required=True,
                        type=str, help='image subtitle')
    parser.add_argument('--wavscp', required=True,
                        type=str, help='wav.scp')
    parser.add_argument('--alignment', required=True,
                        type=str, help='force alignment, generated by Kaldi.')
    parser.add_argument('--chunk_size', required=True,
                        type=int, help='chunk size')
    parser.add_argument('--left_chunks', default=-1,
                        type=int, help='left chunks')
    parser.add_argument('--font', required=True,
                        type=str, help='font file')
    parser.add_argument('--dict', required=True,
                        type=str, help='dict file')
    parser.add_argument('--result_dir', required=True,
                        type=str, help='saving pdf')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    torch.manual_seed(777)

    symbol_table = read_symbol_table(args.dict)
    char_dict = {v: k for k, v in symbol_table.items()}

    # 1. Load model
    with open(args.config, 'r') as fin:
        conf = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_asr_model(conf)
    load_checkpoint(model, args.ckpt)
    model.eval().cuda()

    subsampling = model.encoder.embed.subsampling_rate
    eos = model.eos_symbol()

    with open(args.wavscp, 'r') as fin:
        wavs = fin.readlines()

    # 2. Forward model (get streaming_timestamps)
    timestamps = {}
    for idx, wav in enumerate(wavs):
        if idx % 100 == 0:
            logging.info("processed {}.".format(idx))
        key, wav = wav.strip().split(' ', 1)
        waveform, sr = torchaudio.load(wav)
        resample_rate = conf['dataset_conf']['resample_conf']['resample_rate']
        waveform = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=resample_rate)(waveform)
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(
            waveform,
            num_mel_bins=conf['dataset_conf']['fbank_conf']['num_mel_bins'],
            frame_length=conf['dataset_conf']['fbank_conf']['frame_length'],
            frame_shift=conf['dataset_conf']['fbank_conf']['frame_shift'],
            dither=0.0, energy_floor=0.0,
            sample_frequency=resample_rate,
        )

        # CTC greedy search
        speech = mat.unsqueeze(0).cuda()
        speech_lengths = torch.tensor([mat.size(0)]).cuda()
        # Let's assume batch_size = 1
        encoder_out, encoder_mask = model._forward_encoder(
            speech, speech_lengths, args.chunk_size, args.left_chunks,
            simulate_streaming=False)
        maxlen = encoder_out.size(1)  # (B, maxlen, encoder_dim)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = model.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(1, maxlen)  # (B, maxlen)
        topk_prob = topk_prob.view(1, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, eos)  # (B, maxlen)
        topk_prob = topk_prob.masked_fill_(mask, 0.0)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        hyps = [replace_duplicates_with_blank(hyp) for hyp in hyps]
        scores = [prob.tolist() for prob in topk_prob]
        timestamps[key] = [hyps[0], scores[0], wav]

    # 3. Analyze latency
    with open(args.alignment, 'r') as fin:
        aligns = fin.readlines()
    not_found, len_unequal, ignored = 0, 0, 0
    datas = []
    for align in aligns:
        key, align = align.strip().split(' ', 1)
        if key not in timestamps:
            not_found += 1
            continue
        fa, st = [], []  # force_alignment, streaming_timestamps
        text_fa, text_st = "", ""
        for i, token in enumerate(align.split()):
            if token != '<blank>':
                text_fa += token
                # NOTE(xcsong): W/O subsample
                fa.append(i * 10)
        # ignore alignment_errors >= 70ms
        frames_fa = len(align.split())
        frames_st = len(timestamps[key][0]) * subsampling
        if abs(frames_st - frames_fa) >= 7:
            ignored += 1
            continue
        for i, token_id in enumerate(timestamps[key][0]):
            if token_id != 0:
                text_st += char_dict[token_id]
                # NOTE(xcsong): W subsample
                st.append(i * subsampling * 10)
        if len(fa) != len(st):
            len_unequal += 1
            continue
        # datas[i] = [key, text_fa, text_st, list_of_diff,
        #             FirstTokenDelay, LastTokenDelay, AvgTokenDelay,
        #             streaming_timestamps, force_alignment]
        datas.append([key, text_fa, text_st,
                     [a - b for a, b in zip(st, fa)],
                     st[0] - fa[0], st[-1] - fa[-1],
                     (sum(st) - sum(fa)) / len(st),
                     timestamps[key], align.split()])

    logging.info("not found: {}, length unequal: {}, ignored: {}, \
        valid samples: {}".format(not_found, len_unequal, ignored, len(datas)))

    # 4. Plot and print
    num_datas = len(datas)
    names = ['FirstTokenDelay', 'LastTokenDelay', 'AvgTokenDelay']
    names_index = [4, 5, 6]
    parts = ['max', 'P90', 'P75', 'P50', 'P25', 'min']
    parts_index = [num_datas - 1, int(num_datas * 0.90), int(num_datas * 0.75),
                   int(num_datas * 0.50), int(num_datas * 0.25), 0]
    for name, name_idx in zip(names, names_index):
        def f(name_idx=name_idx):
            return name_idx
        datas.sort(key=lambda x: x[f()])
        logging.info("==========================")
        for p, i in zip(parts, parts_index):
            data = datas[i]
            # i.e., LastTokenDelay P90: 270.000 ms (wav_id: BAC009S0902W0144)
            logging.info("{} {}: {:.3f} ms (wav_id: {})".format(
                name, p, data[f()], datas[i][0]))

            font = fm.FontProperties(fname=args.font)
            plt.rcParams['axes.unicode_minus'] = False
            # we will have 2 sub-plots (force-align + streaming timestamps)
            # plus one wav-plot
            fig, axes = plt.subplots(figsize=(60, 60), nrows=3, ncols=1)
            for j in range(2):
                if j == 0:
                    # subplot-0: streaming_timestamps
                    plt_prefix = args.tag + "_" + name + "_" + p
                    x = np.arange(len(data[7][0])) * subsampling
                    hyps, scores = data[7][0], data[7][1]
                else:
                    # subplot-1: force_alignments
                    plt_prefix = "force_alignment"
                    x = np.arange(len(data[8]))
                    hyps = [symbol_table[d] for d in data[8]]
                    scores = [0.0] * len(data[8])
                axes[j].set_title(plt_prefix, fontsize=30)
                for frame, token, prob in zip(x, hyps, scores):
                    if char_dict[token] != '<blank>':
                        axes[j].bar(
                            frame, np.exp(prob),
                            label='{} {:.3f}'.format(
                                char_dict[token], np.exp(prob)),
                        )
                        axes[j].text(
                            frame, np.exp(prob),
                            '{} {:.3f} {}'.format(
                                char_dict[token], np.exp(prob), frame),
                            fontdict=dict(fontsize=24),
                            fontproperties=font,
                        )
                    else:
                        axes[j].bar(
                            frame, 0.01,
                            label='{} {:.3f}'.format(
                                char_dict[token], np.exp(prob)),
                        )
                axes[j].tick_params(labelsize=25)

            # subplot-2: wav
            # wav, hardcode sample_rate to 16000
            samples, sr = librosa.load(data[7][2], sr=16000)
            time = np.arange(0, len(samples)) * (1.0 / sr)
            axes[-1].plot(time, samples)

            # i.e., RESULT_DIR/BAC009S0768W0342_LTD_P90_120ms.pdf
            plt.savefig(args.result_dir + "/" +
                        data[0] + "_" + name +
                        "_" + p + "_" + str(data[f()]) + "ms.pdf")


if __name__ == '__main__':
    main()
