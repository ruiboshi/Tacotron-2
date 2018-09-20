import argparse
import time
import sys
import os

import tensorflow as tf
import numpy as np
import librosa
from hparams import hparams
from tacotron.utils.text import text_to_sequence
from wavenet_vocoder.synthesize import Synthesizer as WaveSynthesizer
from wavenet_vocoder import util

def _get_config_proto(fraction=None):
    config = tf.ConfigProto()
    if fraction is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = fraction
    return config


def _pad_inputs(x, maxlen, _pad=0):
	return np.pad(x, [(0, maxlen - len(x)), (0, 0)], mode='constant', constant_values=_pad)

def _pad_input(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=0)

def _load_graph(path_to_graph_def):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(path_to_graph_def, 'rb') as f:
        graph_def.ParseFromString(f.read())
    
    with graph.as_default():
        tf.import_graph_def(graph_def)
    
    return graph

def load_tacotron(path_to_tacotron_graph):
    return _load_graph(path_to_tacotron_graph)

def load_wavenet(path_to_wavenet_graph):
    return _load_graph(path_to_wavenet_graph)
    
def preprocess_text(texts):
    seqs = [np.array(text_to_sequence(text, ['english_cleaners'])) for text in texts]
    input_legths = [seq.shape[0] for seq in seqs]
    max_len = max(input_legths)
    seqs = np.stack([_pad_input(x, max_len) for x in seqs])
    return seqs, input_legths

def apply_tacotron(texts, output_dir=None, device=None):
    with tf.device(device):
        graph = load_tacotron

class TacotronPart:
    def __init__(
            self, 
            path_to_tacotron_graph, 
            volatile=True,
            input_tensor_names=['import/inputs', 'import/input_lengths'],
            output_tensor_names=['import/model/inference/add']):
        self.path_to_graph_def = path_to_tacotron_graph
        self.volatile = volatile
        self._graph = load_tacotron(path_to_tacotron_graph)
        self.input_op_1 = self._graph.get_operation_by_name(input_tensor_names[0])
        self.input_op_2 = self._graph.get_operation_by_name(input_tensor_names[1])
        self.output_op = self._graph.get_operation_by_name(output_tensor_names[0])
        if not volatile:
            self._sess = tf.Session(graph=self._graph, config=_get_config_proto())
        else:
            self._sess = None
        
    def text_to_mel(self, texts):
        if self._sess is None:
            self._sess = tf.Session(graph=self._graph, config=_get_config_proto())
        
        time_1 = time.time()
        seqs, input_lengths = preprocess_text(texts)
        mels = self._sess.run(
            self.output_op.outputs[0], 
            {
                self.input_op_1.outputs[0] : seqs,
                self.input_op_2.outputs[0] : input_lengths
            }
            )
        time_2 = time.time()

        if self.volatile:
            self._sess.close()

        return mels

def process_mels(mels, hop_size=300, taco_max_value=4.0, symmetric=False, normalise_for_wavenet=True):
    audio_length = [len(x) * hop_size for x in mels]
    maxlen = max([len(x) for x in mels])
    T2_output_range = (-taco_max_value, taco_max_value) if symmetric else (0.0, taco_max_value)
    
    padded_batch = np.stack([_pad_inputs(x, maxlen, _pad=T2_output_range[0]) for x in mels]).astype(np.float32)

    if normalise_for_wavenet:
        padded_batch = np.interp(padded_batch, T2_output_range, (0, 1))
    
    return audio_length, padded_batch

class WavenetPart:
    def __init__(
            self, 
            path_to_wavenet_graph, 
            volatile=True,
            input_tensor_names=['import/local_condition_features'],
            output_tensor_names=['import/model/inference/Reshape_1']):
        self.path_to_graph_def = path_to_wavenet_graph
        self.volatile = volatile
        self._graph = load_wavenet(path_to_wavenet_graph)
        self.input_op = self._graph.get_operation_by_name(input_tensor_names[0])
        self.output_op = self._graph.get_operation_by_name(output_tensor_names[0])

        if not volatile:
            self._sess = tf.Session(graph=self._graph, config=_get_config_proto())
        else:
            self._sess = None
        
    def mel_to_wav(self, mels):
        if self._sess is None:
            self._sess = tf.Session(graph=self._graph, config=_get_config_proto())
        
        time_1 = time.time()

        audio_lengths, b_mels = process_mels(mels)

        wavs = self._sess.run(
            self.output_op.outputs[0], 
            feed_dict=
            {
                self.input_op.outputs[0] : b_mels
            }
            )
        time_2 = time.time()

        if hparams.input_type == 'mulaw-quantize':
            wavs = util.inv_mulaw_quantize(wavs, mu=hparams.quantize_channels)

        wavs = [wav[:length] for wav, length in zip(wavs, audio_lengths)]

        #log('WaveNet synthesise {} samples in {:.2f} seconds'.format(np.array(wavs.shape, time_2 - time_1))
        if self.volatile:
            self._sess.close()

        return wavs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='texts, separated by | (pipe)')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--taco_graph', type=str)
    parser.add_argument('--wavenet_graph', type=str)

    args = parser.parse_args()

    texts = args.text.split('|')
    print('Texts:       {}'.format(texts))

    basenames = list(range(len(texts)))

    if not os.path.exists(args.taco_graph):
        print('Tacotron frozen graph cannot be found at {}'.format(args.taco_graph))
        sys.exit(-1)

    if not os.path.exists(args.wavenet_graph):
        print('Wavenet frozen graph cannot be found at {}'.format(args.wavenet_graph))
        sys.exit(-1)
    
    os.makedirs(args.output_dir, exist_ok=True)

    taco = TacotronPart(args.taco_graph)
    wavenet = WavenetPart(args.wavenet_graph)

    t_taco_1 = time.time()
    mels = taco.text_to_mel(texts)
    t_taco_2 = time.time()

    print('************ Tacotron ************')
    print('Mels:        {}'.format(mels.shape))
    print('Time:        {:.3f} secs'.format(t_taco_2 - t_taco_1))

    t_wave_1 = time.time()
    wavs = wavenet.mel_to_wav(mels)
    t_wave_2 = time.time()

    for filename, wav in zip(basenames, wavs):
        path = os.path.join(args.output_dir, filename)
        librosa.output.write_wav(path, wav, sr=hparams.sample_rate)

    audio_lengths = [len(x) * hparams.hop_size for x in mels]

    print('************ WaveNet *************')
    print('Time:        {:.3f} secs'.format(t_wave_2 - t_wave_1))
    print('Output:      {}'.format(args.output_dir))
    print('Lengths:     {}'.format(audio_lengths))
