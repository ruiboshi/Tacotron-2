import argparse
import os

import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

from hparams import hparams
from tacotron.synthesize import Synthesizer as TacoSynthesizer
from wavenet_vocoder.synthesize import Synthesizer as WaveSynthesizer


def _get_node_name(tensors):
    if isinstance(tensors, list):
        return [tensor.name.split(':')[0] for tensor in tensors]
    else:
        tensors.name.split(':')[0]

def TacotronFreezer(checkpoint_path, hparams, output_dir):
    tf.reset_default_graph()
    synth = TacoSynthesizer()
    synth.load(checkpoint_path, hparams)

    os.makedirs(output_dir, exist_ok=True)

    const_graph_def = graph_util.convert_variables_to_constants(
        synth.session,
        synth.session.graph.as_graph_def(),
        _get_node_name([synth.mel_outputs])
    )

    optimized_const_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def=const_graph_def,
        input_node_names=_get_node_name([synth.model.inputs, synth.model.input_lengths]),
        output_node_names=_get_node_name([synth.mel_outputs]),
        placeholder_type_enum=tf.float32.as_datatype_enum,
        toco_compatible=False
    )

    optimized_const_graph_def = optimize_for_inference_lib.fold_batch_norms(optimized_const_graph_def)

    try:
        optimize_for_inference_lib.ensure_graph_is_valid(optimized_const_graph_def)
        tf.train.write_graph(optimized_const_graph_def, output_dir, 'frozen_tacotron.pb', as_text=False)
    except ValueError as e:
        print('Graph is invalid - {}'.format(e))
    
def WaveNetFreezer(checkpoint_path, hparams, output_dir):
    tf.reset_default_graph()
    synth = WaveSynthesizer()
    synth.load(checkpoint_path, hparams)

    const_graph_def = graph_util.convert_variables_to_constants(
        synth.session,
        synth.session.graph.as_graph_def(),
        _get_node_name([synth.model.y_hat])
    )

    optimized_const_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def=const_graph_def,
        input_node_names=_get_node_name([synth.local_conditions]),
        output_node_names=_get_node_name([synth.model.y_hat]),
        placeholder_type_enum=tf.float32.as_datatype_enum,
        toco_compatible=False
    )

    optimized_const_graph_def = optimize_for_inference_lib.fold_batch_norms(optimized_const_graph_def)

    os.makedirs(output_dir, exist_ok=True)

    try:
        optimize_for_inference_lib.ensure_graph_is_valid(optimized_const_graph_def)
        tf.train.write_graph(optimized_const_graph_def, output_dir, 'frozen_wavenet.pb', as_text=False)
    except ValueError as e:
        print('Graph is invalid - {}'.format(e))

def main():
    accepted_models = ['Tacotron', 'WaveNet', 'Tacotron-2']
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Tacotron-2')
    parser.add_argument('--wavenet_checkpoint', default='', help='Path to trained wavenet model')
    parser.add_argument('--tacotron_checkpoint', default='', help='Path to trained tacotron model')
    parser.add_argument('--output_dir', default='./saved_graph', help='Path to save the frozen graphs')

    args = parser.parse_args()

    models = []

    if args.model not in accepted_models:
        raise ValueError('Please enter a valid model to freeze: {}'.format(accepted_models))
    
    if args.model == 'Tacotron':
        models.append('Tacotron')
    elif args.model == 'WaveNet':
        models.append('WaveNet')
    elif args.model == 'Tacotron-2':
        models.extend(['Tacotron', 'WaveNet'])
        
    for model in models:
        if model == 'Tacotron':
            try:
                checkpoint_path = tf.train.get_checkpoint_state(args.tacotron_checkpoint).model_checkpoint_path
                print('loaded tacotron model checkpoint at {}'.format(checkpoint_path))
            except:
                ss = 'WARNING: freezing Tacotron graph without'
                ss += 'converting pretrained weights as no checkpoint is supplied'
                raise RuntimeError(ss + args.tacotron_checkpoint)
            
            TacotronFreezer(checkpoint_path, hparams, args.output_dir)

        elif model == 'WaveNet':
            try:
                checkpoint_path = tf.train.get_checkpoint_state(args.wavenet_checkpoint).model_checkpoint_path
                print('loaded WaveNet model checkpoint at {}'.format(checkpoint_path))
            except:
                ss = 'WARNING: freezing WaveNet graph without'
                ss += 'converting pretrained weights as no checkpoint is supplied'
                raise RuntimeError(ss + args.wavenet_checkpoint)
            
            WaveNetFreezer(checkpoint_path, hparams, args.output_dir)
        
        else:
            print('Model {} unknown, skipped...'.format(model))

if __name__ == '__main__':
    main()