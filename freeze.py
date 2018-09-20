import argparse
import os

import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

from hparams import hparams
from tacotron.synthesize import Synthesizer as TacoSynthesizer
from wavenet_vocoder.synthesize import Synthesizer as WaveSynthesizer

def _transform_ops():
    return [
        'add_default_attributes',
        'remove_nodes(op=Identity, op=CheckNumrics)',
        'fold_batch_norms',
        'fold_old_batch_norms',
        'strip_unused_nodes',
        'sort_by_execution_order'
    ]

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

    tf.train.write_graph(synth.session.graph, output_dir, 'taco_variable_graph_def.pb', as_text=False)
    
    transformed_graph_def = TransformGraph(
        synth.session.graph.as_graph_def(), 
        inputs=_get_node_name([synth.model.inputs, synth.model.input_lengths]),
        outputs=_get_node_name([synth.mel_outputs]),
        transforms=_transform_ops()
    )

    const_graph_def = graph_util.convert_variables_to_constants(
        synth.session,
        transformed_graph_def,
        _get_node_name([synth.mel_outputs])
    )

    print('input_tensors: {}'.format(_get_node_name([synth.model.inputs, synth.model.input_lengths])))
    print('output_tensors: {}'.format(_get_node_name([synth.mel_outputs])))

    try:
        optimize_for_inference_lib.ensure_graph_is_valid(const_graph_def)
        tf.train.write_graph(const_graph_def, output_dir, 'optimized_frozen_tacotron.pb', as_text=False)
    except ValueError as e:
        print('Graph is invalid - {}'.format(e))
    
def WaveNetFreezer(checkpoint_path, hparams, output_dir):
    tf.reset_default_graph()
    synth = WaveSynthesizer()
    synth.load(checkpoint_path, hparams)

    tf.train.write_graph(synth.session.graph, output_dir, 'wavenet_variable_graph_def.pb', as_text=False)

    transformed_graph_def = TransformGraph(
        synth.session.graph.as_graph_def(), 
        inputs=_get_node_name([synth.local_conditions]),
        outputs=_get_node_name([synth.model.out_node]),
        transforms=_transform_ops()
    )

    const_graph_def = graph_util.convert_variables_to_constants(
        synth.session,
        transformed_graph_def,
        _get_node_name([synth.model.out_node])
    )

    print('input_tensors: {}'.format(_get_node_name([synth.local_conditions])))
    print('output_tensors: {}'.format(_get_node_name([synth.model.out_node])))

    os.makedirs(output_dir, exist_ok=True)

    try:
        optimize_for_inference_lib.ensure_graph_is_valid(const_graph_def)
        tf.train.write_graph(const_graph_def, output_dir, 'optimized_frozen_wavenet.pb', as_text=False)

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