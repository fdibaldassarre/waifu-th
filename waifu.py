#!/usr/bin/env python3

import os
import sys
import argparse

from src import Waifu

parser = argparse.ArgumentParser(description="Waifu2x")
parser.add_argument('--input', '-i', dest='input', default=None,
                    help='Input image')
parser.add_argument('--output', '-o', dest='output', default=None,
                    help='Output image')
parser.add_argument('--model', '-m', dest='model', default='scale',
                    help='Supported operations: noise, noise_scale, scale')
parser.add_argument('--noise_level', '-n', dest='noise', default='1',
                    help='Noise level, supported values: 1 (low), 2 (medium), 3 (high)')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='optimize for CPU usage instead of GPU')

args = parser.parse_args()

input_path = args.input
output_path = args.output
model_type = args.model
noise_level = args.noise
use_cpu = args.cpu

# Check input
if input_path is None or not os.path.exists(input_path):
  print('Input image not found.')
  sys.exit(1)
elif output_path is None:
  print('Output savepath missing.')
  sys.exit(1)
elif not model_type in ['scale', 'noise_scale', 'noise']:
  print('Operation ' + model_type + ' not supported! Supported values are scale, noise_scale, noise.')
  sys.exit(1)
elif not noise_level in ['1', '2', '3']:
  print('Value ' + noise + ' is not valid. Valid values are 1, 2 and 3.')
  sys.exit(1)

w = Waifu.start(use_cpu=use_cpu)
if w.openImage(input_path) is None:
  print('Cannot open image: ' + input_path)
  sys.exit(2)
# Set operation
if model_type == 'scale':
  w.setOperationScale()
elif model_type == 'noise':
  w.setOperationNoise()
else:
  w.setOperationNoiseScale()
# Set noise level
w.setNoiseLevel(noise_level)
# Run
w.run()
# Save
w.saveOutput(output_path)
