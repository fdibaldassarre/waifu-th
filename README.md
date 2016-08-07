# waifu-th
Reimplementation of [Waifu2x](https://github.com/nagadomi/waifu2x) in Python using Theano and Lasagne.

## Requirements

- Python3
- Python Imaging Library (PIL)
- [Theano](http://www.deeplearning.net/software/theano/install.html)
- [Lasagne](https://github.com/Lasagne/Lasagne)
- The [models folder](https://github.com/nagadomi/waifu2x/tree/master/models) from waifu2x

## Usage

To upscale and clean an image run:
```sh
./waifu.py -i noisy_image.png -m noise_scale --noise-level 1 -o new_image.png
```

To see all the available options type:
```sh
./waifu.py --help
```

## Running on the GPU

To run the program on the GPU (much faster) use:

```sh
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
```

see http://www.deeplearning.net/software/theano/install.html#using-the-gpu for more info.

## TODO

Implement training.

More options.
