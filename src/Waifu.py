#!/usr/bin/env python3

import os
import json

from multiprocessing import Pool

from PIL import Image
from PIL import ImageOps

import numpy

import lasagne
from lasagne.nonlinearities import LeakyRectify
leaky_rectify = LeakyRectify(0.1)

import theano
from theano import tensor as T

path = os.path.abspath(__file__)
SRC_FOLDER = os.path.dirname(path)
MAIN_FOLDER = os.path.dirname(SRC_FOLDER)
MODEL_FOLDER = os.path.join(MAIN_FOLDER, 'models/')

OPERATION_SCALE = 0
OPERATION_NOISE = 1
OPERATION_NOISE_SCALE = 2

BORDER_SIZE = 7

class PoolFunction():

  def __init__(self, theano_function):
    self.tf = theano_function
  
  def __call__(self, src):
    return self.tf(src[0]), src[1]


class Waifu():
  
  def __init__(self, model_folder=None, use_cpu=False):
    self.model_folder = model_folder if model_folder is not None else MODEL_FOLDER
    self.use_cpu = use_cpu

  def openImage(self, path):
    try:
      self.img = Image.open(path)
    except Exception as error:
      self.img = None
    return self.img
  
  def setNoiseLevel(self, lvl):
    self.noise_lvl = lvl
  
  def getImageChannels(self, img=None):
    if img is None:
      img = self.img
    if img.mode.startswith('L'):
      return 1
    else:
      return 3
  
  def imageHasAlpha(self):
    return self.img.mode.endswith('A')
  
  def setOperationScale(self):
    self.op = OPERATION_SCALE
    
  def setOperationNoise(self):
    self.op = OPERATION_NOISE
    
  def setOperationNoiseScale(self):
    self.op = OPERATION_NOISE_SCALE
  
  def getModel(self, op):
    if self.getImageChannels() == 1:
      return self.getModelBW(op)
    else:
      return self.getModelRGB(op)
  
  def getModelBW(self, op):
    model_folder = os.path.join(self.model_folder, 'anime_style_art')
    if op == OPERATION_SCALE:
      return os.path.join(model_folder, 'scale2.0x_model.json')
    else:
      return os.path.join(model_folder, 'noise' + str(self.noise_lvl) +'_model.json')
  
  def getModelRGB(self, op):
    model_folder = os.path.join(self.model_folder, 'anime_style_art_rgb')
    if op == OPERATION_SCALE:
      return os.path.join(model_folder, 'scale2.0x_model.json')
    elif op == OPERATION_NOISE:
      return os.path.join(model_folder, 'noise' + str(self.noise_lvl) +'_model.json')
    elif op == OPERATION_NOISE_SCALE:
      return os.path.join(model_folder, 'noise' + str(self.noise_lvl) +'_scale2.0x_model.json')
    else:
      return None
  
  ## Pre-processing
  def preprocessImageScale(self, img):
    # rescale
    w, h = img.size
    return img.resize((2*w, 2*h), Image.BICUBIC)
  
  def saveOutputAlpha(self, img):
    self.oalpha = img.split()[-1]
  
  ## Operations
  def run(self):
    if self.op == OPERATION_SCALE:
      self.rimg = self.runScale(self.img)
    elif self.op == OPERATION_NOISE:
      self.rimg = self.runNoise(self.img)
    else:
      self.rimg = self.runNoiseScale(self.img)
    return self.rimg
  
  def saveOutput(self, savepath):
    self.rimg.save(savepath)
  
  def runScale(self, img):
    if self.getImageChannels() == 1:
      pimg = self.preprocessImageScale(img)
      double_size = False
    else:
      pimg = img
      double_size = True
    # save image alpha
    if self.imageHasAlpha():
      self.saveOutputAlpha(pimg)
    models, rdata = self.convertImageToTensor(pimg)
    transformed = self.applyFunction(models, OPERATION_SCALE)
    return self.recomposeImage(transformed, rdata, double_size)
  
  def runNoise(self, img):
    # save image alpha
    if self.imageHasAlpha():
      self.saveOutputAlpha(img)
    models, rdata = self.convertImageToTensor(img)
    transformed = self.applyFunction(models, OPERATION_NOISE)
    return self.recomposeImage(transformed, rdata)
 
  def runNoiseScale(self, img):
    if self.getImageChannels() == 1:
      return self.runNoiseScaleBW(img)
    else:
      return self.runNoiseScaleRGB(img)
      
  def runNoiseScaleBW(self, img):
    # Denoise
    dimg = self.runNoise(img)
    # Scale
    return self.runScale(dimg)
  
  def runNoiseScaleRGB(self, img):
    if self.imageHasAlpha():
      self.saveOutputAlpha(img)
    models, rdata = self.convertImageToTensor(img)
    transformed = self.applyFunction(models, OPERATION_NOISE_SCALE)
    return self.recomposeImage(transformed, rdata, double_size=True)
  
  ## Apply function
  def applyFunction(self, *args, **kwargs):
    if self.use_cpu:
      return self.applyFunctionCPU(*args, **kwargs)
    else:
      return self.applyFunctionGPU(*args, **kwargs)
  
  def applyFunctionGPU(self, models, op):
    tfunct = self.createTheanoFunction(op)
    results = []
    for data in models:
      model, pos = data
      tr = tfunct(model)
      results.append((tr, pos))
    return results
  
  def applyFunctionCPU(self, models, op):
    tfunct = self.createTheanoFunction(op)
    mfunc = PoolFunction(tfunct)
    with Pool() as p:
      results = p.map(mfunc, models)
    return results
  
  def createTheanoFunction(self, op):
    model = self.loadModel(op)
    model_config = model[0]['model_config']
    arch_name = model_config['arch_name']
    if arch_name == 'vgg_7':
      return self.createTheanoVgg7(model)
    elif arch_name == 'upconv_7':
      return self.createTheanoUpConv7(model)
    else:
      return None
  
  def createTheanoVgg7(self, model):
    input = T.tensor4(name='input')
    network = lasagne.layers.InputLayer((None, self.getImageChannels(), None, None), input)
    i = 1
    for layer in model:
      n_output = layer['nOutputPlane']
      n_input = layer['nInputPlane']
      # f = convolution width and height (I assume f_h == f_w)
      f_h = layer['kH']
      f_w = layer['kW']
      # weights and bias
      weights = numpy.asarray(layer['weight'], dtype=input.dtype)
      bias = numpy.asarray(layer['bias'], dtype=input.dtype)
      w = theano.shared(weights, name="W"+str(i), borrow=True)
      b = theano.shared(bias, name="b"+str(i), borrow=True)
      if i < 7:
        network = lasagne.layers.Conv2DLayer(network, n_output, (f_w, f_h),
                                             nonlinearity=leaky_rectify,
                                             W=w, b=b)
      else:
        network = lasagne.layers.Conv2DLayer(network, n_output, (f_w, f_h),
                                             nonlinearity=None,
                                             W=w, b=b)
      i += 1
    f = theano.function([input], output)
    return f
  
  def createTheanoUpConv7(self, model):
    input = T.tensor4(name='input')
    network = lasagne.layers.InputLayer((None, self.getImageChannels(), None, None), input)
    i = 1
    for layer in model:
      n_output = layer['nOutputPlane']
      n_input = layer['nInputPlane']
      # f = convolution width and height (I assume f_h == f_w)
      f_h = layer['kH']
      f_w = layer['kW']
      # d = convolution step
      if 'dH' in layer:
        d_h = layer['dH']
        d_w = layer['dW']
      else:
        d_h, d_w = 1, 1
      # weights and bias
      weights = numpy.asarray(layer['weight'], dtype=input.dtype)
      bias = numpy.asarray(layer['bias'], dtype=input.dtype)
      w = theano.shared(weights, name="W"+str(i), borrow=True)
      b = theano.shared(bias, name="b"+str(i), borrow=True)
      if i < 7:
        network = lasagne.layers.Conv2DLayer(network, n_output, (f_w, f_h),
                                             nonlinearity=leaky_rectify,
                                             W=w, b=b)
      else:
        # Deconvolution / Full convolution
        network = lasagne.layers.TransposedConv2DLayer(network, n_output, (f_w, f_h),
                                                       stride=(d_w, d_h),
                                                       nonlinearity=None,
                                                       W=w,
                                                       flip_filters=False,
                                                       crop='full')
      i += 1
    output = lasagne.layers.get_output(network)
    f = theano.function([input], output)
    return f
  
  def loadModel(self, op):
    decoder = json.JSONDecoder()
    model_path = self.getModel(op)
    with open(model_path, 'r') as hand:
      tmp = []
      for line in hand:
        line = line.strip()
        tmp.append(line)
    tmp = ''.join(tmp)
    model = decoder.decode(tmp)
    return model
  
  ## Conversions
  def convertImageToTensor(self, img):
    # Convert to RGB or B/W
    if self.getImageChannels(img) == 1:
      img = img.convert('L')
      channels = 1
    else:
      img = img.convert('RGB')
      channels = 3
    # Split image
    imgs, rdata = self.splitImage(img)
    # Convert to 4d tensors
    tensors = []
    for data in imgs:
      img, pos = data
      w, h = img.size
      arr = numpy.asarray(img, dtype=numpy.float32) / 255.
      if channels == 3:
        # img shape = h, w, channels
        # tensor shape = channels, h, w
        arr = arr.transpose(2, 0, 1)
      model = arr.reshape(1, channels, h, w)
      tensors.append((model, pos))
    return tensors, rdata
  
  def splitImage(self, img):
    # Get best subsize between 200 and 350
    w, h = img.size
    orig_size = (w, h)
    if (-w%200)*(-h%200) < (-w%350)*(-h%350):
      subsize = 200
    else:
      subsize = 350
    # Set sizes multiples of subsize
    n_w = int(numpy.ceil(w/subsize) * subsize)
    n_h = int(numpy.ceil(h/subsize) * subsize)
    new_img = Image.new('RGB', (n_w, n_h), 'black')
    new_img.paste(img, (0,0))
    # Add border size
    new_img = ImageOps.expand(new_img, border=BORDER_SIZE, fill='black')
    # Split
    # We want images of size (subsize+2*BORDER_SIZE)x(subsize+2*BORDER_SIZE)
    imgs = []
    images_on_width = int(n_w/subsize)
    images_on_height = int(n_h/subsize)
    rdata = (orig_size, subsize, images_on_width, images_on_height)
    for i in range(images_on_width):
      for j in range(images_on_height):
        left = i*subsize
        right = (i+1)*subsize + 2*BORDER_SIZE
        top = j*subsize
        bottom = (j+1)*subsize + 2*BORDER_SIZE
        simg = new_img.crop((left, top, right, bottom))
        imgs.append((simg, (i,j)))
    return imgs, rdata
  
  def recomposeImage(self, models, rdata, double_size=False):
    if double_size:
      mult = 2
    else:
      mult = 1
    # Get reconstruct data
    orig_size, subsize, images_on_width, images_on_height = rdata
    # Create tmp image
    channels = self.getImageChannels()
    mode = 'L' if channels == 1 else 'RGB'
    tw, th = mult*subsize*images_on_width, mult*subsize*images_on_height
    tmp_img = Image.new(mode, (tw,th), 'black')
    # Compose tmp image from models
    i = 0
    for data in models:
      model, pos = data
      # Conversion tensor --> PIL.Image
      f_img = model[0, :, :, :]
      # Reshape as the original image
      if channels == 3:
        # tensor shape = channels, h, w
        # image shape = h, w, channels
        f_img = f_img.transpose(1, 2, 0)
      else:
        f_img = f_img[0, :, :]
      # Rescale to [0, 255]
      r_img = numpy.minimum(numpy.maximum(0., f_img), 1.)
      r_img = numpy.uint8(numpy.round(r_img * 255.))
      mimg = Image.fromarray(r_img)
      # Paste in tmp image
      i,j = pos
      x,y = mult*i*subsize, mult*j*subsize
      tmp_img.paste(mimg, (x,y))
    # Crop to original size
    w,h = orig_size
    fimg = tmp_img.crop((0,0,mult*w,mult*h))
    # Add alpha
    if self.imageHasAlpha():
      if double_size:
        self.oalpha = self.oalpha.resize((2*w, 2*h), Image.BICUBIC)
      fimg.putalpha(self.oalpha)
    return fimg   
  
def start(*args, **kwargs):
  w = Waifu(*args, **kwargs)
  return w
