#!/usr/bin/python3


import os


import matplotlib.pyplot as plt
import keras.models
import tensorflow as tf


def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)


def get_label(file_path):
  parts = tf.strings.split(
    input=file_path,
    sep=os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]


def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
    [16000] - tf.shape(waveform),
    dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
    equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram



data_dir = './word_data'
langtypes = tf.io.gfile.listdir(str(data_dir))

input_shape = (124, 129, 1)
model = keras.models.load_model('spectromaster_v4.h5')

model.summary()
print('Load model, so we do not have to ALWAYs retrain!')

def test_file(filename):
  sample_file = data_dir+filename
  target_wav, target_label = get_waveform_and_label(str(sample_file))
  target_spectrogram = get_spectrogram(target_wav)
  
  target_shape = tf.shape(target_spectrogram)
  print(f'target_shape = {target_shape}')
  new_spectrogram = target_spectrogram[None, ...]
  target_shape = tf.shape(new_spectrogram)
  print(f'target_shape = {target_shape}')
  
  prediction = model(new_spectrogram)
  print(f'Prediction is {prediction}')
  simple_prediction = tf.nn.softmax(prediction[0])
  print(f'Simple Prediction is {simple_prediction}')
  print(f'Target Label {target_label}')
  # plt.bar(langtypes, tf.nn.softmax(prediction[0]))
  # plt.title(f'Predictions for {target_label}')
  # plt.show()

def test_pairs(l1, l2):

  test_file(f'/{l1}_wave1/hospital-{l1}-def.wav')
  test_file(f'/{l2}_wave1/hospital-{l2}-def.wav')
  
  
  test_file(f'/{l1}_wave1/bed-{l1}-def.wav')
  test_file(f'/{l2}_wave1/bed-{l2}-def.wav')
  
  test_file(f'/{l1}_wave1/hospital-{l1}-def.wav')
  test_file(f'/{l2}_wave1/hospital-{l2}-def.wav')
  
test_pairs('en','fr')