#!/usr/bin/python3
import copy
import os


import keras.models
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import struct
import tensorflow as tf
import wave

DEBUG = False
# from ctypes import *
#
# # From alsa-lib Git 3fd4ab9be0db7c7430ebd258f2717a976381715d
# # $ grep -rn snd_lib_error_handler_t
# # include/error.h:59:typedef void (*snd_lib_error_handler_t)(const char *file, int line, const char *function, int err, const char *fmt, ...) /* __attribute__ ((format (printf, 5, 6))) */;
# # Define our error handler type
# ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
# def py_error_handler(filename, line, function, err, fmt):
#   print('messages are yummy')
# c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
#
# asound = cdll.LoadLibrary('libasound.so')
# # Set error handler
# asound.snd_lib_error_set_handler(c_error_handler)

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def replay_audio(self, filename):
  # length of data to read.
  chunk = 1024
  wf = wave.open(filename, 'rb')
  
  # create an audio object
  p = pyaudio.PyAudio()
  
  # open stream based on the wave object which has been input.
  stream = p.open(format=
                  p.get_format_from_width(wf.getsampwidth()),
                  channels=wf.getnchannels(),
                  rate=wf.getframerate(),
                  output=True)
  
  # read data (based on the chunk size)
  data = wf.readframes(chunk)
  
  # play stream (looping from beginning of file to the end)
  while data != '':
    # writing to the stream is what *actually* plays the sound.
    stream.write(data)
    data = wf.readframes(chunk)
  
  # cleanup stuff.
  stream.close()
  p.terminate()


def plot_wav(wav_data):
  CHUNK = 1024
  # count = 0
  # stop = len(wav_data) * 2 - 2
  all_frames = tuple()
  for wav_chunks in wav_data:
    super_fine_frame = struct.unpack(str(CHUNK) + 'h', wav_chunks)
    all_frames = all_frames + super_fine_frame
    #print(f'{type(super_fine_frame)} --- {super_fine_frame}')
    # count = count + 2
    
    #print("* done recording")

  #print(f'Len superframe {len(super_fine_frame)}  --- len all {len(all_frames)}')
  ax = plt.subplot()
  ax.autoscale(enable=False, axis='y', tight=None)
  ax.set_yticks(np.arange(-32767, 32767, 1024))
  ax.plot(all_frames)
  ax.set_title('wav, ride it!')
  plt.show(block=False)
  plt.pause(0.3)
  plt.close()
    
def record_audio(wav_filename):
  CHUNK = 1024
  FORMAT = pyaudio.paInt16
  CHANNELS = 1
  RATE = 24000
  RECORD_SECONDS = 1.6
  WAVE_OUTPUT_FILENAME = wav_filename
  
  p = pyaudio.PyAudio()
  
  stream = p.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)
  
  #print("* recording")
  spike_value = 200
  trigger_width = int(CHUNK / 128)
  frames = []
  i = 0
  triggered = False
  
  while i < int(RATE / CHUNK * RECORD_SECONDS):
    data = stream.read(CHUNK)
    frames.append(data)
    last_data = b''
    if i % 2 != 1:
      super_frame = last_data + data
      super_fine_frame = struct.unpack(str(CHUNK) + 'h', super_frame)
      spike_count = 0
      for entry in super_fine_frame:
        if abs(entry) > spike_value:
          spike_count += 1
          #print(f'SPIKE ! # {spike_count}')
        if not triggered:
          if spike_count < trigger_width:
            i = 1
            frames = []
            #print('.', end='')
          else:
            triggered = True
            #print('Trigger recording...')
            #print(f'Spike Count {spike_count} trigger width {trigger_width}')
          
    i += 1
  
  #plot_wav(frames)
  
  stream.stop_stream()
  stream.close()
  p.terminate()
  
  wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(RATE)
  wf.writeframes(b''.join(frames))
  wf.close()


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

input_shape = (124, 129, 1)
model = keras.models.load_model('spectromaster_vEng_vs_Malay.h5')

model.summary()
print('Load model, so we do not have to ALWAYs retrain!')

def test_file(filename):
  sample_file = data_dir+filename
  target_wav, target_label = get_waveform_and_label(str(sample_file))
  target_spectrogram = get_spectrogram(target_wav)
  
  new_spectrogram = target_spectrogram[None, ...]
 
  prediction = model(new_spectrogram)
  val_predict = tf.math.argmax(prediction, axis=1)
  simple_prediction = tf.nn.softmax(prediction[0])
  if DEBUG:
    print(f'Simple Prediction is {simple_prediction} val {val_predict}')
  #print(f'Target Label {target_label}')
  
  #print(f'VAL_PREDICT {val_predict}')
  language = val_predict[0]
  #langtypes = ['Malay', 'English', 'Ukranian', 'Indonesian', 'French', 'Russian']
  langtypes = ['Malay', 'English']

  prediction_pretty = [f'{prediction:.2f}' for prediction in simple_prediction]
  if simple_prediction[language] > 0.65:
    print(f'{langtypes[language]} <<< -- {prediction_pretty}' )
    #replay_audio(f'{language}_id_file.wav')
  else:
    print(f'--no detect-- {prediction_pretty}')

  return target_spectrogram
  # plt.bar(langtypes, tf.nn.softmax(prediction[0]))
  # plt.title(f'Predictions for {target_label}')
  # plt.show()


def graph_9(list_of_spectro):
  rows = 3
  cols = 3
  n = rows*cols
  fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
  count = 0
  for target_spectrogram in list_of_spectro:
    r = count // cols
    c = count % cols
    ax = axes[r][c]
    plot_spectrogram(target_spectrogram.numpy(), ax)
    ax.set_title('Spectrogram')
    ax.axis('off')
    count = count + 1

  plt.show(block=False)
  plt.pause(0.1)
  plt.close()

while True:
  #record_audio('word_data/buffer.wav')
  #test_file('/buffer.wav')
  spectros = []
  for count in range(0, 8):
    filename = f'buffer{count}.wav'
    record_audio(f'word_data/{filename}')
    spectro = test_file(f'/{filename}')
    spectros.append(spectro)
  # spectroA = test_file('/ru_wave1/hospital-ru-def.wav')
  # spectros.append(spectroA)
  # spectroA = test_file('/ms_wave1/hospital-ms-def.wav')
  # spectros.append(spectroA)
  # spectroA = test_file('/en_wave1/hospital-en-def.wav')
  # spectros.append(spectroA)
  # spectroA = test_file('/id_wave1/hospital-id-def.wav')
  # spectros.append(spectroA)
  # spectroA = test_file('/fr_wave1/hospital-fr-def.wav')
  # spectros.append(spectroA)
  #spectroA = test_file('/uk_wave1/hospital-uk-def.wav')
  #spectros.append(spectroA)

  #graph_9(spectros)

