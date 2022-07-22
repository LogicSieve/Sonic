#!/usr/bin/python3

from pydub import AudioSegment
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
from os.path import exists

dic = ('afrikaans', 'af', 'albanian', 'sq', 'amharic', 'am',
       'arabic', 'ar', 'armenian', 'hy', 'azerbaijani', 'az',
       'basque', 'eu', 'belarusian', 'be', 'bengali', 'bn', 'bosnian',
       'bs', 'bulgarian', 'bg', 'catalan', 'ca',
       'cebuano', 'ceb', 'chichewa', 'ny', 'chinese (simplified)',
       'zh-cn', 'chinese (traditional)', 'zh-tw',
       'corsican', 'co', 'croatian', 'hr', 'czech', 'cs', 'danish',
       'da', 'dutch', 'nl', 'english', 'en', 'esperanto',
       'eo', 'estonian', 'et', 'filipino', 'tl', 'finnish', 'fi',
       'french', 'fr', 'frisian', 'fy', 'galician', 'gl',
       'georgian', 'ka', 'german', 'de', 'greek', 'el', 'gujarati',
       'gu', 'haitian creole', 'ht', 'hausa', 'ha',
       'hawaiian', 'haw', 'hebrew', 'he', 'hindi', 'hi', 'hmong',
       'hmn', 'hungarian', 'hu', 'icelandic', 'is', 'igbo',
       'ig', 'indonesian', 'id', 'irish', 'ga', 'italian', 'it',
       'japanese', 'ja', 'javanese', 'jw', 'kannada', 'kn',
       'kazakh', 'kk', 'khmer', 'km', 'korean', 'ko', 'kurdish (kurmanji)',
       'ku', 'kyrgyz', 'ky', 'lao', 'lo',
       'latin', 'la', 'latvian', 'lv', 'lithuanian', 'lt', 'luxembourgish',
       'lb', 'macedonian', 'mk', 'malagasy',
       'mg', 'malay', 'ms', 'malayalam', 'ml', 'maltese', 'mt', 'maori',
       'mi', 'marathi', 'mr', 'mongolian', 'mn',
       'myanmar (burmese)', 'my', 'nepali', 'ne', 'norwegian', 'no',
       'odia', 'or', 'pashto', 'ps', 'persian',
       'fa', 'polish', 'pl', 'portuguese', 'pt', 'punjabi', 'pa',
       'romanian', 'ro', 'russian', 'ru', 'samoan',
       'sm', 'scots gaelic', 'gd', 'serbian', 'sr', 'sesotho',
       'st', 'shona', 'sn', 'sindhi', 'sd', 'sinhala',
       'si', 'slovak', 'sk', 'slovenian', 'sl', 'somali', 'so',
       'spanish', 'es', 'sundanese', 'su',
       'swahili', 'sw', 'swedish', 'sv', 'tajik', 'tg', 'tamil',
       'ta', 'telugu', 'te', 'thai', 'th', 'turkish', 'tr',
       'ukrainian', 'uk', 'urdu', 'ur', 'uyghur', 'ug', 'uzbek',
       'uz', 'vietnamese', 'vi', 'welsh', 'cy', 'xhosa', 'xh',
       'yiddish', 'yi', 'yoruba', 'yo', 'zulu', 'zu')


def readinfile(filename):
    words_list = []
    common_words_file = open(filename, 'r')
    word = ''
    for wordline in common_words_file:
        #print(f'Word is {wordline}')
        words_list.append(wordline.lower().strip())

    return words_list

def word2wav(words=[], to_lang='en', outdir='.'):
    text = ''
    translator = Translator()
    for word in words:
        outfile_name = f'{outdir}/{word}-{to_lang}-def.wav'
        if exists(outfile_name):
            print(f'Out file {outfile_name} found, skip work.')
            continue
        try:
            out_info = translator.translate(word, src='en', dest=to_lang)
        except Exception as e:
            print(f'Out error {str(e)} with word {word}')
            continue
        print(f'Out {out_info}')
        speak = gTTS(text=out_info.text, lang=to_lang, slow=False)
        speak.save("captured_voice.mp3")
        # Using OS module to run the translated voice.
        wav_dude = AudioSegment.from_mp3('captured_voice.mp3')
        outfile_name = f'{outdir}/{word}-{to_lang}-def.wav'
        wav_dude.export(outfile_name, format='wav')

word_list = readinfile('combined_words_f.txt')
process_list = word_list
#print(f'process list {process_list}')
# 'a' messes up (uk)ranian, (fr)ench and others, because of multiple returns from google translate ,handle later.
# feminine/masculine in French
# catching this exception in word2wav

#word2wav(process_list, 'en', './word_data/en_wave1')
word2wav(process_list, 'fr', './word_data/fr_wave1')
word2wav(process_list, 'ja', './word_data/ja_wave1')
word2wav(process_list, 'zh-cn', './word_data/zh-cn_wave1')
#word2wav(process_list, 'ms', './word_data/ms_wave1')
#word2wav(process_list, 'id', './word_data/id_wave1')
word2wav(process_list, 'ru', './word_data/ru_wave1')
word2wav(process_list, 'uk', './word_data/uk_wave1')