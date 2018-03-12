from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer

import sys
import scipy.io.wavfile as wav
import numpy as np
from deepspeech.model import Model
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00

# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


def speech_to_text(baidu, audio_file_path):
    # converts audio to text according to baidu model

    fs, audio = wav.read(audio_file_path)

    if fs != 16000:
        raise RuntimeError('Sample rate must be 16kHz')

    audio_length = len(audio) * (1 / 16000)

    #    print('Running inference.', file=sys.stderr)
    inference_start = timer()
    speech2text = baidu.stt(audio, fs)
    inference_end = timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)
    return speech2text


def find_key_word_times(teager, t, split_points, processed_words, baidu_text, curr_word, iterator):

    # take teager values corresponing to current slice
    t_res = np.size(t) / np.max(t)

    ind0 = int(split_points[iterator] * t_res)

    ind_end = int(split_points[iterator + 1] * t_res)

    curr_teager = teager[ind0:ind_end]
    curr_t = t[ind0:ind_end]

    # find silence and speech sections inside current audio slice such that their duration must be
    # longer than minimal defined duration

    # define threshold, minimal silence duration and minimal speech duration

    t_hold = 0.009

    min_silence_dur = 0.2 * t_res
    min_speech_dur = 0.05 * t_res
    min_silence_dur = min_silence_dur.astype(np.int32)
    low_count = 0
    high_count = 0
    index_marker = 0

    silence_sections = np.array([])
    speech_sections = np.array([])

    binary_teager = curr_teager
    binary_teager[binary_teager <= t_hold] = 0
    binary_teager[binary_teager > t_hold] = 1

    plt.figure(14 + iterator)

    plt.plot(curr_t, curr_teager)
    plt.plot(curr_t, binary_teager)

    previous_val = binary_teager[0]

    for i in np.arange(np.size(binary_teager)):

        if previous_val == 1:

            if binary_teager[i] == 1:

                high_count = high_count + 1

            else:

                if high_count > min_speech_dur:

                    if index_marker != (speech_sections[-1] if np.size(speech_sections != 0) else np.nan):

                        speech_sections = np.append(speech_sections, index_marker)
                        speech_sections = np.append(speech_sections, i)
                        index_marker = i

                    else:

                        speech_sections[-1] = i
                        index_marker = i

                high_count = 0
                low_count = 1


        else:

            if binary_teager[i] == 0:

                low_count = low_count + 1

            else:

                if low_count > min_silence_dur:

                    if index_marker != (silence_sections[-1] if np.size(silence_sections != 0) else np.nan):

                        silence_sections = np.append(silence_sections, index_marker)
                        silence_sections = np.append(silence_sections, i)
                        index_marker = i

                    else:

                        silence_sections[-1] = i
                        index_marker = i

                low_count = 0
                high_count = 1

        previous_val = binary_teager[i]

    if high_count > low_count:

        if high_count > min_speech_dur:

            speech_sections = np.append(speech_sections, index_marker)
            speech_sections = np.append(speech_sections, i)

        else:

            silence_sections[-1] = i
    else:

        if low_count < min_silence_dur:

            speech_sections[-1] = i
        else:

            silence_sections = np.append(silence_sections, index_marker)
            silence_sections = np.append(silence_sections, i)

    # preprocess speeech sections of this audio slice and calculate when they beggin and end relative to
    # beggining of whole audio file

    # also calculate duration of each speech section relative to duration of sum of all speech sections
    # in current audio slice


    speech_sections = speech_sections / t_res + np.min(curr_t)

    speech_proportions = np.append(np.array([0]), (speech_sections[1::2] - speech_sections[::2]))

    speech_proportions = speech_proportions / np.sum(speech_proportions)

    for i in np.arange(1, np.size(speech_proportions)):
        speech_proportions[i] = speech_proportions[i - 1] + speech_proportions[i]

    word_proportions = np.array([0])
    word_lengths = np.array([])
    word_to_section = np.array([])

    # preprocessing words

    for words in baidu_text.split():
        word_proportions = np.append(word_proportions, len(words))
        word_lengths = np.append(word_lengths, len(words))

    word_proportions = word_proportions / float(np.sum(word_proportions))

    for i in np.arange(1, np.size(word_proportions)):
        word_proportions[i] = word_proportions[i - 1] + word_proportions[i]

    # appending words to speech intervals
    print("hah",speech_proportions)
    print(word_proportions)

    for i in np.arange(np.size(word_proportions) - 1):

        up_lim = word_proportions[i + 1]
        low_lim = word_proportions[i]

        k = 1
        max_overlap = 0
        max_index = 1
        while up_lim > speech_proportions[k - 1] and k < np.size(speech_proportions):

            if low_lim <= speech_proportions[k]:

                if low_lim >= speech_proportions[k - 1] and up_lim <= speech_proportions[k]:

                    if up_lim - low_lim > max_overlap:
                        max_overlap = up_lim - low_lim
                        max_index = k

                if low_lim <= speech_proportions[k - 1] and up_lim <= speech_proportions[k]:

                    if up_lim - speech_proportions[k - 1] > max_overlap:
                        max_overlap = up_lim - speech_proportions[k - 1]
                        max_index = k

                if low_lim >= speech_proportions[k - 1] and up_lim >= speech_proportions[k]:

                    if speech_proportions[k] - low_lim > max_overlap:
                        max_overlap = speech_proportions[k] - low_lim
                        max_index = k

                if low_lim <= speech_proportions[k - 1] and up_lim >= speech_proportions[k]:

                    if speech_proportions[k] - speech_proportions[k - 1] > max_overlap:
                        max_overlap = speech_proportions[k] - speech_proportions[k - 1]
                        max_index = k

            k = k + 1

        word_to_section = np.append(word_to_section, max_index)

    print(word_to_section)

    # determine start and end times of key word

    key_word_index = len(processed_words.split())

    key_word_section = int(word_to_section[key_word_index])
    key_word_section_dur = speech_sections[key_word_section] - speech_sections[key_word_section - 1]

    section_words = np.where(word_to_section == key_word_section)

    word_start_time = speech_sections[2 * int(key_word_section - 1)] + key_word_section_dur / np.sum(
        word_lengths[section_words]) * np.sum(word_lengths[np.min(section_words): key_word_index - 1])
    word_end_time = word_start_time + word_lengths[key_word_index] / np.sum(
        np.sum(word_lengths[section_words])) * key_word_section_dur

    key_word_time_coordinates = (word_start_time - 0.1, word_end_time + 0.1)

    return key_word_time_coordinates


def find_slice_times(file_path):

    # this function determines in which moments to cut audio signal so no words are sliced

    fs, data = wav.read(file_path)
    times = np.arange(len(data)) / float(fs)

    plt.figure(0)
    plt.title('raw data')
    plt.plot(times, data)

    # find spectrogram of a signal
    f, t, Sxx = signal.spectrogram(data, fs, nperseg=2000, noverlap=1900, window='hann')

    # calculate teager energy of the signal
    teager = np.zeros(Sxx[1, :].size)
    this_data = 0
    for i in np.arange(0, Sxx[1, :].size - 1):  # iterates through time
        for j in np.arange(0, Sxx[:, 1].size - 1):  # iterastes through freq
            this_data = (f[j] ** 1.5) * (Sxx[j, i]) + this_data  # 1.5
        teager[i] = this_data
        this_data = 0

    teager = teager / np.max(teager)

    # further process teager energy signal by applying moving average
    N = int(t.size / max(t) * 0.26)  # 0.25
    smoothed = np.convolve(teager, np.ones((N,)) / N, mode='valid')
    t1 = t[0:np.size(smoothed)] + N / 2 * max(t) / np.size(t)
    t1 = t1 + N / 4 * max(t1) / np.size(t1)

    plt.figure(4)
    plt.plot(t, teager)
    plt.title('teager energy')
    plt.plot(t1, smoothed)

    teager_bckup = teager
    t_bckup = t

    teager = smoothed
    t = t1

    # find split points

    jump = 0.5  # 0.5
    band = 2  # 2.1
    time_res = int(t.size / max(t))
    jump_res = int(time_res * jump)
    band_res = int(time_res * band)
    t_hold = 0.0075

    t_high = 0

    split_points = np.array([0])
    test_slice = np.array([])

    continuee = 1
    split_search = 0

    base_path = '/home/daki/project_stuff/audio_split/split_no'
    file_iterator = 0

    while continuee == 1:
        t_high = t_high + jump_res
        if t_high < np.size(teager):
            split_search = 1
        else:
            split_search = 0
            continuee = 0

        while split_search == 1 and continuee == 1:
            if t_high + band_res < np.size(teager):
                test_slice = teager[t_high:(t_high + band_res)]
            else:
                test_slice = teager[t_high:np.size(teager)]
                continuee = 0
            min_teager = np.min(test_slice)
            if min_teager < t_hold:
                t_high = t_high + np.argmin(test_slice) - int(0.02*time_res)
                split_points = np.append(split_points, t[t_high])
                split_search = 0
            else:
                t_high = t_high + band_res
                split_search = 1

    dots = np.full(np.size(split_points), 0.01)

    plt.figure(5)
    plt.plot(t_bckup, teager_bckup)
    plt.scatter(split_points, dots, color='red')

    split_points_float = split_points * fs
    split_index = (split_points_float).astype(np.int32)
    split_index = np.append(split_index, np.size(data) - 1)

    return_touple = (fs, data, split_index, teager, t, np.append(split_points, np.max(times)))

    return return_touple

# MAIN


# paths to model  files

model_path = '/home/daki/project_stuff/models/output_graph.pb'
alphabet_path = '/home/daki/project_stuff/models/alphabet.txt'
lm_path = '/home/daki/project_stuff/models/lm.binary'
trie_path = '/home/daki/project_stuff/models/trie'

# paths to audio file we want to process

audio_path_long = '/home/daki/project_stuff/recorded_audio/dalibor2.wav'  # long_video_1_no_bn

# loading model

print('Loading model from file %s' % (model_path), file=sys.stderr)
model_load_start = timer()
ds = Model(model_path, N_FEATURES, N_CONTEXT, alphabet_path, BEAM_WIDTH)
model_load_end = timer() - model_load_start
print('Loaded model in %0.3fs.' % (model_load_end), file=sys.stderr)

if lm_path and trie_path:
    print('Loading language model from files %s %s' % (lm_path, trie_path), file=sys.stderr)
    lm_load_start = timer()
    ds.enableDecoderWithLM(alphabet_path, lm_path, trie_path, LM_WEIGHT,
                               WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
    lm_load_end = timer() - lm_load_start
    print('Loaded language model in %0.3fs.' % (lm_load_end), file=sys.stderr)

# end loading model

# USER CODE


key_words = ['age', 'wisdom', 'dog', 'expensive', 'cheap', 'cat', 'light'] # key words
base_path = '/home/daki/project_stuff/audio_split/split_no' # path to file where to save audio slices
long_audio_data = find_slice_times(audio_path_long) # find slice points
fs = long_audio_data[0]
data = long_audio_data[1]
split_index = long_audio_data[2]
teager = long_audio_data[3]
t = long_audio_data[4]
split_points = long_audio_data[5]

file_iterator = 0
processed_words = ''

# slice long audio and process slices

for file_iterator in np.arange(np.size(split_index) - 1):
    current_slice_path = base_path + str(file_iterator)
    curr_slice_data = data[split_index[file_iterator]:split_index[file_iterator + 1]]
    wav.write(current_slice_path, fs, curr_slice_data)
    baidu_text = speech_to_text(ds, current_slice_path)
    print("result for file no ", file_iterator, ": ", baidu_text)

    # look for key words
    for baidu_word in baidu_text.split():
        for key_word in key_words:
            if baidu_word == key_word:
                word_times = find_key_word_times(teager, t, split_points, processed_words, baidu_text, baidu_word, file_iterator)
                print("key word detected: ", key_word)
                print("start time: ", word_times[0], " s")
                print("end time: ", word_times[1], " s")

        processed_words = processed_words + baidu_word + ' '

    processed_words = ''

plt.show()