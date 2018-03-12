# nabu_speech

Documentation:

This program uses baidu deep speech library for speech to text conversion.

I-->Baidu deep speech library info:

this library is used for speech to text conversion and has fallowing limitations:

-takes .wav file as input
-file must be 1 channel, with sampling rate of 16kHz
-file must be shorter than 5s

II-->Baidu deep speech instalation:

-python 2.7 is required
-can work only on LInux or MAC

installation for lunux:

-pip install deepspeech
-then download pretrained model for american english: "wget -O - https://github.com/mozilla/DeepSpeech/releases/download/v0.1.1/deepspeech-0.1.1-models.tar.gz" and unpack

III-->Algorithm description:

Algorithm is designed to enable using deep speech with long audio files. THis is done by slicing long audio file into shorter files and then use deep speech library on those short files one by one. SLicing points are choosen so no word is sliced in half, slicing should occur only on silent parts of the signal. Audio slices are then processed one by one and are tested if they contain key word. If key word is detected in slice, start and end time of key word are calculated and returned(start of long file is taken as time = 0s, start and end of key word are returned relative to this starting moment)

IV-->Algorithm structure:

0-load deep speech model
1-load audio file(long audio file)
2-calculate teager energy of speech signal from long audio signal. Use moving average to get anvelope of teager energy. By using this anvelope caluclate points at which audio file can be sliced so no words are split during slicing.
3- Use points that are result of previous step to slice long audio file int short audio slices and process them.
4- Processing is done in the fallowing way:
-We take first slice, save it as wav file,process it, and then search for key words in resulting text. IF key word is found then we use function to determine start and end time of this key word. IF key word is not found then we take next slice and repeat algorithm.


V-->Algorithm user manual:

1-to use algorithm first one must install deep speech library
2-for one to use algorithm on his own computer few paths must be customized
2.1-path to model files : by downloading and unpacking pretrained model(described in part II) 4 files are obtained. In code in section that starts from line 367, paths to 4 model files must be updated.
2.2-path to audio file we want to process : this path also must be updated by assignment to variable "audio_path_long", line 376
2.3-because long audio file is sliced into short audio files, those audio slices are saved in directory afterwards. Variable "base_path" on line 400 should contain path to directory you want to save audio slices into. This variable should have fallowing format:
base_path = '[path to directory you want to save slices into]/split_no'

After updating those paths you should be able to use algorithm.
