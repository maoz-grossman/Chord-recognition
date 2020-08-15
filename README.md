# Chord-recognition
## Identify guitar chords by machine learning.




As a final project in the "Machine Learning" course at the university,
we tried to build a program that could decode chords from songs ,in wav file format,using machine learning.
namely, given a new chord, or some piece of music, the program will be able to identify the chords being played.

Because sound files are a function that depends on time and changes at any given moment, 
the main challenge in the project is to convert sound files into data that can be processed by machine learning algorithms.
In other words, it is necessary to create digital data from analog data.

Luckily we found these great lessons of
[Valerio Velardo - The Sound of AI](https://www.youtube.com/channel/UCZPFjMe1uRSirmSpznqvJfQ) on YouTube.
Very high-level lessons, that convey the material in a thorough, comprehensive and understandable way.

For the part of converting the files to hpcp we owe our thanks to [Andres Mauricio Rondon Patiño](https://github.com/amrondonp),
and in particular to his code [pitch_class_profiling.py](https://github.com/amrondonp/Chords.py/blob/master/final_project/preprocessing/pitch_class_profiling.py)


<h4>Data set description </h4>
guitar chords recorded both in the studio and noisy environments. 
The database contains 2000 chords splitted up in 10 classes, giving up to 200 chords per chord type. 
Each sound file is one second to three seconds long.
The files are stored in raw WAV 16 bits mono 44100Hz format. 
(from https://people.montefiore.uliege.be/josmalskyj/research.php )

<h4>Techniques</h4>
<h6>Machine learning:</h6> 

1. [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)

2. [Support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine)

3. [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

4. [Decision tree](https://en.wikipedia.org/wiki/Decision_tree)


<h6>Neuron network (failed):</h6>

1. [Multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)

2. [Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)

3. [Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network)



<h4>Preprocess Techniques:</h4> 

<h5>Less successful techniques </h5>
<h6>1. Mfcc - </h6>

***In sound processing, the mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound,
based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.
Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC. 
They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum").***
(from https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)


<h5>More successful techniques</h5>
<h6>Harmonic pitch class profiles (HPCP) - </h6>

***is a group of features that a computer program extracts from an audio signal,
based on a pitch class profile—a descriptor proposed in the context of a chord recognition system. 
HPCP are an enhanced pitch distribution feature that are sequences of feature vectors that, 
to a certain extent, describe tonality, measuring the relative intensity of each of the 12 pitch classes of the equal-tempered scale within an analysis frame.
Often, the twelve pitch spelling attributes are also referred to as chroma and the HPCP features are closely related to what is called chroma features or chromagrams.***
(https://en.wikipedia.org/wiki/Harmonic_pitch_class_profiles)



<h5>results</h5>
On average the best results we got were from the knn algorithm, with an average accuracy of ~96 percent.
Second place went to the 'Adaboost' algorithm, with an average accuracy of ~95 percent.
Not far from it in third place is  'Decision tree', with ~95 percent accuracy on average.
And last but not least, the SVM algorithm with ~94 percent accuracy on average.

<h5> Test - chords recognition of a real song </h5>
We wanted to test how good our model is on existing songs as well.
As in the Github repository from which we learned about pcp preprocess, we took the first half minute of Nirvana's song "about a girl" and disassembled it to half second chunks and checked what the model says the chords of the song.
The result:
![](/Test/Screen%20Shot%202020-08-15%20at%2008.46.34.png)
You can find the test.py file in the Test folder.
