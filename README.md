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
<p>
In each algorithm we divided the data set into 75% training and 25% testing.<br><br>
 
In the Knn algorithms we ran several versions of a neighbor, three, five and seven  neighbors.<br>
We ran the algorithm several runs of 100 iterations, and each time we changed the data of the test and train,
And we checked which number of neighbors gives the best result.<br>
The results were not too different, all variations gave results around 95 ~ 96 percent accuracy<br>
In the first place (always) was when the number of neighbors was three,<br> 
in the second place sometimes when there were five neighbors and sometimes a neighbor,<br>
and in the last place by a (very) small gap when there were seven neighbors.<br>
results:<br>
<img src="https://github.com/maoz-grossman/Chord-recognition/blob/chords_recognition_only_ML/images/Errors%20comparsion1.JPG?raw=true" ><br>
You can find the KNN_AVG.py file in the Test folder.<br><br>
We also ran the SVM algorithm in several versions - one run of a linear function, and the other of rbf- radial basis function.<br>
Here there were larger gaps between the algorithms, out of a run of 1000 times<br>
it turns out that rbs is about three percent more accurate than linear.<br>
<img src="https://github.com/maoz-grossman/Chord-recognition/blob/chords_recognition_only_ML/images/Errors%20comparsion2.JPG?raw=true" >
<br> You can find the SVM_AVG.py file in the Test folder.<br>
The rest of the algorithm (decision trees and adaboost) we ran only once.<br>
<h6> 
Comparison of all algorithms:
 </h6> 
We also compared all the different algorithms to see which algorithm is the most accurate.<br>
We ran all the algorithms about a hundred times, when in knn we chose the number of neighbors to be 3 and in svm the base function to be rbf.
<br> 
In first place with a high average of about 97 percent- radial basis function  SVM.<br>
In second place with an average not far from that of about 96 percent - KNN, with 3 neighbors <br>
In the third and most respectable place - decision tree , with an average of between 94 and 95 percent accuracy.<br>
And last but not least - Adaboost with an average of 93 to 94 percent accuracy.<br>
<img src="https://github.com/maoz-grossman/Chord-recognition/blob/chords_recognition_only_ML/images/Errors%20comparsion3.JPG?raw=true" >
You can find the bestClass.py file in the Test folder.<br>
</p>

<p>
<h5> Test - chords recognition of a real song </h5>
We wanted to test how good our model is on existing songs as well.<br>
As in the Github repository from which we learned about pcp preprocess, we took the first half minute of Nirvana's song "about a girl" and disassembled it to half second chunks and checked what the model says the chords of the song.<br>
The result:<br>
<img src="https://github.com/maoz-grossman/Chord-recognition/blob/chords_recognition_only_ML/images/Song%20comparsion.JPG?raw=true" >
You can find the test.py file in the Test folder. <br> 
Because of the large gaps between rbf and linear, we decided to see which is more accurate between the two.<br>
To our surprise the linear was more correct than the radial, apparently because the cutting of the files was not accurate enough,
there were sections of a half-second that contained two types of chords, and this is probably what caused the result to be different between the different variations.<br>
If we had the ability to break a song into sections according to its chords we believe we would have reached 100 percent accuracy.<br>
</p>
