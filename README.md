# Chord-recognition
##Identify guitar chords by machine learning.




As a final project in the "Machine Learning" course at the university,
we tried to build a program that could decode chords from songs ,in wav file forma, using machine learning.
namely, given a new chord, or some piece of music, the program will be able to identify the chords being played.

Because sound files are a function that depends on time and changes at any given moment, 
the main challenge in the project is to convert sound files into data that can be processed by machine learning algorithms.
In other words, it is necessary to create digital data from analog data.

Luckily we found these great lessons of
[Valerio Velardo - The Sound of AI](https://www.youtube.com/channel/UCZPFjMe1uRSirmSpznqvJfQ) on YouTube.
Very high-level lessons, that convey the material in a thorough, comprehensive and understandable way.

For the part of converting the files to hpcp we owe our thanks to [Andres Mauricio Rondon Patiño](https://github.com/amrondonp),
and in particular to his code [pitch_class_profiling.py](https://github.com/amrondonp/Chords.py/blob/master/final_project/preprocessing/pitch_class_profiling.py)


<h6>Data set description </h6>
guitar chords recorded both in the studio and noisy environments. 
The database contains 2000 chords splitted up in 10 classes, giving up to 200 chords per chord type. 
The files are stored in raw WAV 16 bits mono 44100Hz format. 
(from [](https://people.montefiore.uliege.be/josmalskyj/research.php))

<h6>Techniques</h6>
Machine learning: 
1.[AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)
2.[Support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine)
3.[k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
4.[Decision tree](https://en.wikipedia.org/wiki/Decision_tree)

Neuron network (failed):
1.[Multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
2.[Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
3.[Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network)



