# Chord-recognition
Identify guitar chords by machine learning.




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
