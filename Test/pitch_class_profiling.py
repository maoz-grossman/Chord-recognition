import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
from math import log2

class PitchClassProfiler():
    def __init__(self, file_name):
        self.file_name = file_name
        self.read = False

    def _read_file(self):
        self._frecuency, self._samples = wavfile.read(self.file_name)
        self.read = True

    def frecuency(self):
        if not self.read:
            self._read_file()
        return self._frecuency

    def samples(self):
        if not self.read:
            self._read_file()
        return self._samples

    def fourier(self):
        return fft(self.samples())

    def plot_signal(self):
        plt.plot(self.samples())
        plt.show()

    def plot_fourier(self):
        plt.plot(self.fourier())
        plt.show()

    def pcp(self, X):
        #The algorithm here is implemented using
        #the names of the math formula as shown in the paper
        fs = self.frecuency()

        #fref = [16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87]
        #fref = [130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94]
        fref = 130.81

        N = len(X)
        #assert(N % 2 == 0)

        def M(l, p):
            if l == 0:
                return -1
            return round(12 * log2( (fs * l)/(N * fref )  ) ) % 12

        pcp = [0 for p in range(12)]
        
        #print("Computing pcp...")
        for p in range(12):
            for l in range(N//2):
                if p == M(l, p):
                    pcp[p] += abs(X[l])**2
        
        #Normalize pcp
        pcp_norm = [0 for p in range(12)]
        for p in range(12):
            pcp_norm[p] = (pcp[p] / sum(pcp))
        #print("finished pcp")
        #pcp_norm.append(0)
        #print(type(pcp_norm))
        return list(pcp_norm)

    def get_profile(self):
        X = self.fourier()
        return self.pcp(X)
        
    def plot_profile(self):
        objects = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
        y_pos = np.arange(len(objects))
        performance = self.get_profile()
        
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Energy')
        plt.title('PCP results')
        
        plt.show()

class LongFileProfiler(PitchClassProfiler):
    def __init__(self, file_name):
        super().__init__(file_name)
        self.current_pointer = 0
        self.window = self.frecuency() // 2
        print(self.window)

    def get_profile(self):
        profiles_list = []
        samples_count = len( self.samples() )

        while self.current_pointer < samples_count:
            rigth_bound =  self.current_pointer + self.window
            
            if rigth_bound >= samples_count:
                rigth_bound = samples_count - 1

            window_samples = self.samples()[self.current_pointer: rigth_bound]
            X = fft(window_samples)
            profiles_list.append( self.pcp(X) )

            self.current_pointer += self.window
        return profiles_list
