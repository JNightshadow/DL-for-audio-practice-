import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
file = "blues.00000.wav" #30 seconds
#load audio file
#waveform
signal, sr =librosa.load(file,sr=22050) #sample rate * duration T of the sound. 22050 * 30 s
#signal is a 1D array of size sr * T i.e. more than 600,000 values and each value has amplitude of the waveform
#visualize this waveform using librosa.display
# librosa.display.waveshow(signal,sr=sr)
# plt.xlabel("Time")
# plt.xlabel("Amplitude")
# plt.show()

#fft 
fft = np.fft.fft(signal)

magnitude = np.abs(fft) # getting the magnitude of values - performing absolute values on complex values - contribution of each frequency bin
frequency = np.linspace(0,sr,len(magnitude)) #gives us number of evenly spaced numbers in an interval


left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]
#plotting this
# plt.plot(left_frequency,left_magnitude)
# plt.xlabel("Frequency")
# plt.xlabel("Magnitude")
# plt.show()

#the plots are symmetrical
#problem: it is static - does not show how frequency changes with time 
#solution: short time fourier transform (STFT)
n_fft = 2048 #expressed in number of samples - the window that we are considering when performing a single FFT
hop_length = 512 #the amount we are shifting each Fourier Transform to the right (the interval)
stft = librosa.core.stft(signal,hop_length=hop_length,n_fft=n_fft)

spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)

# librosa.display.specshow(log_spectrogram,sr=sr,hop_length=hop_length)
# plt.xlabel("Time")
# plt.xlabel("Frequency")
# plt.colorbar()
# plt.show()

#MFFCs
MFFCs = librosa.feature.mfcc(y=signal,n_fft=n_fft,hop_length=hop_length,n_mfcc=13) #13 because 13 is a good number of coefficients to represent audio
#same arguments as stft because is performing stft
#plot the mffcs
librosa.display.specshow(MFFCs,sr=sr,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC Coefficients")
plt.colorbar()
plt.show()