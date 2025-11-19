#Importation des bibliothèques nécessaires

import numpy as np
import librosa
import matplotlib.pyplot as plt


#Fonction pour charger un fichier audio et en extraire les composantes fréquentielles
def fourier_from_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    N = len(y)

    if N == 0:
        raise ValueError("Le fichier audio est vide ou illisible.")
    
    #Fenêtre de Hann
    w = np.hanning(N)
    yw = y * w

    #FFT positive
    Y = np.fft.rfft(yw)
    freqs = np.fft.rfftfreq(N, d=1.0/sr)
    amp = np.abs(Y)

    #Garder uniquement les k amplitudes les plus élevées
    k = min(500, len(amp))
    indices_top = np.argpartition(amp, -k)[-k:]
    
    #Extraire les fréquences et amplitudes correspondantes
    freqs_top = freqs[indices_top]
    amp_top = amp[indices_top]

    #Trier par fréquence croissante
    order = np.argsort(freqs_top)
    freqs_top = freqs_top[order]
    amp_top = amp_top[order]

    return freqs_top, amp_top, sr




#Définition de la fonction source gaussienne
def g(y,f, amp):
    sigma = 2
    y0=0.1

    A = amp[f]
    
    return A*np.exp(-(y-y0)**2/(2*sigma**2))




