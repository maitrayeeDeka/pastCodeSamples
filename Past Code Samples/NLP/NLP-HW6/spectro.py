
import wave
import sys
import numpy as np
import matplotlib.pyplot as plt
from array import array
import math

######################## READ ME ########################
# In order to run this program, please run python spectro.py sa1.wave (for example)
# The second argument should be the file name of the .wav file we are trying to generate the spectrogram for
########################################################


def createDataWindows(numSamples, windowSize, stepSize, reader):
    dataWindows = []
    currentPos = 0
    while currentPos < numSamples:
        reader.setpos(currentPos)
        data = list(array('h', reader.readframes(windowSize)))
        dataWindows.append(data)
        currentPos += stepSize
    return(dataWindows)


def applyHanning(dataWindows):
    for window in dataWindows:
        hanningCoefs = np.hanning(len(window))
        for i in range(0, len(window)):
            window[i] = hanningCoefs[i]*window[i]
    return(dataWindows)


def minMaxScale(allWindowMagnitudes, maxMag, minMag):
    for window in allWindowMagnitudes:
        for i in range(0, len(window)):
            window[i] = (window[i] - minMag) / (maxMag - minMag)
    return(allWindowMagnitudes)


def fourierTransform(dataWindows):
    allWindowMagnitudes = []
    allWindowFreq = []
    hanningDataWindows = applyHanning(dataWindows)
    for window in hanningDataWindows:
        windowMags = []
        windowFreq = np.fft.rfftfreq(len(window),0.025)
        transformed = np.fft.rfft(window)
        for val in transformed:
            squareMag = abs(val)
            logScale = 10*math.log10(squareMag)
            windowMags.append(logScale)
        allWindowMagnitudes.append(windowMags)
        allWindowFreq.append(windowFreq)
    return(allWindowMagnitudes, allWindowFreq)


def readWave(file, windowSize, stepSize):
    obj = wave.open(file,'r')
    numSamples = obj.getnframes()

    dataWindows = createDataWindows(numSamples, windowSize, stepSize, obj)

    allWindowMagnitudes, allWindowFreq = fourierTransform(dataWindows)

    magnitudeNP = np.array(allWindowMagnitudes)
    maxMag = max(magnitudeNP.max())
    minMag = min(magnitudeNP.min())

    timeLabels = list(range(0, len(allWindowMagnitudes)))
    normWindowMags = minMaxScale(allWindowMagnitudes, maxMag, minMag)

    for times, freq, intensity in zip(timeLabels, allWindowFreq, normWindowMags):
        plt.scatter([times] * len(freq), freq, s = 0.04, c = intensity, cmap='gray_r', vmin = 0, vmax = 1)

    plt.ylabel('Frequency (hz)')
    plt.xlabel('Time (ms)')
    plt.colorbar()
    plt.show()
    obj.close()


def main():
    wavFile = sys.argv[1]
    windowSize = 400
    stepSize = 160
    readWave(wavFile, windowSize, stepSize)


if __name__ == '__main__':
   main()
