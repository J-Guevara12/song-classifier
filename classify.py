import os
import time
import argparse
import subprocess
import pickle

import soundfile as sf
import numpy as np

parser = argparse.ArgumentParser(description="Hi")
parser.add_argument("link",type=str,help="The song's youtube link")
args = parser.parse_args()
link = args.link

START = time.time()
start = time.time()
print("Donwloading song...\n")
res = subprocess.call(['yt-dlp', '--extract-audio', '--audio-format', 'mp3', '-o',f'res/music/mp3/test',link])
print(f"Song Dowloaded in {round(time.time()-start,3)}\n")

start = time.time()
print("Converting MP3 into WAV...\n")
subprocess.call(['ffmpeg','-i','./res/music/mp3/test.mp3','./res/music/wav/test.wav'])
print(f"Song converted in {round(time.time()-start,3)}\n")

start = time.time()
print("Sampling song...\n")
x,fs = sf.read('./res/music/wav/test.wav')
#we're going to take at least 10 samples per song, starting at 30s until 165s in 15s lapses
n = 0
X = []
for i in range(30,180,15):
    # We're only going to take the first channel between 30 and 35 seconds, 60 and 90 and 95
    if(x.shape[0]>(i+5)*fs): #Checking that we're not sampling away of the song's end
        n += 1
        xTaken = x[i*fs:(i+5)*fs].copy().T[0]
        xfourier = np.fft.rfft(xTaken)
        
        X.append(abs(xfourier))

print(f"Took {n} samples of the song in {round(time.time()-start,3)}\n")

loaded_model = pickle.load(open('./res/models/randomForest.pickle','rb'))
predictions = loaded_model.predict(X)
res = {}
for pred in predictions:
    res[pred] = res.setdefault(pred,0) + 1

print("The song has been classified, results are:")
for genre, i in res.items():
    print(f"{genre}: {round(i/n*100,3)}%")

os.remove('./res/music/mp3/test.mp3')
os.remove('./res/music/wav/test.wav')

print(f"the process took: {round(time.time()-START,3)}s\n")
