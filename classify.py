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
xTaken = x[30*fs:(30+5)*fs].copy().T[0]
xfourier = np.fft.rfft(xTaken)
n = 1
print(f"Took {n} samples of the song in {round(time.time()-start,3)}\n")

loaded_model = pickle.load(open('./res/models/randomForest.pickle','rb'))
print(loaded_model.predict([abs(xfourier)]))

os.remove('./res/music/mp3/test.mp3')
os.remove('./res/music/wav/test.wav')

print(f"the process took: {round(time.time()-START,3)}\n")
