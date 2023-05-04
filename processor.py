import os
import time
import subprocess
import soundfile as sf
import numpy as np
import csv

def convertToWAV():
    directory = os.scandir('./res/music/mp3')
    for file in directory:
        subprocess.call(['ffmpeg','-i',file.path,'./res/music/wav/'+file.name[:-4]+'.wav'])

convertToWAV()
