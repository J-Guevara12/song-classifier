import pandas as pd
import subprocess

data = pd.read_csv("./res/data/canciones_sys.csv",encoding="latin-1")

genreCount = {}

# iterate over the dataframe containing the songs
for song in data.iterrows():
    genre = song[1][1]
    link = song[1][0]
    
    # Generates a table that contains all the genres and its quantity
    genreCount[genre] = genreCount.setdefault(genre,0) + 1
    res = subprocess.call(['yt-dlp', '--extract-audio', '--audio-format', 'mp3', '-o',f'res/music/mp3/{genre}-{genreCount[genre]}',link,])
