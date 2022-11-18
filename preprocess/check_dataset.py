import os
from glob import glob

# Check if each item in metadata.csv has the corresponding file on disk
file_queue = []
with open('./data/processed/metadata.csv', 'r') as f:
    lines = f.readlines()
    for line in lines:
        file_path, transcribes = line.split('|')
        file_queue.append(file_path)
        if not os.path.exists(file_path):
            print(file_path + ' is not exist. Please check it.')

# Check if each file on disk has the corresponding item in metadata.csv
for file_path in glob('./data/processed/audios/*/*.wav'):
    if file_path not in file_queue:
        print(file_path + ' is not in \'metadata.csv\'. Please check it.')