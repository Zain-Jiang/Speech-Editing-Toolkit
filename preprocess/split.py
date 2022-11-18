import os
import subprocess
import textgrid
from glob import glob


def construct_split_queue(path):
    # Init the split queue
    split_queue = []

    # Read Audio files
    audio_path_list, output_path_list = [], []
    for file in glob(f'{path}/video/Lec_Video_057_001/*'):
        if file.endswith('.wav'):
            audio_name = file.split('/')[-1][:-4]
            # Read TextGrid file for audio file
            textgrid_file = f'{path}/label/' + audio_name + '.TextGrid'
            tg = textgrid.TextGrid.fromFile(textgrid_file)

            # Read the "wrong" tiers in TextGrid
            stop_time_list = [0.0]
            for entry_id in range(len(tg[0])-1): 
                if (tg[0][entry_id].mark == "" or tg[0][entry_id].mark == " ") and (tg[0][entry_id+1].mark == "" or tg[0][entry_id+1].mark == " "):
                    stop_time_list.append(tg[0][entry_id].maxTime)
            stop_time_list.append(tg[-1].maxTime)

            # print(stop_time_list)

            # Get intervals from TextGrid
            intervals = []
            for seg_id in range(len(stop_time_list) - 1):
                intervals.append((stop_time_list[seg_id], stop_time_list[seg_id+1]))

            split_queue.append({
                'audio_name': audio_name,
                'input_path': file,
                'intervals': intervals,
            })

            # print(intervals)

    return split_queue

def split_file(split_queue):
    os.makedirs('./data/processed/audios', exist_ok=True)
    dest_folder = './data/processed/audios'
    
    for task in split_queue:
        audio_name, input_path, intervals = task['audio_name'], task['input_path'], task['intervals']

        seg_id = 0
        for interval in intervals:
            start, end = interval
            output_path = f'{dest_folder}/{audio_name}_SEG{seg_id}.wav'
            command_a = ' '.join([
                "ffmpeg", "-i", input_path,
                "-loglevel", "quiet", # Verbosity arguments
                "-y", # Overwrite if the file exists
                "-ss", str(start), "-to", str(end), # Cut arguments
                "-f", "wav", # Crop arguments
                "-ac", "1", # Single channel
                "-ar", "44100",
                output_path
            ])
            seg_id += 1
            return_code_a = subprocess.call(command_a, shell=True)

            if not return_code_a == 0:
                print('Command failed:', command_a)


if __name__ == '__main__':
    split_queue = construct_split_queue('./data/raw')
    split_file(split_queue)