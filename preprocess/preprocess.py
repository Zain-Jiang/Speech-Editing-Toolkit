import os
import csv
import subprocess
import textgrid
from glob import glob
import webvtt
from nltk.tokenize import sent_tokenize
import xlwt

def construct_split_queue(path):
    # Init the split queue
    split_queue = []

    # Read Audio filename
    audio_path_list, output_path_list = [], []
    for file_path in glob(f'{path}/video/Lec_Video_038_001/*.wav'):
        audio_name = file_path.split('/')[-1][:-4]

        # Read TextGrid file for audio filename
        textgrid_file = f'{path}/label/' + audio_name + '.TextGrid'
        tg = textgrid.TextGrid.fromFile(textgrid_file)
        # Read the sentence boundary in TextGrid
        stop_time_list = [0.0]
        for entry_id in range(len(tg[0])-1): 
            if (tg[0][entry_id].mark == "" or tg[0][entry_id].mark == " ") and (tg[0][entry_id+1].mark == "" or tg[0][entry_id+1].mark == " "):
                stop_time_list.append(tg[0][entry_id].maxTime)
        stop_time_list.append(tg[-1].maxTime)
        # Get intervals from TextGrid
        intervals = []
        for seg_id in range(len(stop_time_list) - 1):
            intervals.append((stop_time_list[seg_id], stop_time_list[seg_id+1]))
        # Read the "wrong" tiers in TextGrid
        wrong_time_list = []
        for entry_id in range(len(tg[0])-1): 
            if tg[0][entry_id].mark != "":
                wrong_time_list.append((tg[0][entry_id].minTime, tg[0][entry_id].maxTime))
        print(len(wrong_time_list))

        # Read transcribes file
        textgrid_file = file_path[:-4] + '.vtt'
        transcribes = process_text(textgrid_file)

        split_queue.append({
            'audio_name': audio_name,
            'input_path': file_path,
            'intervals': intervals,
            'transcribes': transcribes,
        })

    return split_queue

def split_file(split_queue, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    
    for task in split_queue:
        audio_name, input_path, intervals, transcribes = \
            task['audio_name'], task['input_path'], task['intervals'], task['transcribes']

        seg_id = 0
        output_folder = os.path.join(dest_folder, audio_name)
        os.makedirs(output_folder, exist_ok=True)
        for interval in intervals:
            start, end = interval
            output_path = f'{output_folder}/{audio_name}_SEG{seg_id}.wav'
            command_a = ' '.join([
                "ffmpeg", "-i", input_path,
                "-loglevel", "quiet", # Verbosity arguments
                "-y", # Overwrite if the file exists
                "-ss", str(start), "-to", str(end), # Cut arguments
                "-f", "wav", # Crop arguments
                "-ac", "1", # Single channel
                "-codec", "copy",
                "-ar", "44100",
                output_path
            ])
            seg_id += 1
            return_code_a = subprocess.call(command_a, shell=True)

            if not return_code_a == 0:
                print('Command failed:', command_a)

        # Write the meta.csv file 
        with open(f'{dest_folder}/{audio_name}.csv', 'w', newline='') as csvfile:
            seg_id = 0
            for text_item in transcribes:
                csvfile.write(f'{output_folder}/{audio_name}_SEG{seg_id}.wav|{text_item}\n')
                seg_id += 1
        # book = xlwt.Workbook()
        # sheet = book.add_sheet('Sheet1')
        # sheet.write(0,0,'file_name')
        # sheet.write(0,1,'text')
        # seg_id = 0
        # for text_item in transcribes:
        #     sheet.write(seg_id+1, 0, f'{output_folder}/{audio_name}_SEG{seg_id}.wav')
        #     sheet.write(seg_id+1, 1, text_item)
        #     seg_id += 1
        # book.save(f'{output_folder}/{audio_name}.xls')

# def process_text(path):
#     # Read transcribes from webvtt files
#     vtt = webvtt.read(path)
#     transcribes = (" ".join([caption.text for caption in vtt])).replace("\n", " ")
#     transcribes = sent_tokenize(transcribes)
#     return transcribes
def process_text(path):
    # Read transcribes from webvtt files
    vtt = webvtt.read(path)
    transcribes = (" ".join([caption.text for caption in vtt])).replace("\n", " ")
    transcribes = transcribes.replace("\n", " ")
    transcribes = transcribes.replace("i.e.", "i e")
    transcribes = transcribes.replace("i.e", "i e")
    transcribes = transcribes.replace("Dr.", "Dr")
    transcribes = transcribes.replace("Mr.", "Mr")
    transcribes = sent_tokenize(transcribes)
    new_transcribes = []
    for sentence in transcribes:
        sentence_wop = sentence[:-1]
        sentence_wop = sentence_wop.split('.')  #分句子中没分割掉的.
        for sub_sent in sentence_wop:   #加.
            sub_sent = sub_sent + "."
            new_transcribes.append(sub_sent)
    transcribes = new_transcribes
    return transcribes


if __name__ == '__main__':
    split_queue = construct_split_queue('./data/raw')
    split_file(split_queue, './data/processed/audios')