import os
import re
from glob import glob
import webvtt
from g2p_en import G2p
from nltk.tokenize import sent_tokenize


def process_text(path):
    # Init the g2p module
    g2p = G2p()

    for file in glob(f'{path}/video/Lec_Video_033_000/*'):
        if file.endswith('.vtt'):

            # Read transcribes from webvtt files
            vtt = webvtt.read(file)
            transcribes = ""
            for caption in vtt:
                transcribes += caption.text + " "
            transcribes = transcribes.replace("\n", " ")
            transcribes = transcribes.replace("i.e.", "i e")
            transcribes = transcribes.replace("i.e", "i e")
            transcribes = transcribes.replace("Dr.", "Dr")
            transcribes = transcribes.replace("Mr.", "Mr")
            transcribes = transcribes.replace("St.", "St")
            transcribes = sent_tokenize(transcribes)    #句子分割
            new_transcribes = []
            for sentence in transcribes:
                sentence_wop = sentence[:-1]
                sentence_wop = sentence_wop.split('.')  #分句子中没分割掉的.
                for sub_sent in sentence_wop:   #加.
                    sub_sent = sub_sent + "."
                    new_transcribes.append(sub_sent)
            
            transcribes = new_transcribes

            print(len(transcribes))
            print(transcribes[0:10])


            import sys
            sys.exit(0)
            # Text normalization
            transcribes.remove(" ")
            _transcribes = []
            for transcribe in transcribes:
                if transcribe[0] == " ":
                    transcribe = transcribe[1:]
                _transcribes.append(transcribe)
            transcribes = _transcribes

            phonemes = []
            for transcribe in transcribes:
                g2p_out = g2p(transcribe)
                phonemes.append(g2p_out)


            print(file)

            print(len(transcribes))
            print(transcribes[28:32])
            import sys
            sys.exit(0)
    
    # transcribes = ""
    # for text_line in text_lines:
    #     transcribes += text_line
    return split_queue


if __name__ == '__main__':
    split_queue = process_text('./data/raw')