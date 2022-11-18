# def get_mel2ph(tg_fn, ph, mel, hop_size, audio_sample_rate, min_sil_duration=0):
#     ph_list = ph.split(" ")
#     itvs = TextGrid.fromFile(tg_fn)[1]
#     itvs_ = []
#     for i in range(len(itvs)):
#         itvs_.append(itvs[i])
#     itvs.intervals = itvs_
#     itv_marks = [itv.mark for itv in itvs]
#     tg_len = len([x for x in itvs])
#     ph_len = len([x for x in ph_list])
#     assert tg_len == ph_len, (tg_len, ph_len, itv_marks, ph_list, tg_fn)
#     mel2ph = np.zeros([mel.shape[0]], int)
#     i_itv = 0
#     i_ph = 0
#     while i_itv < len(itvs):
#         itv = itvs[i_itv]
#         ph = ph_list[i_ph]
#         itv_ph = itv.mark
#         start_frame = int(itv.minTime * audio_sample_rate / hop_size + 0.5)
#         end_frame = int(itv.maxTime * audio_sample_rate / hop_size + 0.5)
#         if is_sil_phoneme(itv_ph) and not is_sil_phoneme(ph):
#             mel2ph[start_frame:end_frame] = i_ph
#             i_itv += 1
#         elif not is_sil_phoneme(itv_ph) and is_sil_phoneme(ph):
#             i_ph += 1
#         else:
#             if not ((is_sil_phoneme(itv_ph) and is_sil_phoneme(ph)) \
#                     or re.sub(r'\d+', '', itv_ph.lower()) == re.sub(r'\d+', '', ph.lower())):
#                 print(f"| WARN: {tg_fn} phs are not same: ", itv_ph, ph, itv_marks, ph_list)
#             mel2ph[start_frame:end_frame] = i_ph + 1
#             i_ph += 1
#             i_itv += 1
#     mel2ph[-1] = mel2ph[-2]
#     assert not np.any(mel2ph == 0)
#     T_t = len(ph_list)
#     dur = mel2token_to_dur(mel2ph, T_t)
#     return mel2ph.tolist(), dur.tolist()


import os
import textgrid
import numpy as np
from glob import glob


def construct_data_queue(path):
    # Init the data queue
    data_queue = []

    # Read Audio files
    for file in glob(f'{path}/video/*/*.wav'):

        # Read TextGrid file for audio file
        textgrid_file = f'{path}/label/' + file.split('/')[-1][:-4] + '.TextGrid'
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
        
        # Read the stutter tiers' boundary in TextGrid
        stutter_time_list = []
        for entry_id in range(len(tg[0])-1): 
            if tg[0][entry_id].mark != "":
                assert tg[0][entry_id].mark in ['多读', '漏读', '错读', '其它', '其他'], 'some of the labels are incorrect'
                if tg[0][entry_id].mark == '多读':
                    stutter_label = 1
                elif tg[0][entry_id].mark == '漏度':
                    stutter_label = 2
                elif tg[0][entry_id].mark == '错读':
                    stutter_label = 3
                elif tg[0][entry_id].mark in ['其它', '其他']:
                    stutter_label = 4
                stutter_time_list.append((tg[0][entry_id].minTime, tg[0][entry_id].maxTime, stutter_label))

                # if tg[0][entry_id].mark not in ['多读', '漏读', '错读', '其它', '其他']:
                #     print(textgrid_file)
                #     print((tg[0][entry_id].minTime, tg[0][entry_id].maxTime))
                #     print(tg[0][entry_id].mark)

        # Get stutter intervals from TextGrid
        stutter_intervals = []
        for interval in intervals:
            stutter_interval = []
            for stutter_item in stutter_time_list:
                min_boundary, max_boundary = interval[0], interval[1]
                if (stutter_item[0] > min_boundary and stutter_item[0] < max_boundary) or (stutter_item[1] > min_boundary and stutter_item[1] < max_boundary):
                    stutter_item_min = min_boundary if stutter_item[0] < min_boundary else stutter_item[0]
                    stutter_item_max = max_boundary if stutter_item[1] > max_boundary else stutter_item[1]
                    stutter_interval.append((stutter_item_min-min_boundary, stutter_item_max-min_boundary, stutter_item[2]))
            stutter_intervals.append(stutter_interval)

        # Construct data queue
        for idx in range(len(stutter_intervals)):
            audio_name = file.split('/')[-1][:-4] + '_SEG' + str(idx)
            input_path = './data/processed/audios/' + file.split('/')[-1][:-4] + '/' + audio_name
            data_queue.append({
                'audio_name': audio_name,
                'input_path': input_path,
                'interval': intervals[idx],
                'stutter_interval': stutter_intervals[idx]
            })
    return data_queue

def generate_stutter_labels(data_item):
    output_folder = './data/processed/stutter_labels/'+data_item['input_path'].split('/')[4]
    output_filename = data_item['input_path'].replace('audios', 'stutter_labels')
    os.makedirs(output_folder, exist_ok=True)
    np.save(output_filename, data_item['stutter_interval'])


if __name__ == '__main__':
    data_queue = construct_data_queue('./data/raw')
    for data_item in data_queue:
        generate_stutter_labels(data_item)