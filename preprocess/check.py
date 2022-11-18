# data check
import os
import random
import shutil
import pandas as pd
from glob import glob

data_queue = []
for item in sorted(glob('data/raw/filter/*.xls')):
    xls = pd.ExcelFile(item)
    sheetX = xls.parse(0)
    for file_id in range(len(sheetX['Unnamed: 0'])):
        data_flag = sheetX['Unnamed: 0'][file_id]
        item_name = os.path.basename(item)[:-4]
        input_file = item.replace('raw', 'processed').replace('filter', item_name)[:-4]+'_SEG'+str(file_id)+'.wav'
        output_file = 'data/check/' + item_name + '_SEG'+str(file_id)+'_'+str(data_flag)+'.wav'
        data_dict = {'input_file': input_file, 'output_file': output_file}
        data_queue.append(data_dict)

data_queue = random.sample(data_queue, int(len(data_queue)*0.03))

for data_item in data_queue:
    shutil.copyfile(data_item['input_file'], data_item['output_file'])
