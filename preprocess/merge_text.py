from g2p_en import G2p
from glob import glob
import pandas as pd

name_list_, text_list_ = [], []
for filename in sorted(glob('data/processed/*/*.xls')):
    print(filename)
    xls = pd.ExcelFile(filename)
    sheetX = xls.parse(0)
    name_list, text_list = sheetX['file_name'], sheetX['text']
    name_list_ += [name for name in name_list]
    text_list_ += [text for text in text_list]

with open('data/processed/metadata.csv', 'w') as f:
    for i in range(len(name_list_)):
        f.write(name_list_[i] + '|' + text_list_[i] + '\n')