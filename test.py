import os
import json
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sns
from pathlib import Path

#label
filetype = r'.json'
file_list = [file for file in os.listdir('/home/jihun/pytorch-cifar/yolov5/example/') if file.endswith(filetype)]
for filename in file_list:
    #print(filename)
    path = Path('/home/jihun/pytorch-cifar/yolov5/example/')
    json_file = path.joinpath(filename)
    with open(json_file, encoding='utf-8') as f:
        json_data = json.load(f)
        #print(len(json_data['shapes']))
        #for j in range(0,len(json_data['shapes'])):
            #labels = json_data['shapes'][j]['label']
            #print(labels)
            #class
        df = pd.DataFrame(json_data['shapes'])
        label_counts = df['label'].value_counts()
        print(label_counts)
    y_position = 1.02
    f = plt.subplots(1, figsize=(12, 5))
    df['label'].value_counts().sort_values().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3','#C1F80A'])
    