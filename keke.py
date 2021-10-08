import os
import numpy as np
import json
import torch
import pandas as pd
import seaborn as sn
import matplotlib
import matplotlib.pyplot  as plt
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from utils.general import xywh2xyxy

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

def convert(size,box,idx):
    dw = 1./size[0]
    dh = 1./size[1]
    
    x = (max(box[idx][:][0]) + min(box[idx][:][0]))/2.0
    y = (max(box[idx][:][1]) + min(box[idx][:][1]))/2.0
    width = max(box[idx][:][0]) - min(box[idx][:][0])
    height = max(box[idx][:][1]) - min(box[idx][:][1])
    x = x * dw
    w = width * dw
    y = y * dh
    h = height * dh
    return (x,y,w,h)

def plot_labels(labels, names=(), save_dir=Path('')):
    # plot dataset labels
    print('Plotting labels... ')
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = cnt #int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])
    x = x.apply(pd.to_numeric)

    # seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(os.path.join(save_dir, filename + '_labels_correlogram.jpg'), dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use('svg')  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    # [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # update colors bug #3195
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(os.path.join(save_dir, filename + '_labels.jpg'), dpi=200)
    matplotlib.use('Agg')
    plt.close()

if __name__ == "__main__":
    filetype = r'.json'
    file_list = [file for file in os.listdir('/home/jihun/pytorch-cifar/yolov5/example/') if file.endswith(filetype)]
    for filename in file_list:
        filename = filename.split('.')[0]
        print(filename)
        path = Path('/home/jihun/pytorch-cifar/yolov5/example/')
        json_file = path.joinpath(filename+'.json')
        img_file = path.joinpath(filename+'.jpg')
        with open(json_file, encoding='utf-8') as f:
            json_data = json.load(f)
            df = pd.DataFrame(json_data['shapes'])
            im=Image.open(img_file)
            w= int(im.size[0]) 
            h= int(im.size[1])
            box = df['points']
            box = box.to_numpy()
            classes = []
            labels = []
            cnt = 0
            for i in range(0,len(json_data['shapes'])-1):
                if json_data['shapes'][i]['label'] not in classes:
                    classes.append(json_data['shapes'][i]['label'])
                    cnt += 1
                k= np.hstack([np.array(df['label'][i]),np.array(convert((w,h),box,i))])
                labels.append(k)

            labels = np.array(labels)
            labels = np.where((labels=='paper') | (labels=='c_1'),0,labels)
            labels = np.where((labels=='paperpack') | (labels=='c_2'),1,labels)
            labels = np.where((labels=='can') | (labels=='c_3'),2,labels)
            labels = np.where((labels=='glass') | (labels=='c_4'),3,labels)
            labels = np.where((labels=='pet') | (labels=='c_5'),4,labels)
            labels = np.where((labels=='plastic') | (labels=='c_6'),5,labels)
            labels = np.where((labels=='vinyl') | (labels=='c_7'),6,labels)
            labels = labels.astype('f8')
            #print(labels)

            plot_labels(labels, names=(), save_dir= '/home/jihun/pytorch-cifar/yolov5/output' )