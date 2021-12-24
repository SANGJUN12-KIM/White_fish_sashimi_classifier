import os
import glob
from PIL import Image
from tqdm import tqdm
import csv


def one_hot_encoding(fishname):
    if fishname == '광어':
        fishnum = [1, 0, 0, 0, 0, 0, 0, 0]
    if fishname == '농어':
        fishnum = [0, 1, 0, 0, 0, 0, 0, 0]
    if fishname == '도다리':
        fishnum = [0, 0, 1, 0, 0, 0, 0, 0]
    if fishname == '민어':
        fishnum = [0, 0, 0, 1, 0, 0, 0, 0]
    if fishname == '숭어':
        fishnum = [0, 0, 0, 0, 1, 0, 0, 0]
    if fishname == '우럭':
        fishnum = [0, 0, 0, 0, 0, 1, 0, 0]
    if fishname == '전어':
        fishnum = [0, 0, 0, 0, 0, 0, 1, 0]
    if fishname == '참돔':
        fishnum = [0, 0, 0, 0, 0, 0, 0, 1]

    return fishnum

# 광어:1, 농어:2, 도다리:3, 민어:4, 숭어:5, 우럭:6, 전어:7, 참돔:8
# [1, 0, 0, 0, 0, 0, 0, 0]

fishname='참돔'
files = glob.glob('./Dataset_original/'+fishname+'/*.png')
output_dir = './Dataset_proc'

if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)

for i, f in enumerate(tqdm(files)):
    img = Image.open(f)
    cw, ch = 250, 250
    w, h = img.size
    box = w // 2 - cw // 2, h // 2 - ch // 2, w // 2 + cw // 2, h // 2 + ch // 2
    cropped_img = img.crop(box)
    resized_img = cropped_img.resize((128, 128))
    #resized_img.save(output_dir + f'{i:02d}' + '.png')


    if i <40:
        train_dir = output_dir + '/train'
        if os.path.isdir(train_dir) == False:
            os.makedirs(train_dir)
        file_name= train_dir +'/'+ fishname + ' '+f'{i:02d}' + '.png'
        resized_img.save(file_name)
        with open(output_dir + '/trainLabel.csv', 'a') as f:
            wr = csv.writer(f)
            #encoding_fishname = one_hot_encoding(fishname)
            wr.writerow([os.path.basename(file_name), fishname])

    else:
        valid_dir = output_dir + '/valid'
        if os.path.isdir(valid_dir) == False:
            os.makedirs(valid_dir)
        file_name = valid_dir + '/' + fishname + ' ' + f'{i:02d}' + '.png'
        resized_img.save(file_name)
        with open(output_dir + '/validLabel.csv', 'a') as f:
            wr = csv.writer(f)
            #encoding_fishname = one_hot_encoding(fishname)
            wr.writerow([os.path.basename(file_name), fishname])

fishnum = 1

