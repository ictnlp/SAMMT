import os
import json
import torch
import numpy as np
import torch.nn as nn
import clip
from os.path import join, abspath, dirname
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

split = 'train'
dic = {
    'train': 'flickr30k-images',
    'valid': 'flickr30k-images',
    'test': 'flickr30k-images',
    'test1': 'test_2017_flickr',
    'test2': 'test_2017_mscoco'
}
dic1 = {
    'train': 'train',
    'valid': 'val',
    'test': 'test_2016_flickr',
    'test1': 'test_2017_flickr',
    'test2': 'test_2017_mscoco'
}
imagepth = join('flickr30k', dic[split])
imagenamepth = join('multi30k-dataset/data/task1/image_splits',dic1[split] + '.txt')
savedir = 'data/clip'

clipmodel, preprocess = clip.load("ViT-B/32")
clipmodel.cuda().eval()
modeltrans = timm.create_model('ghostnet_100', pretrained=True, num_classes=0)
modeltrans.eval()
config = resolve_data_config({}, model=modeltrans)
transform = create_transform(**config)

def main():
    with open(imagenamepth, 'r', encoding='utf-8') as src_file:
        name_inputs = list(map(str.strip, src_file.readlines()))
    chunk_size = 659  # adjust according to your GPU graphics memory
    for chunk_id in range(len(name_inputs) // chunk_size + 1):
        begin = chunk_id * chunk_size
        end = min((chunk_id + 1) * chunk_size, len(name_inputs))
        img_features = []
        for idx in range(begin, end):
            print('{0}/{1}'.format(idx, len(name_inputs)))
            fname = join(imagepth, name_inputs[idx])
            img = Image.open(fname).convert('RGB')
            img = transform(img).unsqueeze(0)
            img_features.append(clipmodel.encode_image(img.to('cuda:0')))
        img_features = torch.stack(img_features, dim=0).cpu()
        img_features = np.array(img_features.detach().numpy())
        np.save(join(savedir, 'clip_' + split + '% 02d'% chunk_id + '.npy'), img_features)
        all_features = []
        num = len(name_inputs) // chunk_size + 1
        for i in range(num):
            path = join(savedir, 'clip_' + str(split) + '_' + ('%02d' % i) + '.npy')
        print('loading {0}/' + str(num) + '...'.format(i))
        features = np.load(path)
        all_features.extend(features.tolist())
        all_features = np.array(all_features)
        all_features = torch.from_numpy(all_features).float()
        all_features = all_features.view(all_features.size(0), all_features.size(1), all_features.size(2))
        print(all_features.shape)
        torch.save(all_features, str(split) + '_clip_.pth')
        for i in range(num):
            path = join(savedir, 'clip_' + str(split) + '_' + ('%02d' % i) + '.npy')  #
            os.remove(path)

if __name__ == '__main__':
    main()