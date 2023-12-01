# Bridging the Gap between Synthetic and Authentic Images for Multimodal Machine Translation
This repository contains code for EMNLP'23 submission "Bridging the Gap between Synthetic and Authentic Images for Multimodal Machine Translation".

## Get started
```
Text-to-image Generation Environment:
conda create -n stable python==3.8
pip install torch==2.0.1
pip install Pillow==9.5.0
pip install transformers==4.27.4
pip install diffusers==0.16.1
pip install scipy==1.10.1
pip install accelerate==0.18.0

Training environment:
conda create -n sammt python==3.6.7
pip install -r requirements.txt
pip install --editable ./
```
## Data
Multi30K texts and images can be downloaded [here](https://github.com/multi30k/dataset) and [here](https://github.com/BryanPlummer/flickr30k_entities). We get Multi30K text data from [fairseq_mmt](https://github.com/zhulifengsheng/fairseq_mmt).
```
cd fairseq_sammt
git clone https://github.com/multi30k/dataset.git
git clone https://github.com/BryanPlummer/flickr30k_entities.git
# Organize the downloaded dataset
flickr30k
├─ flickr30k-images
├─ test_2017_flickr
└─ test_2017_mscoco
multi30k-dataset
└─ data
    └─ task1
        ├─ tok
        └─ image_splits
```
## Text-to-image Generation
```
conda activate stable
python train_stable_diffusion_step50.py train
```
* args choices=['train','valid','test', 'test1', 'test2']
## Extract Image Feature
```
conda activate sammt
python image_process.py train synth
```
arguments:
* $1: choices=['train','valid','test', 'test1', 'test2']
* $2: choices=['synth','authe']
## Train and Test
### 1. Preprocess
```
conda activate sammt
bash preprocess.sh
```
### 2. Train
```
bash train_mmt.sh
```
### 3. Test
```
# bash translate_mmt.sh $1 $2
bash translate_mmt.sh clip test
```
script parameters:
* $1: choices=['clip']
* $2: choices=['test', 'test1', 'test2']

## Acknowledgements
This project is built on several open-source repositories/codebases, including:
* [fairseq](https://github.com/facebookresearch/fairseq)
* [fairseq_mmt](https://github.com/zhulifengsheng/fairseq_mmt)
* [Revisit-MMT](https://github.com/LividWo/Revisit-MMT)