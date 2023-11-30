#!/usr/bin/bash
set -e

_image_feat=$1
who=$2

# set device
gpu=0

model_root_dir=checkpoints

# set task
task=multi30k-en2de
# task=msctd

image_feat=$_image_feat

length_penalty=0.8

# set tag
model_dir_tag=$image_feat/11290150_0.5klmulti_32_sd_ran0+0.1beforeimgotlossvishal_release

if [ $task == "multi30k-en2de" ]; then
	tgt_lang=de
	data_dir=multi30k.en-de
elif [ $task == 'multi30k-en2fr' ]; then
	tgt_lang=fr
	data_dir=multi30k.en-fr
elif [ $task == 'multi30k-en2cs' ]; then
	tgt_lang=cs
	data_dir=multi30k.en-cs
elif [ $task == 'msctd' ]; then
	tgt_lang=de
        	data_dir=msctd.en-de
fi

if [ $image_feat == "clip" ]; then
	synth_feat_path=data/$image_feat/synth_
                 authe_feat_path=data/$image_feat/authe_
	image_feat_dim=1
                 image_feat_len=512
fi

# data set
ensemble=10
batch_size=128
beam=5
src_lang=en

model_dir=$model_root_dir/$task/$model_dir_tag

checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi

output=$model_dir/translation_$who.log

export CUDA_VISIBLE_DEVICES=$gpu

cmd="fairseq-generate data-bin/$data_dir 
  -s $src_lang -t $tgt_lang 
  --path $model_dir/$checkpoint 
  --gen-subset $who 
  --batch-size $batch_size --beam $beam --lenpen $length_penalty 
  --task image_mmt
  --remove-bpe --quiet
  --synth-feat-path $synth_feat_path --image-feat-dim $image_feat_dim --image-feat-len $image_feat_len  
  --authe-feat-path $authe_feat_path
  --output $model_dir/hypo.txt" 

cmd=${cmd}" | tee "${output}
eval $cmd

python3 rerank.py $model_dir/hypo.txt $model_dir/hypo.sorted