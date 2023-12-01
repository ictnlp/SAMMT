#! /usr/bin/bash
set -e

device=0
task=multi30k-en2de
image_feat=clip  #vit_tiny_patch16_384
date=$(date '+%m%d%H%M')
save_dir=checkpoints/$task/$image_feat/${date}_0.5klmulti_32_sd_ran0+0.1beforeimgotlossvishal_release

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi

if [ $task == 'multi30k-en2de' ]; then
	src_lang=en
	tgt_lang=de
        	data_dir=multi30k.en-de
elif [ $task == 'multi30k-en2fr' ]; then
	src_lang=en
	tgt_lang=fr
        	data_dir=multi30k.en-fr
elif [ $task == 'multi30k-en2cs' ]; then
	src_lang=en
	tgt_lang=cs
        	data_dir=multi30k.en-cs
elif [ $task == 'msctd' ]; then
	src_lang=en
	tgt_lang=de
        	data_dir=msctd.en-de
fi

criterion=label_smoothed_cross_entropy
fp16=0
lr=0.005
warmup=2000
max_tokens=2048
update_freq=4
keep_last_epochs=10
patience=10
max_update=9500
dropout=0.3

arch=multimodal_transformer_sammt
image_dropout=0.1

if [ $image_feat == "clip" ]; then
	synth_feat_path=data/$image_feat/synth_
                 authe_feat_path=data/$image_feat/authe_
	image_feat_dim=1
                 image_feat_len=512
fi

cp ${BASH_SOURCE[0]} $save_dir/train.sh

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

cmd="fairseq-train data-bin/$data_dir
  --save-dir $save_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --dropout $dropout
  --criterion $criterion --label-smoothing 0.1
  --task image_mmt --synth-feat-path $synth_feat_path --image-feat-dim $image_feat_dim --image-feat-len $image_feat_len  
  --authe-feat-path $authe_feat_path
  --optimizer adam --adam-betas '(0.9, 0.98)'
  --lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --max-tokens $max_tokens --update-freq $update_freq
  --find-unused-parameters --share-all-embeddings
  --max-update $max_update --keep-last-epochs $keep_last_epochs"

if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi

if [ -n "$image_dropout" ]; then
cmd=${cmd}" --image-dropout "${image_dropout}
fi

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log