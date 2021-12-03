#!/bin/bash
script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}
cd ..
nvidia-smi

ATTEMBDIM=21
MODEL=dpn
DATANAME=sun
BACKBONE=resnet101
DATAPATH=
SAVEPATH=/output/${DATANAME}/${script_name}
RESNETPRE=

STAGE1=1
if [ ${STAGE1} = 1 ]
then
  python main.py \
    --batch-size 64 \
    --att-emb-dim ${ATTEMBDIM} \
    --lr 2e-4 \
    --epochs 90 \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --save_H_OPT \
    --is_fix \
    --seed 9007 \
    --seed1 9007 \
    --seed2 5224 \
    --resnet_pretrain ${RESNETPRE}
fi
