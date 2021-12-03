#!/bin/bash
script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}
cd ..
nvidia-smi

MODEL=dpn
DATANAME=cub
BACKBONE=resnet101
DATAPATH=
SAVEPATH=/output/${DATANAME}/${script_name}
RESNETPRE=

STAGE1=1
STAGE2=1

if [ ${STAGE1} = 1 ]
then
  python main.py \
    --batch-size 256 \
    --lr 2e-4 \
    --epochs 90 \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --is_fix \
    --seed 8432 \
    --seed1 5554 \
    --seed2 9169 \
    --resnet_pretrain ${RESNETPRE}
fi

if [ ${STAGE2} = 1 ]
then
  python main.py \
    --batch-size 32 \
    --lr 2e-4 \
    --epochs 90 \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --resume ${SAVEPATH}/fix.model \
    --seed 414 \
    --seed1 8375 \
    --seed2 5004 \
    --resnet_pretrain ${RESNETPRE}
fi
