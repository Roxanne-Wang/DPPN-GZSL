#!/bin/bash
script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}
cd ..
nvidia-smi

ATTEMBDIM=32
MODEL=dpn_ood
DATANAME=apy
BACKBONE=resnet101
SAVEPATH=/output/${DATANAME}/${script_name}
DATAPATH=
RESNETPRE=

STAGE1=1
STAGE2=1

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
    --is_fix  \
    --seed 5279 \
    --seed1 8490 \
    --seed2 1057 \
    --resnet_pretrain ${RESNETPRE}
fi

if [ ${STAGE2} = 1 ]
then
  python main.py \
    --opti_type sgd \
    --batch-size 32 \
    --att-emb-dim ${ATTEMBDIM} \
    --lr 2e-5 \
    --epochs 90 \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --resume ${SAVEPATH}/fix.model \
    --seed 3829 \
    --seed1 9359 \
    --seed2 5730 \
    --resnet_pretrain ${RESNETPRE}
fi