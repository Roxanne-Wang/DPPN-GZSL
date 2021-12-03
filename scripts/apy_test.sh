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
MODELPATH=

STAGE1=1

if [ ${STAGE1} = 1 ]
then
  python main.py \
    --batch-size 64 \
    --att-emb-dim ${ATTEMBDIM} \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --is_fix  \
    --resume ${MODELPATH} \
    --eval_only
fi
