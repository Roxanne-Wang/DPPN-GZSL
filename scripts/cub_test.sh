#!/bin/bash
script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}
cd ..
nvidia-smi

MODEL=dpn
DATANAME=cub
BACKBONE=resnet101
SAVEPATH=/output/${DATANAME}/${script_name}
MODELPATH=
DATAPATH=


STAGE1=1

if [ ${STAGE1} = 1 ]
then
  python main.py \
    --batch-size 64 \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --is_fix \
    --resume ${MODELPATH} \
    --eval_only
fi