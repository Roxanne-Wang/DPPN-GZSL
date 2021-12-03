#!/bin/bash
script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}
cd ..
nvidia-smi
#export CUDA_VISIBLE_DEVICES=1,3
ATTEMBDIM=21
MODEL=dpn
DATANAME=sun
BACKBONE=resnet101
DATAPATH=
SAVEPATH=/output/${DATANAME}/${script_name}
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
    --is_fix \
    --resume ${MODELPATH} \
    --eval_only
fi
