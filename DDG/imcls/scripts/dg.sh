#!/bin/bash

cd ..

#DATA=~/kaiyang/data
#DASSL=~/kaiyang/code/Dassl.pytorch
# Author-added
DATA=../../data
DASSL=../../Dassl.pytorch

DATASET=$1
TRAINER=$2
NET=$3 # e.g. resnet18_ms_l123, resnet18_ms_l12
MIX=$4

if [ ${DATASET} == pacs ]; then
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == office_home_dg ]; then
    D1=art
    D2=clipart
    D3=product
    D4=real_world
elif [ ${DATASET} == vlcs ]; then
    D1=caltech
    D2=labelme
    D3=pascal
    D4=sun
elif [ ${DATASET} == digits_dg ]; then
    D1=mnist
    D2=mnist_m
    D3=svhn
    D4=syn
fi

for SEED in $(seq 1 5)
do
    for SETUP in $(seq 1 4)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file ${DASSL}/configs/datasets/dg/${DATASET}.yaml \
        --config-file configs/trainers/mixstyle/${DATASET}_${MIX}.yaml \
        --output-dir output/${DATASET}/${TRAINER}/${NET}/${MIX}/${T}/seed${SEED} \
        --num-clients 3 \
        --graph-conn 2 \
        MODEL.BACKBONE.NAME ${NET}
    done
done
