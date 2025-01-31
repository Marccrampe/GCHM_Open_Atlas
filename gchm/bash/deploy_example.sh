#!/bin/bash

DEPLOY_IMAGE_PATH="./deploy_example/sentinel2/2020/S2A_MSIL2A_20200623T103031_N0214_R108_T32TMT_20200623T142851.zip"
GCHM_DEPLOY_DIR="./deploy_example/predictions/2020"

GCHM_MODEL_DIR="./trained_models/GLOBAL_GEDI_2019_2020"
GCHM_NUM_MODELS="5"

filepath_failed_image_paths="./deploy_example/log_failed.txt"

# AWS-related settings removed
# GCHM_DOWNLOAD_FROM_AWS="False" # Not needed anymore
# GCHM_DEPLOY_SENTINEL2_AWS_DIR="./deploy_example/sentinel2_aws" # Not needed anymore

# create directories
mkdir -p ${GCHM_DEPLOY_DIR}

# Use the local Sentinel-2 directory for the deployment
export sentinel2_dir="./deploy_example/sentinel2/${YEAR}"

python3 gchm/deploy.py --model_dir=${GCHM_MODEL_DIR} \
                       --deploy_image_path=${DEPLOY_IMAGE_PATH} \
                       --deploy_dir=${GCHM_DEPLOY_DIR} \
                       --deploy_patch_size=512 \
                       --num_workers_deploy=4 \
                       --num_models=${GCHM_NUM_MODELS} \
                       --finetune_strategy="FT_Lm_SRCB" \
                       --filepath_failed_image_paths=${filepath_failed_image_paths} \
                       --sentinel2_dir=${sentinel2_dir} \
                       --remove_image_after_pred="False"
