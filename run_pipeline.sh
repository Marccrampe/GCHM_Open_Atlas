#!/bin/bash
# Define variables for the GEE script
CONFIG_PATH="/Users/marccrampe/Desktop/OPEN_Atlas/Global_Height_Canopy_OA/config.json"
GEE_SCRIPT="/Users/marccrampe/Desktop/OPEN_Atlas/Global_Height_Canopy_OA/ee_preprocess.py"
# Run the GEE script
echo "Running the GEE export process..."
python3 "$GEE_SCRIPT" --config "$CONFIG_PATH"
# Check if the GEE script completed successfully
if [ $? -ne 0 ]; then
  echo "GEE script failed. Exiting..."
  exit 1
fi


# Define variables for the deployment command
GCHM_MODEL_DIR="/content/drive/MyDrive/GCHM_New_Archi/trained_models/GLOBAL_GEDI_2019_2020"
DEPLOY_IMAGE_PATH="/Users/marccrampe/Desktop/OPEN_Atlas/Global_Height_Canopy_OA/Dataset_GEE_TIF"
GCHM_DEPLOY_DIR="/Users/marccrampe/Desktop/OPEN_Atlas/Global_Height_Canopy_OA/Predictions"
FAIL_DIR="/Users/marccrampe/Desktop/OPEN_Atlas/Global_Height_Canopy_OA/Log_Failed/log_failed.txt"
S2_DIR="/Users/marccrampe/Desktop/OPEN_Atlas/Global_Height_Canopy_OA/Dataset_GEE_TIF"

# Loop through all image files in deploy_image_path
echo "Running the deployment command for all images..."
count=1
for deploy_image in "$DEPLOY_IMAGE_PATH"/*; do
  if [[ -f "$deploy_image" ]]; then  # Check if it's a file
    echo "*************************************"
    echo "Processing image ${count}: $deploy_image"
    
    # Run the deployment command
    python3 /content/drive/MyDrive/GCHM_GEE_Open_Atlas/gchm/deploy.py \
      --model_dir="${GCHM_MODEL_DIR}" \
      --deploy_image_path="${deploy_image}" \
      --deploy_dir="${GCHM_DEPLOY_DIR}" \
      --deploy_patch_size=512 \
      --num_workers_deploy=4 \
      --num_models=5 \
      --finetune_strategy="FT_Lm_SRCB" \
      --filepath_failed_image_paths="${FAIL_DIR}" \
      --download_from_aws="False" \
      --sentinel2_dir="${S2_DIR}" \
      --remove_image_after_pred="False"
    
    # Check if the deployment command completed successfully
    exit_status=$?  # store the exit status
    if [ $exit_status -ne 0 ]; then
      echo "Deployment failed for image $deploy_image. Logging failed image."
      echo "$deploy_image" >> "$FAIL_DIR"
    fi
    
    count=$((count + 1))
  fi
done

echo "All processes completed successfully!"