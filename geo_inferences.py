import boto3
from google.cloud import storage
from datetime import datetime, timedelta
import numpy as np
import json
import torch
from pathlib import Path
import sys
import copy
import os
import time
import ee
import rasterio

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gchm.models.architectures import Architectures
from gchm.utils.transforms import Normalize
from gchm.datasets.dataset_sentinel2_deploy import Sentinel2Deploy
from gchm.utils.gdal_process import save_array_as_geotif
from gchm.utils.parser import load_args_from_json, str2bool, str_or_none
from ee_preprocess import (
    initialize_gee, load_and_validate_geojson, get_square_encompassing_polygon,
    create_composite, export_image_to_gcs, get_dlc_mask,
    move_image_after_prediction,move_pred
)
from deploy import predict, setup_parser


import google.auth


# ✅ 1. Load script parameters
parser = setup_parser()
args, unknown = parser.parse_known_args()
bucket_gcs = "gchm-predictions-test"

# Charger les credentials explicitement
credentials, project = google.auth.load_credentials_from_file(args.service_account_key)
storage_client = storage.Client(credentials=credentials, project=project)

# ✅ 2. Initialize Google Earth Engine
initialize_gee(args.service_account_key)

# ✅ 3. Load and validate AOI (GeoJSON)
aoi = load_and_validate_geojson(args.bucket_name, args.geojson_path)

# ✅ 4. Generate AOI geometry and retrieve DLC masks
geometry = get_square_encompassing_polygon(aoi)
dlc_masks = get_dlc_mask(geometry, args.date_array)

# ✅ 5. Export Sentinel-2 images from GEE to GCS (`DATASET_GEE_TIF/`)
for date in args.date_array:
    lowest_cloud_image = create_composite(date, geometry)
    reprojected = lowest_cloud_image.reproject(crs='EPSG:4326', scale=10)
    export_image_to_gcs(reprojected, f"{args.aoi_name}_{date}", bucket_gcs, geometry)
    print(f"🛰️ Image for {date} exported to gs://{bucket_gcs}/DATASET_GEE_TIF/")

# ✅ 6. Verify that the images are available in GCS
bucket = storage_client.bucket(bucket_gcs)
blobs = bucket.list_blobs(prefix="DATASET_GEE_TIF/")
image_names = [blob.name.replace("DATASET_GEE_TIF/", "") for blob in blobs if blob.name.endswith(".tif")]

if not image_names:
    print("❌ No images found in DATASET_GEE_TIF/. Exiting.")
    exit()

print(f"📂 Images found in DATASET_GEE_TIF/: {image_names}")

# ✅ 7. Load a random model from the 5 available CNNs
num_model = np.random.choice(5)
print('Selected model number:', num_model)
model_dir_path = os.path.join(args.model_dir, f"model_{num_model}", args.finetune_strategy)

# Set model-related arguments
args.model_id = num_model
args.model_dir = model_dir_path
if args.input_lat_lon:
    args.channels = 15 

args_dict = vars(args)
args_dict_deploy = copy.deepcopy(args_dict)

# Load args from experiment dir with full train set
print("Loading args from trained models directory...")
args_saved = load_args_from_json(os.path.join(args.model_dir, 'args.json'))
# Update args with model_dir args
args_dict.update(args_saved)
# Update args with deploy args
args_dict.update(args_dict_deploy)


# Setup the device
DEVICE = torch.device("cuda:0")
print('DEVICE: ', DEVICE, torch.cuda.get_device_name(0))


# Load training statistics
train_input_mean = np.load(os.path.join(model_dir_path, 'train_input_mean.npy'))
train_input_std = np.load(os.path.join(model_dir_path, 'train_input_std.npy'))
train_target_mean = np.load(os.path.join(model_dir_path, 'train_target_mean.npy'))
train_target_std = np.load(os.path.join(model_dir_path, 'train_target_std.npy'))

input_transforms = Normalize(mean=train_input_mean, std=train_input_std)

# ✅ 8. Load the model
architecture_collection = Architectures(args=args)
net = architecture_collection(args.architecture)(num_outputs=1)
net.cuda()  # Move model to GPU

# Load model weights
print('Loading model weights from latest checkpoint ...')
checkpoint_path = Path(model_dir_path) / 'checkpoint.pt'
checkpoint = torch.load(checkpoint_path)
model_weights = checkpoint['model_state_dict']

# ✅ 9. Loop through each image for inference
for image_name in image_names:
    print(f"🖼️ Processing image: {image_name}")

    # ⬇️ Download the image from GCS to a local temp file
    local_image_path = f"/tmp/{image_name}"
    blob = bucket.blob(f"DATASET_GEE_TIF/{image_name}")
    blob.download_to_filename(local_image_path)

    # Extract the date from the file name
    base_filename_without_extension = os.path.splitext(image_name)[0]
    date_str = base_filename_without_extension.split('_')[1]  # Adjust according to your naming convention

    # Load the image into Sentinel2Deploy
    ds_pred = Sentinel2Deploy(
        path=local_image_path,
        input_transforms=input_transforms,
        input_lat_lon=args.input_lat_lon,
        patch_size=args.batch_size,
        border=8
    )

    # Perform prediction
    pred_dict = predict(
        model=net,
        args=args,
        model_weights=model_weights,
        ds_pred=ds_pred,
        batch_size=args.batch_size,
        num_workers=4,
        train_target_mean=train_target_mean,
        train_target_std=train_target_std
    )



    # ✅ 10. Save and upload the prediction to GCS (`Predictions/`)
    for k in pred_dict:
        recomposed_tiles = ds_pred.recompose_patches(pred_dict[k], out_type=np.float32)
        if date_str in dlc_masks:
            dlc_mask = dlc_masks[date_str]
            print(f"Unique values in DLC mask for {date_str}: {np.unique(dlc_mask)}")

            dlc_mask = np.array(dlc_mask, dtype=np.float32)  # Convert to NumPy array
            recomposed_tiles[dlc_mask == 0] = np.NaN  # Apply the mask

        # Define output file path
        output_filename = f"{base_filename_without_extension}_{k}.tif"
        local_output_path = f"/tmp/{output_filename}"
        save_array_as_geotif(local_output_path, recomposed_tiles, ds_pred.tile_info)

        # ⬆️ Upload the file to GCS (`Predictions/`)
        prediction_blob = bucket.blob(f"GCHM_AWS_Transfer/{output_filename}")
        prediction_blob.upload_from_filename(local_output_path)
        print(f"✅ Prediction saved to gs://{bucket_gcs}/GCHM_AWS_Transfer/{output_filename}")
        move_pred(output_filename, bucket_gcs, args.bucket_name)

    # ✅ 11. Move the original image after prediction (prevents infinite loop)
    move_image_after_prediction(image_name, bucket_gcs, args.bucket_name)


