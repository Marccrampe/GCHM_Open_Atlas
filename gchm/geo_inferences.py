from datetime import datetime, timedelta
import glob
import os
import numpy as np
import json
import torch
from pathlib import Path
import sys
import copy

sys.path.append('/content/drive/MyDrive/GCHM_geo_inferences/')

from gchm.models.architectures import Architectures
from gchm.utils.transforms import Normalize
from gchm.datasets.dataset_sentinel2_deploy import Sentinel2Deploy
from gchm.utils.gdal_process import save_array_as_geotif
from gchm.utils.parser import load_args_from_json, str2bool, str_or_none
from ee_preprocess import initialize_gee, load_and_validate_geojson, create_composite, export_image_to_drive,get_dlc_mask
from deploy import predict,setup_parser

# Config file path
CONFIG_PATH = "/content/drive/MyDrive/GCHM_geo_inferences/gchm/config.json"

# Load the configuration
with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)

# Configuration variables from config.json
SERVICE_ACCOUNT_KEY = config["service_account_key"]
GEOJSON_PATH = config["geojson_path"]
OUTPUT_TIF_DIR = config["tif_dir"]
DATE_ARRAY = config["date_array"]  # List of dates instead of start and end
AOI_NAME = config["aoi_name"]
PATCH_SIZE = config["batch_size"]
MODEL_DIR = config["model_dir"]
DEPLOY_IMAGE_PATH = config["image_path"]
DEPLOY_DIR = config["deploy_dir"]
NUM_MODELS = config["num_models"]
#FINETUNE_STRATEGY = config["finetune_strategy"]
SAVE_LATLON_MASKS = config["save_latlon_masks"]
REMOVE_IMAGE_AFTER_PRED = config["remove_image_after_pred"]

# Initialize GEE
initialize_gee(SERVICE_ACCOUNT_KEY)

# Load and validate the GeoJSON
geometry = load_and_validate_geojson(GEOJSON_PATH)

# Get DLC masks for all the dates in the array (dict[date,mask])
dlc_masks = get_dlc_mask(geometry,DATE_ARRAY)

# Export images for each date to Google Drive (if needed)
for date in DATE_ARRAY:
    lowest_cloud_image = create_composite(date, geometry)
    reprojected = lowest_cloud_image.reproject(crs='EPSG:4326', scale=10)
    export_image_to_drive(reprojected, f"{AOI_NAME}_{date}", OUTPUT_TIF_DIR, geometry)
    print(f"Pre-process completed for {date}")



# Setup the device
DEVICE = torch.device("cuda:0")
print('DEVICE: ', DEVICE, torch.cuda.get_device_name(0))

#Take randomly one of the 5 CNNS
num_model = np.random.choice(5)
print('Number of the model : ' ,num_model)
finetune_strategy = "FT_Lm_SRCB"

#Set the path to the model weight
model_dir_path = os.path.join(MODEL_DIR, f"model_{num_model}", finetune_strategy)

# Load training statistics
train_input_mean = np.load(os.path.join(model_dir_path, 'train_input_mean.npy'))
train_input_std = np.load(os.path.join(model_dir_path, 'train_input_std.npy'))
train_target_mean = np.load(os.path.join(model_dir_path, 'train_target_mean.npy'))
train_target_std = np.load(os.path.join(model_dir_path, 'train_target_std.npy'))


print("Training statistics loaded successfully:")
print(f"train_input_mean: {train_input_mean}")
print(f"train_input_std: {train_input_std}")
print(f"train_target_mean: {train_target_mean}")
print(f"train_target_std: {train_target_std}")

# Setup input transforms
input_transforms = Normalize(mean=train_input_mean, std=train_input_std)

print(f"DEPLOY_IMAGE_PATH: {DEPLOY_IMAGE_PATH}")  # Vérifier le chemin utilisé
print("Contenu du dossier :", os.listdir(DEPLOY_IMAGE_PATH))  # Lister les fichiers dans le dossier

# Path to the images to deploy
image_paths = glob.glob(f"{DEPLOY_IMAGE_PATH}/*.tif")
print(image_paths)


#Set the args for the model 
# Parse deploy arguments
parser = setup_parser()
args, unknown = parser.parse_known_args()

args.model_id = num_model
args.model_dir = model_dir_path

args_dict = vars(args)
args_dict_deploy = copy.deepcopy(args_dict)

    # Set missing args:
if not hasattr(args, 'channels'):
    args.channels = 12   
if not hasattr(args, 'return_variance'):
    args.return_variance = True   
if not hasattr(args, 'input_lat_lon'):
    args.input_lat_lon = True
if not hasattr(args, 'manual_init'):
    args.manual_init = False
if not hasattr(args, 'freeze_last_mean'):
    args.freeze_last_mean = False
if not hasattr(args, 'freeze_last_var'):
    args.freeze_last_var = False
if not hasattr(args, 'geo_shift'):
    args.geo_shift = False
if not hasattr(args, 'geo_scale'):
    args.geo_scale = False
if not hasattr(args, 'separate_lat_lon'):
    args.separate_lat_lon = True
print(args_dict)

if args.input_lat_lon:
    args.channels = 15 

# Load args from experiment dir with full train set
print("Loading args from trained models directory...")
args_saved = load_args_from_json(os.path.join(args.model_dir, 'args.json'))
# Update args with model_dir args
args_dict.update(args_saved)
# Update args with deploy args
args_dict.update(args_dict_deploy)


for image_path in image_paths:
    print(f"Processing image: {image_path}")
    
    # Extract date from image name (assuming the image name contains the date in YYYY-MM-DD format)
    base_filename = os.path.basename(image_path)
    base_filename_without_extension = os.path.splitext(base_filename)[0]  # retire l'extension .tif

    date_str = base_filename_without_extension.split('_')[1]  # Adjust according to your naming convention
    print(f"Image date extracted: {date_str}")
    
    # Load dataset for each image
    ds_pred = Sentinel2Deploy(
        path=image_path,
        input_transforms=input_transforms,
        input_lat_lon=True,  # Adjust as needed
        patch_size=PATCH_SIZE,
        border=8
    )

    # Load model architecture
    architecture_collection = Architectures(args=args)
    net = architecture_collection(args.architecture)(num_outputs=1)

    net.cuda()  # Move model to GPU

    # Load latest weights from checkpoint file
    print('Loading model weights from latest checkpoint ...')
    checkpoint_path = Path(model_dir_path) / 'checkpoint.pt'
    checkpoint = torch.load(checkpoint_path)
    model_weights = checkpoint['model_state_dict']

    # Predict for this image
    pred_dict = predict(
        model=net,
        args=args,
        model_weights=model_weights,
        ds_pred=ds_pred,
        batch_size=PATCH_SIZE,
        num_workers=4,
        train_target_mean=train_target_mean,
        train_target_std=train_target_std
    )

    # Recompose predictions and apply DLC mask for this image
    for k in pred_dict:
        recomposed_tiles = ds_pred.recompose_patches(pred_dict[k], out_type=np.float32)

        if date_str in dlc_masks:
            dlc_mask = dlc_masks[date_str]
            print(f"Unique values in DLC mask for {date_str}: {np.unique(dlc_mask)}")
        
            print("Shape of recomposed_tiles: ", recomposed_tiles.shape)
            print("Shape of dlc_mask: ", dlc_mask.shape)
            dlc_mask = np.array(dlc_mask, dtype=np.float32)  # Convert to a regular NumPy array

            # Appliquer le masque (mettre à NaN ou 0 les valeurs masquées)
            recomposed_tiles[dlc_mask == 0] = np.NaN  # Ou une autre valeur selon ton besoin

            # Sauvegarder l'image masquée
            tif_path = os.path.join(DEPLOY_DIR, f"{base_filename_without_extension}_{k}.tif")
            print(f"Saving to: {tif_path}")
            save_array_as_geotif(tif_path, recomposed_tiles, ds_pred.tile_info)
    
    # Save lat/lon masks if needed
    if SAVE_LATLON_MASKS:
        lat_path = os.path.join(DEPLOY_DIR, f"{base_filename_without_extension}_lat.tif")
        lon_path = os.path.join(DEPLOY_DIR, f"{base_filename_without_extension}_lon.tif")

#        save_array_as_geotif(lat_path, ds_pred.lat_mask[ds_pred.border:-ds_pred.border, ds_pred.border:-ds_pred.border], ds_pred.tile_info)
#        save_array_as_geotif(lon_path, ds_pred.lon_mask[ds_pred.border:-ds_pred.border, ds_pred.border:-ds_pred.border], ds_pred.tile_info)

    # Optionally remove original image after processing
    if REMOVE_IMAGE_AFTER_PRED:
        os.remove(image_path)
        print(f"Removed original image: {image_path}")
