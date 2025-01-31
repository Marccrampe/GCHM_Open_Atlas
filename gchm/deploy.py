from osgeo import gdal, osr, ogr, gdalconst
import os
import sys
import numpy as np
import argparse
from pathlib import Path
import copy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


from gchm.models.architectures import Architectures
from gchm.utils.transforms import Normalize, NormalizeVariance, denormalize
from gchm.datasets.dataset_sentinel2_deploy import Sentinel2Deploy
from gchm.utils.gdal_process import save_array_as_geotif
from gchm.utils.parser import load_args_from_json, str2bool, str_or_none


def setup_parser():
    parser = argparse.ArgumentParser()

    # Add arguments based on the provided configuration
    parser.add_argument("--model_dir", default='/content/drive/MyDrive/GCHM_geo_inferences/trained_models/GLOBAL_GEDI_2019_2020', 
                        help="Répertoire du modèle pré-entrainé")
    parser.add_argument("--deploy_image_path", default='/content/drive/MyDrive/GCHM_geo_inferences/DATASET_GEE_TIF', 
                        help="Chemin vers le dossier contenant les images Sentinel-2 pour le déploiement")
    parser.add_argument("--deploy_dir", default='/content/drive/MyDrive/GCHM_geo_inferences/Predictions', 
                        help="Répertoire de sortie pour enregistrer les prédictions")
    parser.add_argument("--filepath_failed_image_paths", default="", 
                        help="Chemin vers le fichier .txt pour écrire les chemins des images échouées")
    parser.add_argument("--deploy_patch_size", default=64, type=int, 
                        help="Taille des patches carrés (hauteur=largeur) utilisés pour la prédiction")
    parser.add_argument("--deploy_batch_size", default=2, type=int, 
                        help="Taille du batch : Nombre de patches par batch pendant la prédiction")
    parser.add_argument("--num_workers_deploy", default=4, type=int, 
                        help="Nombre de workers dans le DataLoader")
    parser.add_argument("--num_models", default=5, type=int, 
                        help="Nombre de modèles dans l'ensemble (model_dir/model_0)")
    parser.add_argument("--save_latlon_masks", type=str2bool, nargs='?', const=True, default=False, 
                        help="Si True : Enregistre les masques latlon utilisés pour la prédiction en tant que geotif.")
    parser.add_argument("--remove_image_after_pred", type=str2bool, nargs='?', const=True, default=False, 
                        help="Si True : Supprime l'image après avoir enregistré la prédiction.")
    
    # Sentinel-2 Directory (temporary storage path)
    parser.add_argument("--sentinel2_dir", default="", 
                        help="Répertoire pour enregistrer les données Sentinel-2 (temporairement)")
    
    # Fine-tuning strategy
    parser.add_argument("--finetune_strategy", default='FT_Lm_SRCB', choices=['', None,
                        'FT_ALL_CB', 'FT_L_CB', 'RT_L_CB',
                        'FT_ALL_SRCB', 'FT_L_SRCB', 'RT_L_SRCB', 
                        'FT_Lm_SRCB', 'RT_Lm_SRCB', 'RT_L_IB',
                        'ST_geoshift_IB', 'ST_geoshiftscale_IB'], 
                        help="Stratégie de fine-tuning du modèle")

    return parser



def predict(model, args, model_weights=None,
            ds_pred=None, batch_size=1, num_workers=8,
            train_target_mean=0, train_target_std=1):
    DEVICE = torch.device("cuda:0")

    train_target_mean = torch.tensor(train_target_mean)
    train_target_std = torch.tensor(train_target_std)

    dl_pred = DataLoader(ds_pred, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model.load_state_dict(model_weights)
    model.eval()

    pred_dict = {'predictions': []}
    if args.return_variance:
        pred_dict['std'] = []

    with torch.no_grad():
        for step, data_dict in enumerate(tqdm(dl_pred, ncols=100, desc='pred', file=sys.stdout)):
            inputs = data_dict[args.input_key]
            inputs = inputs.to(DEVICE, non_blocking=True)

            if args.return_variance:
                predictions, variances = model.forward(inputs)
                std = torch.sqrt(variances)
                pred_dict['std'].extend(list(std.cpu()))
            else:
                predictions = model.forward(inputs)
            pred_dict['predictions'].extend(list(predictions.cpu()))

        for key in pred_dict.keys():
            if pred_dict[key]:
                pred_dict[key] = torch.stack(pred_dict[key], dim=0)
                print("val_dict['{}'].shape: ".format(key), pred_dict[key].shape)

    if args.normalize_targets:
        pred_dict['predictions'] = denormalize(pred_dict['predictions'], train_target_mean, train_target_std)
        if args.return_variance:
            pred_dict['std'] *= train_target_std

    for key in pred_dict.keys():
        pred_dict[key] = pred_dict[key].data.cpu().numpy()

    return pred_dict
