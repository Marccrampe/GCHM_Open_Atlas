import os
import osgeo
from osgeo import gdal, osr, ogr, gdalconst
import numpy as np
from skimage.transform import resize
from zipfile import ZipFile
import time
import rasterio

gdal.UseExceptions()


GDAL_TYPE_LOOKUP = {'float32': gdal.GDT_Float32,
                    'float64': gdal.GDT_Float64,
                    'uint16': gdal.GDT_UInt16,
                    'uint8': gdal.GDT_Byte}

def sort_band_arrays(band_arrays, channels_last=True):
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    out_arr = []
    for b in bands:
        out_arr.append(band_arrays[b])
    out_arr = np.array(out_arr)
    if channels_last:
        out_arr = np.moveaxis(out_arr, source=0, destination=-1)
    return out_arr


def read_band(path_band, num_retries=10, max_sleep_sec=5):
    for i in range(num_retries):
        try:
            ds = gdal.Open(path_band)
            band = ds.GetRasterBand(1)
            print('reading full band array...')
            band_array = band.ReadAsArray()
            return band_array
        except:
            print('Attempt {}/{} failed reading path: {}'.format(i, num_retries, path_band))
            time.sleep(np.random.randint(max_sleep_sec))
            continue
        # raise an error if max retries is reached
    raise RuntimeError("read_band() failed {} times reading path: {}".format(num_retries, path_band))


def read_sentinel2_bands(data_path, channels_last=False):
    import numpy as np
    from skimage.transform import resize
    import rasterio

    # Listes des bandes à différentes résolutions
    bands10m = ['B2', 'B3', 'B4', 'B8']
    bands20m = ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12', 'SCL']
    bands60m = ['B1', 'B9']

    # Dictionnaire pour les bandes
    bands_dir = {10: bands10m, 20: bands20m, 60: bands60m}
    band_arrays = {}
    tile_info = None

    # Ouverture du fichier GeoTIFF contenant toutes les bandes
    with rasterio.open(data_path) as src:
        # Extraction des métadonnées (informations sur la tuile)
        tile_info = {
          'projection': src.crs.to_wkt(),  # Convertir le CRS en chaîne WKT
          'geotransform': [src.transform[2], src.transform[0], 0, src.transform[5], 0, src.transform[4]],  # Transformation géospatiale
          'width': src.width,  # Largeur du raster en pixels
          'height': src.height,  # Hauteur du raster en pixels
          'bounds': src.bounds  # Limites géographiques du raster
      }



        # Vérification des descriptions des bandes
        band_descriptions = src.descriptions

        # Lecture des bandes
        for resolution, band_names in bands_dir.items():
            for band_name in band_names:
                if band_name in band_descriptions:
                    band_index = band_descriptions.index(band_name) + 1  # +1 car les indices sont à partir de 1
                    band_arrays[band_name] = src.read(band_index)
                else:
                    print(f"Warning: Bande {band_name} non trouvée dans le fichier.")

        # Lecture de la bande de probabilité des nuages (renommée en CLD)
        if 'MSK_CLDPRB' in band_descriptions:
            band_index = band_descriptions.index('MSK_CLDPRB') + 1
            band_arrays['CLD'] = src.read(band_index)
        else:
            print("Warning: Bande des nuages (MSK_CLDPRB) non trouvée dans le fichier.")

    # Redimensionnement des bandes pour correspondre à la taille de la bande B2 (si elle existe)
    target_shape = band_arrays.get('B2').shape if 'B2' in band_arrays else None
    if target_shape is not None:
        for band_name, band_array in band_arrays.items():
            if band_array.shape != target_shape:
                order = 0 if band_name == 'SCL' else 3
                band_arrays[band_name] = resize(
                    band_array,
                    target_shape,
                    mode='reflect',
                    order=order,
                    preserve_range=True,
                ).astype(np.uint16)

    # Tri des bandes avec votre fonction sort_band_arrays
    print('sorting bands...')
    print(tile_info)
    image_array = sort_band_arrays(band_arrays=band_arrays, channels_last=channels_last)

    # Retourner les objets dans le format souhaité
    return image_array, tile_info, band_arrays.get('SCL'), band_arrays.get('CLD')



def to_latlon(x, y, ds):
    bag_gtrn = ds.GetGeoTransform()
    bag_proj = ds.GetProjectionRef()
    bag_srs = osr.SpatialReference(bag_proj)
    geo_srs = bag_srs.CloneGeogCS()
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        bag_srs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
        geo_srs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(bag_srs, geo_srs)

    # in a north up image:
    originX = bag_gtrn[0]
    originY = bag_gtrn[3]
    pixelWidth = bag_gtrn[1]
    pixelHeight = bag_gtrn[5]

    easting = originX + pixelWidth * x + bag_gtrn[2] * y
    northing = originY + bag_gtrn[4] * x + pixelHeight * y

    geo_pt = transform.TransformPoint(easting, northing)[:2]
    lon = geo_pt[0]
    lat = geo_pt[1]
    return lat, lon


def create_latlon_mask(height, width, refDataset, out_type=np.float32):
    # compute lat, lon of top-left and bottom-right corners
    lat_topleft, lon_topleft = to_latlon(x=0, y=0, ds=refDataset)
    lat_bottomright, lon_bottomright = to_latlon(x=width-1, y=height-1, ds=refDataset)

    # interpolate between the corners
    lat_col = np.linspace(start=lat_topleft, stop=lat_bottomright, num=height).astype(out_type)
    lon_row = np.linspace(start=lon_topleft, stop=lon_bottomright, num=width).astype(out_type)

    # expand dimensions of row and col vector to repeat
    lat_col = lat_col[:, None]
    lon_row = lon_row[None, :]

    # repeat column and row to get 2d arrays --> lat lon coordinate for every pixel
    lat_mask = np.repeat(lat_col, repeats=width, axis=1)
    lon_mask = np.repeat(lon_row, repeats=height, axis=0)

    print('lat_mask.shape: ', lat_mask.shape)
    print('lon_mask.shape: ', lon_mask.shape)

    return lat_mask, lon_mask



def get_reference_band_path(path_tif_file):
    # Vérifie si le fichier existe
    if os.path.exists(path_tif_file):
        return path_tif_file
    else:
        raise FileNotFoundError(f"Le fichier {path_tif_file} n'a pas été trouvé.")

def get_reference_band_ds_gdal(path_file):
    # Récupère le chemin vers le fichier TIF
    refDataset_path = get_reference_band_path(path_file)
    
    # Ouvre le fichier TIF avec GDAL
    ds = gdal.Open(refDataset_path)
    
    # Vérifie si le dataset a été correctement ouvert
    if ds is None:
        raise RuntimeError(f"Le fichier {refDataset_path} n'a pas pu être ouvert.")
    
    # Retourne l'objet dataset complet
    return ds



def get_tile_info(refDataset):
    tile_info = {}
    # Extraction des informations de projection et de géotransformation
    tile_info['projection'] = refDataset.GetProjection()
    tile_info['geotransform'] = refDataset.GetGeoTransform()

    # Vérifier si la géotransformation a été extraite correctement
    if tile_info['geotransform'] is None:
        raise ValueError("La géotransformation est invalide ou inexistante.")
    
    tile_info['width'] = refDataset.RasterXSize
    tile_info['height'] = refDataset.RasterYSize
    print(f"Tile info: {tile_info}")  # Debugging line
    return tile_info


def save_array_as_geotif(out_path, array, tile_info, out_type=None, out_bands=1, dstnodata=None,
                         compress='DEFLATE', predictor=2):
    if out_type is None:
        out_type = array.dtype.name
    out_type = GDAL_TYPE_LOOKUP[out_type]
    # PACKBITS is a lossless compression.
    # predictor=2 saves horizontal differences to previous value (useful for empty regions)
    dst_ds = gdal.GetDriverByName('GTiff').Create(out_path, tile_info['width'], tile_info['height'], out_bands, out_type,
                                                  options=['COMPRESS={}'.format(compress), 'PREDICTOR={}'.format(predictor)])
    dst_ds.SetGeoTransform(tile_info['geotransform'])
    dst_ds.SetProjection(tile_info['projection'])
    dst_ds.GetRasterBand(1).WriteArray(array)  # write r-band to the raster
    if dstnodata is not None:
        dst_ds.GetRasterBand(1).SetNoDataValue(dstnodata)
    dst_ds.FlushCache()  # write to disk
    dst_ds = None


def load_tif_as_array(path, set_nodata_to_nan=True, dtype=float):
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)

    array = band.ReadAsArray().astype(dtype)
    tile_info = get_tile_info(ds)
    # set the nodata values to nan
    nodata_value = band.GetNoDataValue()
    tile_info['nodata_value'] = nodata_value
    if set_nodata_to_nan:
        array[array == nodata_value] = np.nan
    return array, tile_info

