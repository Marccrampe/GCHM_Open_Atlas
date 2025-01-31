import ee
import json
import os
from datetime import datetime, timedelta
import time
import numpy as np
import requests
from io import BytesIO

def initialize_gee(service_account_key):
    """Authenticate and initialize Google Earth Engine."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key
    ee.Initialize()

def load_and_validate_geojson(geojson_path):
    """Load and validate the GeoJSON file."""
    with open(geojson_path, 'r') as geojson_file:
        geojson = json.load(geojson_file)
    return ee.Geometry.Polygon(geojson['features'][0]['geometry']['coordinates'])

def create_composite(start_date, polygon, cloud_percentage=30):
    """Create a composite with lower cloud coverage."""
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = start_date_obj + timedelta(days=60)
    end_date = end_date_obj.strftime("%Y-%m-%d")
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(polygon)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_percentage)))
    
    sorted_collection = collection.sort('CLOUDY_PIXEL_PERCENTAGE')
    lowest_cloud_images = sorted_collection.limit(3)
    
    return lowest_cloud_images.median()



def export_image_to_drive(image, file_name, output_dir, polygon):
    """Export the image to Google Drive and wait until it's done."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=file_name,
        folder=output_dir,
        scale=10,
        region=polygon,
        crs='EPSG:4326',
        fileFormat='GeoTIFF',
        maxPixels=1e9
    )
    task.start()

    # Wait for the task to finish
    while task.active():
        print(f"Exporting {file_name}...")  # Optional: Display the export status
        time.sleep(30)  # Sleep for 10 seconds before checking again
    
    print(f"Export of {file_name} completed.")


# Function to get the DLC mask from Dynamic World (GOOGLE/DYNAMICWORLD/V1)

def get_dlc_mask(aoi, date_array):
    """Retrieve the DLC mask as a numpy array for the given area of interest and list of dates."""
    dlc_masks = {}  # Dictionary to store masks for each date
    
    for start_date in date_array:
        # Convert the start date to a datetime object and calculate the end date (60 days later)
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = start_date_obj + timedelta(days=90)
        end_date = end_date_obj.strftime("%Y-%m-%d")
        
        # Load the Dynamic World (DLC) dataset from Google Earth Engine
        dlc_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        
        # Filter the collection by the area of interest and date range
        dlc_filtered = dlc_collection.filterBounds(aoi).filterDate(start_date, end_date).first().clip(aoi)

        # Define unwanted land cover classes to exclude (e.g., water, urban, snow, etc.)
        exclude_classes = [0, 5, 6, 7, 8]  
        dlc_mask = dlc_filtered.select("label").neq(ee.Image.constant(exclude_classes)).reduce(ee.Reducer.min())

        # Convert the mask to a binary format (1 for valid land cover, 0 for excluded areas)
        dlc_mask_binary = dlc_mask.eq(1)

        dlc_mask_resampled = dlc_mask_binary.reproject(crs='EPSG:4326', scale=10)
        # Get the download URL for the mask
        url = dlc_mask_resampled.getDownloadURL({'scale': 10, 'region': aoi, 'format': 'NPY'})

        # Download the numpy array from the URL
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            mask_array = np.load(BytesIO(response.content))  # Load the image as a numpy array
            dlc_masks[start_date] = mask_array
        else:
            print(f"Failed to download DLC mask for {start_date}: HTTP {response.status_code}")
            dlc_masks[start_date] = None  # Store None if download fails

    return dlc_masks




