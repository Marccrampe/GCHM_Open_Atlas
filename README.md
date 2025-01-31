# GCHM Model - Adapted for Time Series on OpenAtlas

This repository contains an implementation of the **GCHM** model, originally developed by **Lang et al. (2023)**. This version has been adapted for **time series analysis** on **OpenAtlas**.

## 📖 Reference

Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2023). A high-resolution canopy height model of the Earth. *Nature Ecology & Evolution*, 1-12.

## 📂 Project Structure

The repository includes all necessary scripts to run the model.  
All files and resources can be accessed via the following link:  
[**GCHM_geo_inference - Google Drive**](https://drive.google.com/drive/folders/1ApEOdYm1NsVrIBE-GpUBfaE35ZAGrmvB?usp=drive_link)

## ⚠️ Requirements

### 1️⃣ Hardware Requirements  
- **GPU Required**: A **GPU** is necessary for efficient model execution.

### 2️⃣ File Placement  
- Download the folder from Google Drive and place it in:  
/content/drive/MyDrive

- If you encounter issues with file paths, modify them in:
- `deploy.py`
- `config.json`
- `ee_preprocess.py`
- `geo_inferences.py`

## 🚀 Running the Model

### 1️⃣ Prepare Your GeoJSON File  
- Place your **GeoJSON** file in the project directory.
- Update the file path in `config.json`.
- **Important:**
- The **Area of Interest (AoI)** must be a **perfect square**.
- If not, the CNN may fail due to batch processing issues.
- **Solution**: Use [geojson.io](https://geojson.io) to select an AoI and adjust it into a perfect square.

### 2️⃣ Run the Inference  
Execute the following command in Bash to run the model:

```bash
bash geoinference.py
```

###📊 Test Data
Pre-existing test runs are available on Google Drive:

Greece 2020
Greece 2024
You can visualize the results using QGIS.

###⚠️ Limitations

- Cloudy Areas: The model's accuracy may be affected in regions with high cloud coverage.


- The input image must be square; non-square images will cause errors.

-If the image tiles are too small, reduce the border size proportionally.
The default setting is border = 16 for a 100km × 100km tile.
Adjust this value accordingly for smaller tiles.

###🛠️ Troubleshooting
Path Issues?
Check and update the paths in deploy.py, config.json, and ee_preprocess.py.

AoI Format Issues?
Ensure your GeoJSON defines a perfect square.

Small Tile Issues?
Reduce the border parameter proportionally to the tile size.

###🔹 Acknowledgment
This project is based on the original GCHM model developed by Lang et al. (2023).
This version has been adapted for time series analysis on OpenAtlas.

###🔹 Contact
For any questions or improvements, feel free to reach out! 🚀



