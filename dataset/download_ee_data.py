"""
This script processes and exports a time series of Normalized Difference Vegetation Index (NDVI) data from Landsat satellites
for a specific region of interest (ROI) from 1984 to 2024. 
It utilizes Google Earth Engine to access Landsat 5, 7, and 8 imagery, selecting the most recent available satellite for each period.
The script calculates mean NDVI for spring (October to April) and fall (May to September) seasons of each year, 
averaging across all available images within each period after filtering for low cloud cover (<20%). 
Preprocessing includes applying scale factors to surface reflectance bands and calculating NDVI using near-infrared and red bands. 
The resulting time series is exported as a multi-band GeoTIFF file, with each band representing the mean NDVI for a specific season and year.
The script handles data gaps and provides error reporting for each processed year, ensuring a comprehensive long-term vegetation analysis.
"""

import ee
import geemap
import os
import rasterio
from rasterio.merge import merge

# Landsat 9 OLI-2/TIRS-2 2021-Present https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2
# Landsat 8 OLI/TIRS 2013–Present https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
# Landsat 7 ETM+ 1999–2021  https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2
# Landsat 5 TM 1984–2012 https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2
# Landsat 4 TM 1982–1993 https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT04_C02_T1_L2
# Landsat Collection 2 Tier 1 Level 2 32-Day NDVI Composite  https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_COMPOSITES_C02_T1_L2_32DAY_NDVI#description


def create_geometries() -> dict[str, ee.Geometry.Point]:  # type: ignore
    """Create and return a dictionary of point geometries."""
    return {
        "geometry8": ee.Geometry.Point([34.82755914862318, 31.365302333927794]),
        "geometry9": ee.Geometry.Point([34.84352365667982, 31.37684447255592]),
        "geometry10": ee.Geometry.Point([34.888842260195446, 31.3579366093761]),
        "geometry11": ee.Geometry.Point([34.85107675726576, 31.401572852121834]),
        "geometry12": ee.Geometry.Point([35.02543440072939, 31.35637198741008]),
        "geometry13": ee.Geometry.Point([35.020542051486224, 31.37198250118421]),
        "geometry14": ee.Geometry.Point([34.8503397962372, 31.422900178668392]),
        "geometry15": ee.Geometry.Point([34.86038198678896, 31.427660851097695]),
        "geometry16": ee.Geometry.Point([34.877161886386126, 31.469946978988034]),
        "geometry17": ee.Geometry.Point([34.88120394884191, 31.50745763889178]),
        "geometry18": ee.Geometry.Point([34.87764197527013, 31.51583601317523]),
        "geometry19": ee.Geometry.Point([34.88214808641515, 31.54807565749123]),
        "geometry20": ee.Geometry.Point([34.89193278490148, 31.559046768020536]),
        "geometry21": ee.Geometry.Point([34.835069953785755, 31.58957623189093]),
        "geometry22": ee.Geometry.Point([34.83867484270177, 31.59897071316063]),
        "geometry23": ee.Geometry.Point([34.83193713365636, 31.61165358724694]),
        "geometry24": ee.Geometry.Point([34.92963041915097, 31.622500733127755]),
        "geometry25": ee.Geometry.Point([34.89688601149716, 31.60839394222365]),
        "geometry26": ee.Geometry.Point([34.91332258834042, 31.61219494604687]),
        "geometry27": ee.Geometry.Point([34.927835330614876, 31.65056177042573]),
        "geometry28": ee.Geometry.Point([34.91753912879618, 31.669993175126187]),
    }


def create_roi() -> ee.Geometry.Polygon:  # type: ignore
    return ee.Geometry.Polygon([[[34.74670927531805, 31.787419838054763], [34.74670927531805, 31.302894493856027], [35.08179228313055, 31.302894493856027], [35.08179228313055, 31.787419838054763]]])


def apply_scale_factors(image: ee.Image) -> ee.Image:
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    return image.addBands(optical_bands, None, True)


def calculate_ndvi(image: ee.Image) -> ee.Image:
    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")
    return image.addBands(ndvi)


def get_period_name(year: int, is_spring: bool) -> str:
    if is_spring:
        return f"Landsat_NDVI_spring_10-{year-1}_04-{year}"
    else:
        return f"Landsat_NDVI_fall_05-{year}_09-{year}"


def process_period(start_date: str, end_date: str, roi: ee.Geometry.Polygon) -> ee.Image | None:  # type: ignore
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    if start_year >= 2013:
        # Use Landsat 8 for 2013 onwards
        collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    elif start_year >= 1999:
        # Use Landsat 7 for 1999-2012
        collection = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
    else:
        # Use Landsat 5 for 1984-1998
        collection = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")

    filtered_collection = collection.filterBounds(roi).filterDate(start_date, end_date).filter(ee.Filter.lt("CLOUD_COVER_LAND", 20))

    if filtered_collection.size().getInfo() == 0:
        return None

    def apply_scale_factors(image):
        optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
        return image.addBands(optical_bands, None, True)

    def calculate_ndvi(image):
        if start_year >= 2013:
            # Landsat 8 band names
            nir = "SR_B5"
            red = "SR_B4"
        else:
            # Landsat 5 and 7 band names
            nir = "SR_B4"
            red = "SR_B3"

        ndvi = image.normalizedDifference([nir, red]).rename("ndvi")
        return image.addBands(ndvi)

    processed_collection = filtered_collection.map(apply_scale_factors).map(calculate_ndvi)
    ndvi = processed_collection.select("ndvi").mean()
    return ndvi.clip(roi)


def export_single_band(image: ee.Image, band_name: str, roi: ee.Geometry.Polygon, output_dir: str) -> str:  # type: ignore
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{band_name}.tif")

    url = image.select(band_name).getDownloadURL({"scale": 30, "crs": "EPSG:4326", "region": roi.getInfo()["coordinates"], "format": "GEO_TIFF"})

    geemap.download_file(url, file_path)
    print(f"Downloaded {band_name}.tif to {output_dir}")
    return file_path


def combine_tiffs(input_files: list, output_file: str):
    src_files_to_mosaic = []
    for file in input_files:
        src = rasterio.open(file)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans, "count": len(input_files)})

    with rasterio.open(output_file, "w", **out_meta) as dest:
        for i, file in enumerate(input_files, start=1):
            with rasterio.open(file) as src:
                dest.write(src.read(1), i)
                dest.set_band_description(i, os.path.basename(file)[:-4])  # Use filename as band description

    print(f"Combined TIFFs saved to {output_file}")


def export_time_series(time_series: ee.Image, roi: ee.Geometry.Polygon, output_dir: str, file_name: str) -> str:  # type: ignore
    band_names = time_series.bandNames().getInfo()
    if not band_names:
        print("No bands found in the image.")
        return ""

    single_band_files = []

    for band_name in band_names:
        single_band_file = export_single_band(time_series, band_name, roi, output_dir)
        single_band_files.append(single_band_file)

    combined_file_path = os.path.join(output_dir, f"{file_name}.tif")
    combine_tiffs(single_band_files, combined_file_path)

    # # Optionally, remove single band files to save space
    # for file in single_band_files:
    #     os.remove(file)

    return combined_file_path


def process_year(year: int, roi: ee.Geometry.Polygon) -> dict:  # type: ignore
    spring_start = f"{year-1}-10-01"
    spring_end = f"{year}-04-30"
    spring_ndvi = process_period(spring_start, spring_end, roi)

    fall_start = f"{year}-05-01"
    fall_end = f"{year}-09-30"
    fall_ndvi = process_period(fall_start, fall_end, roi)

    return {get_period_name(year, True): spring_ndvi, get_period_name(year, False): fall_ndvi}


def main():
    roi = create_roi()
    start_year = 1984
    end_year = 2024
    output_dir = "data"

    # Create a list to store all NDVI images
    ndvi_images = []

    for year in range(start_year, end_year + 1):
        try:
            print(f"Processing year {year}...")
            year_data = process_year(year, roi)
            for period_name, ndvi_image in year_data.items():
                if ndvi_image is not None:
                    ndvi_images.append(ndvi_image.rename(period_name))
                else:
                    print(f"No data available for {period_name}")
        except Exception as e:
            print(f"Error processing year {year}: {str(e)}")

    # Combine all NDVI images into a single multi-band image
    if ndvi_images:
        time_series = ee.Image.cat(ndvi_images)
    else:
        print("No valid NDVI data found for the specified period.")
        return

    # Print band names for verification
    print("Final band names:", time_series.bandNames().getInfo())

    # Export the time series
    file_name = f"Landsat_NDVI_time_series_{start_year}_to_{end_year}"
    file_path = export_time_series(time_series, roi, output_dir, file_name)


if __name__ == "__main__":
    # Initialize Earth Engine
    ee.Authenticate()
    ee.Initialize(project="ee-faranido")
    main()
