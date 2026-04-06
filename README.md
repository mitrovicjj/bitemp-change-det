# Deep Learning for Infrastructure Change Detection

A geospatial AI project for monitoring infrastructure changes over time using satellite imagery, OpenStreetMap data, and deep learning.

The project is focused on practical change monitoring use cases such as new construction, building expansion and urban land-use changes. It combines geospatial preprocessing, mask generation from OSM and bi-temporal image analysis to produce clean, interpretable change maps.

## What the project does

- Detects infrastructure changes between two satellite images of the same area.
- Generates binary change masks and visual overlays.
- Uses OSM building data as a geospatial reference signal.
- Validates spatial alignment through raster export and QGIS review.
- Builds toward a Siamese U-Net baseline on the OSCD benchmark.

## Why that's important

Satellite-based change detection is useful for infrastructure monitoring, urban development tracking, and land-use analysis. 
This project is designed to support those practical workflows by turning raw imagery into spatially aligned, human-readable change maps. 
Deep learning is helpful because satellite data is often noisy, partially occluded by clouds, affected by seasonal and lighting differences, and too large to inspect manually at scale, making learned models far better suited for robust change extraction than simple threshold-based methods.

## Current progress

- Sentinel-2 + OSM test pipeline completed and tested.
- Clean reproducible notebook prepared.
- OSM rasterization and change mask generation working.
- GeoTIFF export functional, QGIS validation ready.
- OSCD benchmark training with a Siamese U-Net model.

## Practical workflow

1. Load satellite imagery for two time points.
2. Extract building data from OpenStreetMap.
3. Rasterize vector geometries into aligned masks.
4. Compare temporal inputs and generate a change mask.
5. Export results for GIS validation and visual inspection.

## Example outputs

- Sentinel image, OSM building mask, and change mask comparison.
- Satellite overlay with semi-transparent change detection result.
- Exported GeoTIFF masks for GIS review.

## Tech stack

- PyTorch
- Rasterio
- GeoPandas
- OpenStreetMap / GeoFabrik
- Google Earth Engine
- QGIS