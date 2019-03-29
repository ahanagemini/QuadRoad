"""
Select impervious shapes which correspond to road regions
and burn them to an image. This will serve as the mask.

Helpful resources:
    http://ceholden.github.io/open-geo-tutorial/python/chapter_4_vector.html

    http://gdal.org/1.11/gdal_rasterize.html

Brief note on shapefiles related to Impervious Surfaces for Leon County, FL
===========================================================================

 1. Impervious Surface Shapefiles for a tile are in lasfiles/tile/tile_imperv.
    Shapefiles:
        BLDG_2015.shp, BREAKLINES_2015.shp, BRIDGE_2015.shp, CONTOURS_2015.shp,
        HYDRO_2015.shp, HYDRO_LN_2015.shp, IMPERV_2015.shp, IMPERV_HYDRO_2015.shp,
        LIDAR_LOW_CONFIDENCE_2015.shp, PROJECT_BOUNDARY_2015.shp, RDEDGE_2015.shp,
        SPOT_ELEVATION_2015.shp

 2. IMPERV_2015.shp holds almost all impervious surfaces.
    Its attribute table has an attribute "DXF_LAYER" which holds
    labels about the type of impervious region.
    Example labels:
        PAVED_ROAD, SIDEWALK, BUILDING, PAVED-DRIVEWAY, PAVED-PARKING,
        WATERBODY, PAVED-ISLAND, LANDSCAPE-ISLAND

 3. BRIDGE_2015.shp has polygons for bridges. Attribute "DXF_LAYER" has
    the value BRIDGE.

Notes on Road Mask creation
===========================

 1. Ignore:
    SIDEWALK, BRIDGE, any DRIVEWAY, any PARKING, any ISLAND.

 2. Non-road:
    BUILDING, any WATERBODY

 3. Road:
    PAVED_ROAD

Attribute TYPE for each DXF_LAYER
=================================
$ ogrinfo -al lasfiles/48837_7/48837_7_imperv/IMPERV_2015.shp > tmp
$ grep "TYPE (Integer)" tmp | cut -d"=" -f 2 > type_
$ grep "DXF_LAYER (" tmp | cut -d"=" -f 2 > dxf_
$ paste type_ dxf_ > type_dxf
$ uniq type_dxf
 100	 PAVED-ISLAND
 101	 PAVED-ROAD
 102	 SIDEWALK
 50	 BUILDING
 51	 UNFINISHED-BUILDING
 63	 RUIN
 67	 SIDEWALK
 75	 UNPAVED-ROAD
 77	 PAVED-DRIVEWAY
 78	 UNPAVED-DRIVEWAY
 80	 AIRPORT
 85	 PAVED-PARKING
 86	 UNPAVED-PARKING
 87	 LANDSCAPE-ISLAND
 999	 WATERBODY
 99	 TENNIS-COURT

"""

import os
from  osgeo import ogr, osr, gdal


def draw_road_mask(tiff_img, imperv_shp, *ignore_shps):
    """
    Given a geotiff satellite image and it's corresponding impervious surface
    data as shapefiles, create a mask with road (pixel value 0) regions and
    other classes like buildings, driveway and other regions. 
    """
    leaf = os.path.basename(tiff_img)
    dirname = os.path.dirname(tiff_img)

    # "lasfiles/48837_7/mask_48837_7_sat.tif"
    mask_f = os.path.join(dirname, f"mask_{leaf}")

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(imperv_shp)

    if not dataset:
        print("Error opening shapefile")
        exit(0)
    
    layer = dataset.GetLayer()
    if not layer:
        print("layer not loaded")
        exit(0)
    
    # First we will open our raster image, to understand how we will want to rasterize our vector
    raster_ds = gdal.Open(tiff_img, gdal.GA_ReadOnly)
    
    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize
    
    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()
    
    raster_ds = None
    
    # Create the raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(mask_f, ncol, nrow, 1, gdal.GDT_Byte)
    
    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)
    
    
    # Set ignore filter for the layer.
    # Filter string should be in the format of an SQL WHERE clause.
    mask_attrs = ['PAVED-ROAD', 'PAVED-ISLAND', 'SIDEWALK', 'BUILDING',
       'UNFINISHED-BUILDING', 'RUIN', 'SIDEWALK', 'UNPAVED-ROAD',
       'PAVED-DRIVEWAY', 'UNPAVED-DRIVEWAY', 'AIRPORT', 'PAVED-PARKING',
       'UNPAVED-PARKING', 'LANDSCAPE-ISLAND', 'WATERBODY', 'TENNIS-COURT'
    ]

    keep_mask_attrs = mask_attrs
    #print(keep_mask_attrs)

    # Fill output band with a number denoting the background
    b = out_raster_ds.GetRasterBand(1)
    #b.Fill(len(keep_mask_attrs))
    b.Fill(0)

    # Give a number for each class
    for i, attr in enumerate(keep_mask_attrs):
        attr_filter = f"DXF_LAYER = '{attr}'"
        layer.SetAttributeFilter(attr_filter)
        status = burn_shp_layer_to_geotiff(out_raster_ds, layer, i+1)

    # Close dataset
    out_raster_ds = None


def burn_shp_layer_to_geotiff(out_tiff, layer, pix_val):
    status = gdal.RasterizeLayer(out_tiff,  # output to our new dataset
                                 [1],  # output to our new dataset's first band
                                 layer,  # rasterize this layer
                                 None, None,  # don't worry about transformations since we're in same projection
                                 [pix_val],  # burn value
                                 ['ALL_TOUCHED=TRUE'],  # rasterize all pixels touched by polygons
                                 )
    return status
