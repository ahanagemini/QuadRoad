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
 50	 BUILDING
 63	 RUIN
 67	 SIDEWALK
 70	 PAVED-ROAD
 77	 PAVED-DRIVEWAY
 78	 UNPAVED-DRIVEWAY
 85	 PAVED-PARKING
 86	 UNPAVED-PARKING
 999	 WATERBODY

"""

from  osgeo import ogr, osr, gdal


def draw_paved_road_mask(tiff_img, imperv_shp):
    """
    Given a geotiff satellite image and it's corresponding impervious surface
    data as shapefiles, create a mask with road (pixel value 0).
    """
    *base, leaf = tiff_img.split("/")
    #if "sat" not in leaf:
    #    leaf = leaf.split(".")[0] + "_sat.tif"
    mask_img = "/".join(["/".join(base), "paved_rd_mask_" + leaf])    # "lasfiles/48837_7/paved_rd_mask_48837_7_sat.tif"

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
    out_raster_ds = memory_driver.Create(mask_img, ncol, nrow, 1, gdal.GDT_Byte)
    
    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)
    
    # Fill our output band with 255, no class label, value
    b = out_raster_ds.GetRasterBand(1)
    b.Fill(255)
    
    # Set road filter and rasterize
    road_filter = "DXF_LAYER = 'PAVED-ROAD'"
    layer.SetAttributeFilter(road_filter)
    status = burn_shp_layer_to_geotiff(out_raster_ds, layer, 0)

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
