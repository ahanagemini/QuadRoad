#!/bin/bash

# This script uses available data to create 
#the multi-spectral 8 band images for each tile
# Usage:
# run_8band_split.sh tile.txt hsi_src_dir rgb_dir dest_dir
# where tile.txt is list of all tile names
# hsi_src_dir is the ortho source files directory
#rgb_dir is rgb geotiff directory
#sest_dir is directory where we save the multi-spectral tiles

for tl in $(cat $1 );
do
  #echo $tl

  #hsi_src="/home/ahana/road_data/DigitalGlobe2016/65218703_2015-2017_TLCGIS/SatelliteOrthoimagery/ortho_${tl}"
  #rgb_dir="/home/ahana/road_data/aerial_geotiff_500/"
  #dest_dir="/home/ahana/road_data/data_hsi_split"

  hsi_src="$2/ortho_${tl}"
  rgb_dir=$3
  dest_dir=$4
  
  python geotiff_tools/geotiff_info.py data_rgb data_hsi_8_band/ortho_49922.tif data_hsi_split
  #python3 clip_split_hsi_band.py "$rgb_dir" "$hsi_src"  "$dest_dir"
done
