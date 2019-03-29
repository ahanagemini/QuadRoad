#!/bin/bash

for tl in $(cat $1 );
do
  #echo $tl

  hsi_src="/home/ahana/road_data/DigitalGlobe2016/65218703_2015-2017_TLCGIS/SatelliteOrthoimagery/ortho_${tl}"
  rgb_dir="/home/ahana/road_data/aerial_geotiff_500/"
  dest_dir="/home/ahana/road_data/data_hsi_split"

  #echo $hsi_src
  #ls -l "$hsi_src"
  # python geotiff_tools/geotiff_info.py data_rgb data_hsi_8_band/ortho_49922.tif data_hsi_split
  python3 clip_split_hsi_band.py "$rgb_dir" "$hsi_src"  "$dest_dir"

done
