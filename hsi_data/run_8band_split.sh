#!/bin/bash

for tl in $( cat needed_big_tiles );
do
  #echo $tl

  hsi_src="/media/biswas/My Passport/TLCGIS Data/DigitalGlobe2016/65218703_2015-2017_TLCGIS/SatelliteOrthoimagery/ortho_${tl}.tif"
  rgb_dir="/home/biswas/repositories/road_graph_contraction/data_rgb"
  dest_dir="/home/biswas/repositories/road_graph_contraction/data_hsi_split"

  #echo $hsi_src
  #ls -l "$hsi_src"
  # python geotiff_tools/geotiff_info.py data_rgb data_hsi_8_band/ortho_49922.tif data_hsi_split
  python geotiff_tools/geotiff_info.py "$rgb_dir" "$hsi_src"  "$dest_dir"

done
