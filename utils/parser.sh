sed s/]/]\\n/g test >test1
intersection=`grep true_pos_1.0[[0-9]*] test1 | cut -d '[' -f2 | cut -d ']' -f1`
#intersection=`grep true_pos_1.0[[0-9]*] test1 | cut -d '[' -f2 | cut -d ']' -f1`
false_p=`grep false_pos_1.0[[0-9]*] test1 | cut -d '[' -f2 | cut -d ']' -f1`
false_n=`grep false_neg_1.0[[0-9]*] test1 | cut -d '[' -f2 | cut -d ']' -f1`
union=`expr $false_p + $false_n`
union=`expr $union + $intersection`
iou=$(($((10000*$intersection))/union))
echo $iou
