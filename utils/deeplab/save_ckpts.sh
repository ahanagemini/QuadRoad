# Just a code that runs iteratively and copies deeplab models 
#to another directory so they are not deleted
while true
do
  echo $1 $2
  cp $1/*.* $2/
  cp $1/* $2
  sleep 1800
done
