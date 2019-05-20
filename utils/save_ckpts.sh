
while true
do
  echo $1 $2
  cp $1/*.* $2/
  cp $1/* $2
  sleep 1800
done
