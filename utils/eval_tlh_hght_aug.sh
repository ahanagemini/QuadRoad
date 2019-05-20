# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
#cd ..

# Update PYTHONPATH.
if [ -z $PYTHONPATH ]
then
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
fi
#TLH_DATASET="/home/ahana/road_data/tfrecord/biswas/rgb"
#WORK_DIR="/home/ahana/tensorflow/tensorflow/models/research/deeplab"
#TRAIN_LOGDIR="/home/ahana/road_data/exp/train/biswas/rgb"

#TLH_DATASET="/home/ahana/road_data/tfrecord/new_tf/rgb"
#WORK_DIR="/home/ahana/tensorflow_new/models/research/deeplab"
#TRAIN_LOGDIR="/home/ahana/road_data/exp/train/tf_new/rgb_dice"
#EVAL_LOGDIR="/home/ahana/road_data/exp/eval/tf_new/rgb_dice"

TLH_DATASET="/home/ahana/road_data/tfrecord/new_tf/hght_aug"
WORK_DIR="/home/ahana/tensorflow_new/models/research/deeplab"
TRAIN_LOGDIR="/home/ahana/road_data/exp/train/tf_new/hght_aug_ce"
EVAL_LOGDIR="/home/ahana/road_data/exp/eval/tf_new/hght_aug_ce"

# Run evaluation. 
# This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
python3 "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=501 \
  --eval_crop_size=501 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${TLH_DATASET}" \
  --dataset="tlh_seg_rgb" \
  --max_number_of_evaluations=1

