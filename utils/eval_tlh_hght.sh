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

#TLH_DATASET="/data/biswas/tlhroad_data_deeplab/hght_lpu/tfrecord"
#WORK_DIR="/home/biswas/repositories/models_tensorflow/research/deeplab"
#TRAIN_LOGDIR="/data/biswas/deeplab_model/hght_lpu"
#EVAL_LOGDIR="/data/biswas/deeplab_model/hght_lpu/eval"

TLH_DATASET="/home/ahana/road_data/tfrecord/new_tf/hght"
WORK_DIR="/home/ahana/tensorflow_new/models/research/deeplab"
TRAIN_LOGDIR="/home/ahana/road_data/exp/train/tf_new/hght_dice"
EVAL_LOGDIR="/home/ahana/road_data/exp/eval/tf_new/hght_dice"

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
  --dataset="tlh_seg_hght" \
  --max_number_of_evaluations=1

