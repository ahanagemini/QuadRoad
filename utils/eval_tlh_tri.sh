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
cd ..

# Update PYTHONPATH.
if [ -z $PYTHONPATH ]
then
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
fi

TLH_DATASET="/data/biswas/tlhroad_data_deeplab/tri/tfrecord"
WORK_DIR="/home/biswas/repositories/models_tensorflow/research/deeplab"
TRAIN_LOGDIR="/data/biswas/deeplab_model/tri"
EVAL_LOGDIR="/data/biswas/deeplab_model/tri/eval"

# Run evaluation. 
# This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=500 \
  --eval_crop_size=500 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${TLH_DATASET}" \
  --dataset="tlh_seg_tri" \
  --max_number_of_evaluations=1

