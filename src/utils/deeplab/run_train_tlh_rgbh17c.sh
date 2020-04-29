# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

#Used to train on images with the following:
# R: sftmx of rgb aug, G: sftmx of hght aug, G: 17 class pred

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
#cd ..

# Update PYTHONPATH.
if [ -z $PYTHONPATH ]
then
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
fi

#TLH_DATASET="/data/biswas/tlhroad_data_deeplab/rgb/tfrecord"
#WORK_DIR="/home/biswas/repositories/models_tensorflow/research/deeplab"
#TRAIN_LOGDIR="/data/biswas/deeplab_model/rgb"

TLH_DATASET="/home/ahana/road_data/tfrecord/new_tf/rgbh17c_c"
WORK_DIR="/home/ahana/tensorflow_new/models/research/deeplab"
TRAIN_LOGDIR="/home/ahana/road_data/exp/train/tf_new/rgbh17c_c"

# Train 10 iterations.
NUM_ITERATIONS=50000
python3 "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=500 \
  --train_crop_size=500 \
  --train_loss="ce" \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${TLH_DATASET}" \
  --dataset="tlh_seg_rgbh17c" \
  --initialize_last_layer=false \
  --base_learning_rate=0.001 \
  --learning_rate_decay_step=20000 \
  --tf_initial_checkpoint="${WORK_DIR}/init_models/deeplabv3_pascal_train_aug/model.ckpt"
#  --tf_initial_checkpoint="${TRAIN_LOGDIR}/model.ckpt-4020" \

