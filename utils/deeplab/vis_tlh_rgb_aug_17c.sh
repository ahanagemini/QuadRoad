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

#TLH_DATASET="/home/ahana/road_graph_extract_dataset/tfrecord/rgb_overlap"
#WORK_DIR="/home/ahana/tensorflow/tensorflow/models/research/deeplab"
#TRAIN_LOGDIR="/home/ahana/road_graph_extract_dataset/exp/train/rgb"
#EVAL_LOGDIR="/home/ahana/road_graph_extract_dataset/exp/train_eval/rgb"

#TLH_DATASET="/home/ahana/road_data/tfrecord/rgb"
#WORK_DIR="/home/ahana/tensorflow/tensorflow/models/research/deeplab"
#TRAIN_LOGDIR="/home/ahana/road_data/exp/train/rgb/saved_models"
#EVAL_LOGDIR="/home/ahana/road_data/exp/eval/rgb"

#TLH_DATASET="/home/ahana/road_data/tfrecord/new_tf/rgb"
#WORK_DIR="/home/ahana/tensorflow_new/models/research/deeplab"
#TRAIN_LOGDIR="/home/ahana/road_data/exp/train/tf_new/rgb_dice"
#EVAL_LOGDIR="/home/ahana/road_data/exp/eval/tf_new/rgb_dice"

#TLH_DATASET="/home/ahana/road_data/tfrecord/new_tf/rgb_aug"
#WORK_DIR="/home/ahana/tensorflow_new/models/research/deeplab"
#TRAIN_LOGDIR="/home/ahana/road_data/exp/train/tf_new/rgb_aug_dice"
#EVAL_LOGDIR="/home/ahana/road_data/exp/eval/tf_new/rgb_aug_dice"

TLH_DATASET="/home/ahana/road_data/tfrecord/new_tf/rgb_aug_17c"
WORK_DIR="/home/ahana/tensorflow_new/models/research/deeplab"
TRAIN_LOGDIR="/home/ahana/road_data/exp/train/tf_new/rgb_aug_17c_ce"
EVAL_LOGDIR="/home/ahana/road_data/exp/eval/tf_new/rgb_aug_17c_ce"

# Visualize the results.
python3 "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --purpose="eval" \
  --folder_name="pred_results" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=501 \
  --vis_crop_size=501 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${TLH_DATASET}" \
  --dataset="tlh_seg_rgb_17c" \
  --also_save_raw_predictions=true \
  --max_number_of_iterations=1

