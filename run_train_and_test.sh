#! /bin/bash
<< COMMENTOUT
This script is used for gpu cluster PC.
You might use bat script if you want to use on Windows PC.
COMMENTOUT

set -eux

if [ $# -lt 2 ]; then 
  echo "$0 <encode dims> <log index>" >&2
  exit -1
fi

_ENCODE_DIMS=$1
LOG_INDEX=$2

SEG_ROOT_DIR="/win/scallop/user/kabashima/tl-net"
LOG_ROOT="${SEG_ROOT_DIR}/logs"
#pushd ${SEG_ROOT_DIR}
PYTHONPATH=${SEG_ROOT_DIR}
SCRIPT_ROOT="${SEG_ROOT_DIR}"
DATASET_DIR="/win/scallop/user/kabashima/RibSegmentation/dataset/DRR"

PYTHON_CMD="python3"

K_FOLD_CONFIG_ROOT="${SEG_ROOT_DIR}/ini/k-fold-list-20_with_FLIP"
N_FOLD=4
N_FOLD_ID_START=0
N_FOLD_ID_END=3
N_TRIAL=0

TRAIN_CMD="${PYTHON_CMD} ${SCRIPT_ROOT}/train_tl_net.py"
TEST_CMD="${PYTHON_CMD} ${SCRIPT_ROOT}/test_tl_net.py"
GPU_ID=0
N_LAYERS=5
MAX_ITERATION=50000
LR_RATE=1e-5
ENCODE_DIMS=${_ENCODE_DIMS}
EXTRA_OPTIONS=""

for i in `seq ${N_FOLD_ID_START} ${N_FOLD_ID_END}`
do
  MODEL_DIR=""
  for stage in `seq 1 3`
  do
    LOG_ROOT_DIR_OPTION=""
    if [ -z "${MODEL_DIR}"]; then
      LOG_ROOT_DIR_OPTION="--log-root-dir ${LOG_ROOT} --log-index ${LOG_INDEX}"
    else
      LOG_ROOT_DIR_OPTION="--log-root-dir ${MODEL_DIR}"
    fi
    echo "Stage ${stage}-------------------------------------"
    cmd="${TRAIN_CMD} 
      --gpu ${GPU_ID} 
      --train-index ${K_FOLD_CONFIG_ROOT}/${N_FOLD}/id-list_trial-${N_TRIAL}_training-$i.txt 
      --valid-index  ${K_FOLD_CONFIG_ROOT}/${N_FOLD}/id-list_trial-${N_TRIAL}_test-$i.txt 
      --dataset-dir ${DATASET_DIR} 
      --batch-size 5 
      --n-max-train-iter ${MAX_ITERATION} 
        --n-max-valid-iter 200
      --lr-rate ${LR_RATE} 
      --n-layers ${N_LAYERS} 
      ${LOG_ROOT_DIR_OPTION}
      --dropout-mode dropout
      --use-batch-norm on
      --use-skipping-connection add
      --encode-dims ${ENCODE_DIMS}
      --stage-index ${stage}
      ${EXTRA_OPTIONS}
    "
    
    echo ${cmd}

    for x in `${cmd}`
    do 
      MODEL_DIR=${x}
    done

    if [ -z "$MODEL_DIR" ]; then
      echo "Empty model directory"
      exit -1
    fi

    # Test inferrence
    cmd="${TEST_CMD} 
      --gpu ${GPU_ID} 
      --test-index ${K_FOLD_CONFIG_ROOT}/${N_FOLD}/id-list_trial-${N_TRIAL}_test-$i.txt 
      --dataset-dir ${DATASET_DIR} 
      --log-root-dir ${MODEL_DIR}
      --stage-index ${stage}
    "
    if [ "$stage" = "1" ]; then
      cmd="${cmd} --store-codes"
    fi
    echo $cmd
    $cmd
  done
done