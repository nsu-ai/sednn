#!/bin/bash

MINIDATA=0
if [ $MINIDATA -eq 1 ]; then
  WORKSPACE="workspace"
  mkdir $WORKSPACE
  TR_SPEECH_DIR="mini_data/train_speech"
  TR_NOISE_DIR="mini_data/train_noise"
  TE_SPEECH_DIR="mini_data/test_speech"
  TE_NOISE_DIR="mini_data/test_noise"
  echo "Using mini data. "
else
  WORKSPACE="./workspace"
  mkdir ${WORKSPACE}

  TRAIN_SPEECH_DIR="${WORKSPACE}/data/train/clean/"
  TRAIN_NOISE_DIR="${WORKSPACE}/data/train/noise/"
  TEST_SPEECH_DIR="${WORKSPACE}/data/test/clean/"
  TEST_NOISE_DIR="${WORKSPACE}/data/test/noise/"

  echo "Using full user data."
fi

# Create rules for mixing speech with noises.
python prepare_data.py create_mixing_rules --workspace=${WORKSPACE} --speech_dir=${TRAIN_SPEECH_DIR} --noise_dir=${TRAIN_NOISE_DIR} --data_type=train --magnification=2
python prepare_data.py create_mixing_rules --workspace=${WORKSPACE} --speech_dir=${TEST_SPEECH_DIR} --noise_dir=${TEST_NOISE_DIR} --data_type=test

# Calculate mixture features.
TRAIN_SNR=0
TEST_SNR=0
python prepare_data.py calculate_mixture_features --workspace=${WORKSPACE} --speech_dir=${TRAIN_SPEECH_DIR} --noise_dir=${TRAIN_NOISE_DIR} --data_type=train --snr=${TRAIN_SNR}
python prepare_data.py calculate_mixture_features --workspace=${WORKSPACE} --speech_dir=${TEST_SPEECH_DIR} --noise_dir=${TEST_NOISE_DIR} --data_type=test --snr=${TEST_SNR}

# Pack features. 
N_CONCAT=7
N_HOP=3
python prepare_data.py pack_features --workspace=${WORKSPACE} --data_type=train --snr=${TRAIN_SNR} --n_concat=${N_CONCAT} --n_hop=${N_HOP}
python prepare_data.py pack_features --workspace=${WORKSPACE} --data_type=test --snr=${TEST_SNR} --n_concat=${N_CONCAT} --n_hop=${N_HOP}

# Compute scaler. 
python prepare_data.py compute_scaler --workspace=${WORKSPACE} --data_type=train --snr=${TRAIN_SNR}

# Train. 
LEARNING_RATE=1e-4
CUDA_VISIBLE_DEVICES=3 python main_dnn.py train --workspace=${WORKSPACE} --tr_snr=${TRAIN_SNR} --te_snr=${TEST_SNR} --lr=${LEARNING_RATE}

# Plot training stat.
# python evaluate.py plot_training_stat --workspace=${WORKSPACE} --tr_snr=${TRAIN_SNR} --bgn_iter=0 --fin_iter=10001 --interval_iter=1000

# Inference, enhanced wavs will be created. 
ITERATION=10000
CUDA_VISIBLE_DEVICES=3 python main_dnn.py inference --workspace=${WORKSPACE} --tr_snr=${TRAIN_SNR} --te_snr=${TEST_SNR} --n_concat=${N_CONCAT} --iteration=${ITERATION}

# Calculate PESQ of all enhanced speech.
python evaluate.py calculate_pesq --workspace=${WORKSPACE} --speech_dir=${TEST_SPEECH_DIR} --te_snr=${TEST_SNR}

# Calculate overall stats. 
python evaluate.py get_stats
