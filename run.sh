#!/bin/bash

python hyperparameter_tuning.py \
  --data_path "datasets/emotion417k.csv" \
  --text_column "text" \
  --label_column "label" \
  --max_length 128 \
  --min_freq 2 \
  --cv_folds 3 \
  --cv_epochs 2 \
  --output_dir "./emotion_cv_results" \
  --seed 42 \
  --run_cnn \
  --run_rnn \
  --run_cnn_lstm \
  --batch_sizes 32 64 \
  --learning_rates 0.001 0.0005 0.0001 \
  --embedding_dims 100 300 \
  --dropout_rates 0.3 0.5 \
  --num_filters_list 100 200 \
  --filter_sizes_list "[3,4,5]" "[2,3,4]" \
  --hidden_dims 128 256 \
  --rnn_types "lstm" "gru" \
  --rnn_layers_list 1 2 3 \
  --bidirectional_list True False \
  --lstm_layers_list 1 2 3