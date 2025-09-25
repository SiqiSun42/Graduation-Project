# run-dad.sh
export PYTHONPATH=$PWD

python main.py \
  --root_path /root/autodl-tmp/dataset/DAD \
  --mode train \
  --view top_depth \
  --model_type shufflenet \
  --width_mult 2.0 \
  --pre_train_model False \
  --n_train_batch_size 10 \
  --a_train_batch_size 150 \
  --val_batch_size 50 \
  --learning_rate 0.01 \
  --epochs 25 \
  --cal_vec_batch_size 50 \
  --tau 0.1 \
  --train_crop random \
  --n_scales 5 \
  --downsample 2 \
  --val_step 5 \
  --n_split_ratio 1.0 \
  --a_split_ratio 1.0 \

#--root_path
#/root/autodl-tmp/dataset/DAD
#--mode
#train
#--view
#front_depth
#--model_type
#mobilenet
#--width_mult
#2.0
#--pre_train_model
#False
#--n_train_batch_size
#10
#--a_train_batch_size
#150
#--val_batch_size
#5
#--learning_rate
#0.01
#--epochs
#25
#--cal_vec_batch_size
#50
#--tau
#0.1
#--train_crop
#random
#--n_scales
#5
#--downsample
#2
#--val_step
#5
#--n_split_ratio
#1.0
#--a_split_ratio
#1.0
