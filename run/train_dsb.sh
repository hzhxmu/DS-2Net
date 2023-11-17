train_path=./dataset/DSB/train/
test_path=./dataset/DSB/test/
train_save=./model/
train_log=./train_log/

model_name=DSNet
dataset=DSB

train_dsb(){
python Train.py \
--train_path $train_path \
--test_path $test_path \
--model_name $model_name \
--train_log $train_log \
--train_save $train_save \
--dataset $dataset \
--epoch 100 \
--lr 1e-4 \
--optimizer AdamW \
--size_rates 0.75 1 1.25 \
--batchsize 8 \
--trainsize 352
}

train_dsb
