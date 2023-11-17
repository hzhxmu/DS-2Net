train_path=./dataset/Polyp/train/
test_path=./dataset/Polyp/test/
train_save=./model/
train_log=./train_log/

model_name=DSNet
dataset=Polyp

train_polyp(){
python Train.py \
--train_path $train_path \
--test_path $test_path \
--model_name $model_name \
--train_log $train_log \
--train_save $train_save \
--dataset $dataset \
--epoch 150 \
--lr 1e-5 \
--optimizer AdamW \
--size_rates 0.5 1 1.5 \
--batchsize 8 \
--trainsize 352
}

train_polyp
