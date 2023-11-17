test_path=./dataset/DSB/test/
result_map=./result_map/
model_pth=./model/DSNet_DSB.pth
result_evaluate=./results/

model_name=DSNet
dataset=DSB

test_dsb(){
python Test.py \
--test_path $test_path \
--model_name $model_name \
--dataset $dataset \
--result_map $result_map \
--model_pth $model_pth \
--testsize 352
}

evaluate_dsb(){
python Eval.py \
--model_name $model_name \
--dataset $dataset \
--gt_path $test_path \
--result_evaluate $result_evaluate \
--result_map $result_map
}

# generate result_map
test_dsb
# evaluate
evaluate_dsb
