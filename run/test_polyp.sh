test_path=./dataset/Polyp/test/
result_map=./result_map/
model_pth=./checkpoint/DSNet_Polyp.pth
result_evaluate=./results/

model_name=DSNet
dataset=Polyp

test_polyp(){
python Test.py \
--test_path $test_path \
--model_name $model_name \
--dataset $dataset \
--result_map $result_map \
--model_pth $model_pth \
--testsize 352
}

evaluate_polyp(){
python Eval.py \
--model_name $model_name \
--dataset $dataset \
--gt_path $test_path \
--result_evaluate $result_evaluate \
--result_map $result_map
}

# generate result_map
test_polyp
# evaluate
evaluate_polyp