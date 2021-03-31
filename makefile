.SILENT=all

setup:
	julia setup.jl

train:
	julia train.jl --batch_size 10 --mode 2 --laterals 5 --start_epoch 1

continue:
	julia train.jl --batch_size 10 --mode 2 --laterals 5 --log_file ./logs/nocascade_cont.txt --start_epoch 14 --load_dir ./weights/nocascade_final.jld2
    
predict:
	julia predict.jl --mode 2 --laterals 5 --image_path "./data/images/test.jpg" --load_dir ./weights/nocascade_final.jld2 --save_dir ./data/results/predicted.png
    
evaluate:
	julia extract_preds.jl --mode 2 --laterals 5 --load_dir ./weights/nocascade_final.jld2 --eval_dir ./wider_evaluate/ground_truth/ --save_dir ./data/nocascade_eval/
#	python ./wider_evaluate/setup.py build_ext --inplace
	python3 ./wider_evaluate/evaluation.py -p ./data/nocascade_eval/ -g ./wider_evaluate/ground_truth/
