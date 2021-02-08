.SILENT=all

setup:
	julia setup.jl

train:
	julia train.jl --batch_size 2 --mode 1 --laterals 5 --start_epoch 1

continue:
	julia train.jl --batch_size 10 --mode 2 --laterals 3 --log_file ./logs/mode2_lat3_cont.txt --start_epoch 14 --load_dir ./weights/PyTorch_R50_mode2_laterals3.jld2
    
predict:
	julia predict.jl --mode 2 --laterals 3 --image_path "./data/images/test.jpg" --load_dir ./weights/PyTorch_R50_mode2_laterals3.jld2 --save_dir ./data/results/predicted.png
    
evaluate:
	julia extract_preds.jl --mode 2 --laterals 3 --load_dir ./weights/PyTorch_R50_mode2_laterals3.jld2 --eval_dir ./wider_evaluate/ground_truth/ --save_dir ./data/wider_txt_lat3/
#	python ./wider_evaluate/setup.py build_ext --inplace
	python3 ./wider_evaluate/evaluation.py -p ./data/wider_txt_lat3/ -g ./wider_evaluate/ground_truth/
