.SILENT=all

setup:
	julia setup.jl

train-full:
	julia train.jl --batch_size 10 --mode 1 --laterals 5 --log_file ./logs/mode1_fulltrain.txt --start_epoch 1

train-no-cascade:
	julia train.jl --batch_size 10 --mode 2 --laterals 5 --log_file ./logs/mode2_fulltrain.txt --start_epoch 1

train-baseline:
	julia train.jl --batch_size 10 --mode 0 --laterals 5 --log_file ./logs/baseline_fulltrain.txt --start_epoch 1

train-lateral3:
	julia train.jl --batch_size 10 --mode 2 --laterals 3 --log_file ./logs/mode2_lateral3_fulltrain.txt --start_epoch 1

continue-full:
	julia train.jl --batch_size 10 --mode 1 --laterals 5 --log_file ./logs/mode1_fulltrain_cont.txt --start_epoch 14 --load_dir ./weights/model_mode1_13.jld2
    
continue-no-cascade:
	julia train.jl --batch_size 10 --mode 2 --laterals 5 --log_file ./logs/mode2_fulltrain_cont.txt --start_epoch 14 --load_dir ./weights/model_mode2_26.jld2

continue-baseline:
	julia train.jl --batch_size 10 --mode 0 --laterals 5 --log_file ./logs/baseline_fulltrain_cont.txt --start_epoch 14 --load_dir ./weights/model_mode0_13.jld2

continue-lateral3:
	julia train.jl --batch_size 10 --mode 2 --laterals 3 --log_file ./logs/mode2_lateral3_fulltrain_cont.txt --start_epoch 14 --load_dir ./weights/PyTorch_R50_mode2_laterals3.jld2
    
predict:
	julia predict.jl --mode 1 --laterals 5 --image_path "./data/images/test.jpg" --load_dir ./weights/model_mode1_13.jld2 --save_dir ./data/results/predicted.png
    
evaluate:
	julia evaluate.jl --mode 1 --laterals 5 --load_dir ./weights/model_mode1_13.jld2 --eval_dir ./data/ --log_file ./logs/evaluate_log.txt
