.SILENT=all

setup:
	julia setup.jl

train:
	julia train.jl --batch_size 10 --mode 2 --laterals 3 --log_file ./logs/mode2_lat3_fulltrain.txt --start_epoch 1

continue:
	julia train.jl --batch_size 10 --mode 2 --laterals 3 --log_file ./logs/mode2_lat3_cont.txt --start_epoch 14 --load_dir ./weights/PyTorch_R50_mode2_laterals3.jld2
    
predict:
	julia predict.jl --mode 2 --laterals 3 --image_path "./data/images/test.jpg" --load_dir ./weights/PyTorch_R50_mode2_laterals3.jld2 --save_dir ./data/results/predicted.png
    
evaluate:
	julia evaluate.jl --mode 2 --laterals 3 --load_dir ./weights/PyTorch_R50_mode2_laterals3.jld2 --eval_dir ./data/ --log_file ./logs/evaluate_log.txt
