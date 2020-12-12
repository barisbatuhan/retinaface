.SILENT:run-virtual run

setup:
	julia setup.jl

predict-virtual:
	julia --project=./venv/Project.toml predict.jl

predict:
	julia predict.jl

train-virtual:
	julia --project=./venv/Project.toml train.jl

train:
	julia train.jl