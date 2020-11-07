.SILENT:run-virtual run

setup:
	julia setup.jl

run-virtual:
	julia --project=./venv/Project.toml main.jl

run:
	julia main.jl