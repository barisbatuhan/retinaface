using Pkg

if isfile("./venv/Project.toml") && isfile("./venv/Manifest.toml")
    Pkg.activate("./venv/.")
else
    Pkg.activate("venv")
end

Pkg.add("CUDA")
Pkg.add("Knet")

