using Pkg

if isfile("./venv/Project.toml") && isfile("./venv/Manifest.toml")
    Pkg.activate("./venv/.")
else
    Pkg.activate("venv")
end

Pkg.add("CUDA")
Pkg.add("Knet")
Pkg.add("AutoGrad")
Pkg.add("Images")
Pkg.add("ImageMagick")
Pkg.add("ImageDraw")
Pkg.add("MAT")
Pkg.add("JLD2")
Pkg.add("JLD")
Pkg.add("FileIO")
Pkg.add("PyCall")
Pkg.add("IJulia")

Pkg.precompile()
