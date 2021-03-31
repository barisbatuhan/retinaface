using Pkg

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
Pkg.add("ArgParse")
Pkg.add("Plots")

Pkg.precompile()
