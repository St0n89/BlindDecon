# Thin wrapper: julia examples/demo_image.jl [bmp_in] [outdir]
include(joinpath(dirname(@__DIR__), "BlindDecon.jl"))
bmpin = length(ARGS) >= 1 ? ARGS[1] : ""
outdir = length(ARGS) >= 2 ? ARGS[2] : "demo_image_out"
demo_image(bmpin, outdir)
