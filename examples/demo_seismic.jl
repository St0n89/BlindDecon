# Thin wrapper: julia examples/demo_seismic.jl [outdir]
include(joinpath(dirname(@__DIR__), "BlindDecon.jl"))
outdir = length(ARGS) >= 1 ? ARGS[1] : "demo_seismic_out"
demo_seismic(outdir)
