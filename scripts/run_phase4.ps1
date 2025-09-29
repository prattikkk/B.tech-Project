param(
    [int]$NumSamples = 2000,
    [int]$CpuLimit = 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "python .\Phase_Four.py --benchmark --num_samples $NumSamples --cpulimit $CpuLimit"
