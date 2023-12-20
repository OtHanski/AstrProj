# PowerShell script to compile and run astral.cu

# Compile astral.cu with nvcc
nvcc -o astral astral.cu

# Check if the compilation was successful
if ($?) {
    Write-Output "Running Astral..."
    # Run the compiled program
    ./astral
    Write-Output "Astral done"
} else {
    Write-Output "Compilation failed"
}