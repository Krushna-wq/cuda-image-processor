#!/bin/bash

# Exit on any error
set -e

echo "=== CUDA Batch Image Grayscale Processor ==="

# 1. Ensure input images exist
if [ ! -d "input" ] || [ -z "$(ls -A input)" ]; then
    echo "Generating procedural input images..."
    python3 generate_images.py
else
    echo "Input images already exist inside 'input/'."
fi

# 2. Build the project
echo "Building CUDA program..."
make clean
make

# 3. Create output directory if it doesn't exist
mkdir -p output

# 4. Run the image processor
echo "Running image processor..."
./image_processor --input input --output output --images 100

echo "=== Execution Complete ==="
