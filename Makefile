# Compiler
NVCC = nvcc

# Compiler Flags
NVCC_FLAGS = -O3 -arch=sm_75 -Wno-deprecated-gpu-targets

# Target Executable
TARGET = image_processor

# Source Files
SRC = src/image_processor.cu

# Default Target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)
