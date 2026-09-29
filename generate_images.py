import os
import numpy as np

def generate_ppm_image(filepath, width, height, image_index):
    # Procedural generation: create different color gradients/patterns depending on the index
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    
    # Color channels vary based on image index to ensure unique patterns
    r = np.uint8((xx + image_index * 2.5) % 256)
    g = np.uint8((yy - image_index * 2.5) % 256)
    b = np.uint8((xx * 0.5 + yy * 0.5 + image_index * 5) % 256)
    
    # Combine channels to shape (height, width, 3)
    img_data = np.stack([r, g, b], axis=-1)
    
    # Write binary PPM format (P6)
    with open(filepath, 'wb') as f:
        # Header: P6, Width, Height, Maxval (255)
        f.write(f"P6\n{width} {height}\n255\n".encode('ascii'))
        f.write(img_data.tobytes())

def main():
    output_dir = "cuda-image-processor/input"
    num_images = 100
    width, height = 512, 512
    
    print(f"Generating {num_images} images inside '{output_dir}'...")
    for i in range(1, num_images + 1):
        filepath = os.path.join(output_dir, f"image_{i:03d}.ppm")
        generate_ppm_image(filepath, width, height, i)
    print("Successfully generated 100 PPM images.")

if __name__ == '__main__':
    main()
