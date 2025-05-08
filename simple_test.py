"""
Simplified version of E-DALL-E for testing
This version only implements the edge pixel repeating technique
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def repeat_edge_pixels(img, target_width, target_height):
    """Create extended image by repeating edge pixels
    
    Args:
        img (PIL.Image): Input image
        target_width (int): Target width
        target_height (int): Target height
        
    Returns:
        PIL.Image: Extended image with repeated edge pixels
    """
    orig_width, orig_height = img.size
    extended = Image.new("RGB", (target_width, target_height))
    extended.paste(img, ((target_width - orig_width) // 2, (target_height - orig_height) // 2))
    
    # Get pixel data
    pixels = np.array(extended)
    
    # Fill in left side
    left_edge = (target_width - orig_width) // 2
    if left_edge > 0:
        edge_pixels = pixels[:, left_edge:left_edge+8, :]
        avg_edge = np.mean(edge_pixels, axis=1, keepdims=True)
        pixels[:, :left_edge, :] = np.repeat(avg_edge, left_edge, axis=1)
        
        # Add gradient blend (optional)
        blend_width = min(32, left_edge)
        for i in range(blend_width):
            alpha = i / blend_width
            pixels[:, left_edge-blend_width+i, :] = (1-alpha) * pixels[:, left_edge-blend_width+i, :] + alpha * pixels[:, left_edge, :]
    
    # Fill in right side
    right_edge = (target_width + orig_width) // 2
    if right_edge < target_width:
        edge_pixels = pixels[:, right_edge-8:right_edge, :]
        avg_edge = np.mean(edge_pixels, axis=1, keepdims=True)
        pixels[:, right_edge:, :] = np.repeat(avg_edge, target_width - right_edge, axis=1)
        
        # Add gradient blend (optional)
        blend_width = min(32, target_width - right_edge)
        for i in range(blend_width):
            alpha = i / blend_width
            pixels[:, right_edge+i, :] = (1-alpha) * pixels[:, right_edge, :] + alpha * pixels[:, right_edge+i, :]
    
    # Fill in top side (if needed)
    top_edge = (target_height - orig_height) // 2
    if top_edge > 0:
        edge_pixels = pixels[top_edge:top_edge+8, :, :]
        avg_edge = np.mean(edge_pixels, axis=0, keepdims=True)
        pixels[:top_edge, :, :] = np.repeat(avg_edge, top_edge, axis=0)
        
        # Add gradient blend (optional)
        blend_width = min(32, top_edge)
        for i in range(blend_width):
            alpha = i / blend_width
            pixels[top_edge-blend_width+i, :, :] = (1-alpha) * pixels[top_edge-blend_width+i, :, :] + alpha * pixels[top_edge, :, :]
    
    # Fill in bottom side (if needed)
    bottom_edge = (target_height + orig_height) // 2
    if bottom_edge < target_height:
        edge_pixels = pixels[bottom_edge-8:bottom_edge, :, :]
        avg_edge = np.mean(edge_pixels, axis=0, keepdims=True)
        pixels[bottom_edge:, :, :] = np.repeat(avg_edge, target_height - bottom_edge, axis=0)
        
        # Add gradient blend (optional)
        blend_width = min(32, target_height - bottom_edge)
        for i in range(blend_width):
            alpha = i / blend_width
            pixels[bottom_edge+i, :, :] = (1-alpha) * pixels[bottom_edge+i, :, :] + alpha * pixels[bottom_edge, :, :]
    
    return Image.fromarray(pixels.astype(np.uint8))

def create_test_image(size=256):
    """Create a test image with a colored grid pattern"""
    # Create a new blank image
    img = Image.new('RGB', (size, size), color=(240, 240, 240))
    pixels = np.array(img)
    
    # Draw grid lines
    grid_spacing = 32
    grid_color = (200, 200, 200)
    
    for i in range(0, size, grid_spacing):
        pixels[i, :, :] = grid_color
        pixels[:, i, :] = grid_color
    
    # Draw colored shapes
    # Red circle in the center
    center = size // 2
    radius = size // 4
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            if dist < radius:
                pixels[y, x] = (220, 50, 50)
    
    # Blue square in the corner
    square_size = size // 3
    for y in range(square_size):
        for x in range(square_size):
            pixels[y, x] = (50, 50, 220)
    
    # Green triangle
    for y in range(size):
        for x in range(size - square_size, size):
            if y > size - (x - (size - square_size)) * (size / square_size):
                pixels[y, x] = (50, 220, 50)
    
    return Image.fromarray(pixels)

def main():
    parser = argparse.ArgumentParser(description="Test edge pixel expansion")
    parser.add_argument("--aspect", type=float, default=16/9, help="Target aspect ratio (width/height)")
    parser.add_argument("--output", type=str, default="expanded_test.png", help="Path to save output image")
    parser.add_argument("--portrait", action="store_true", help="Use portrait orientation (height > width)")
    args = parser.parse_args()
    
    # Create test image
    print("Creating test image...")
    original_image = create_test_image(256)
    original_image.save("original_test.png")
    
    # Calculate target dimensions
    orig_width, orig_height = original_image.size
    
    if args.portrait:
        # Portrait mode - calculate height based on inverse of aspect ratio
        aspect_ratio = 1.0 / args.aspect
        target_width = orig_width
        target_height = int(orig_width / aspect_ratio)
        print(f"Using portrait orientation (1:{args.aspect:.2f})")
    else:
        # Landscape mode - calculate width based on aspect ratio
        aspect_ratio = args.aspect
        target_width = int(orig_height * aspect_ratio)
        target_height = orig_height
        print(f"Using landscape orientation ({args.aspect:.2f}:1)")
    
    print(f"Original dimensions: {orig_width}x{orig_height}")
    print(f"Target dimensions: {target_width}x{target_height} (aspect ratio: {target_width/target_height:.2f})")
    
    # Expand image
    print("Expanding image...")
    expanded_image = repeat_edge_pixels(original_image, target_width, target_height)
    expanded_image.save(args.output)
    
    # Display comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    ax1.axis("off")
    
    ax2.imshow(expanded_image)
    ax2.set_title(f"Expanded Image ({target_width}x{target_height})")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.show()
    
    print(f"Images saved to 'original_test.png', '{args.output}', and 'comparison.png'")

if __name__ == "__main__":
    main() 