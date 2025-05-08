"""
Expand a local image file using the edge pixel technique
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def repeat_edge_pixels(img, target_width, target_height):
    """Create extended image by repeating edge pixels"""
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
        
        # Add gradient blend
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
        
        # Add gradient blend
        blend_width = min(32, target_width - right_edge)
        for i in range(blend_width):
            alpha = i / blend_width
            pixels[:, right_edge+i, :] = (1-alpha) * pixels[:, right_edge, :] + alpha * pixels[:, right_edge+i, :]
    
    # Fill in top side
    top_edge = (target_height - orig_height) // 2
    if top_edge > 0:
        edge_pixels = pixels[top_edge:top_edge+8, :, :]
        avg_edge = np.mean(edge_pixels, axis=0, keepdims=True)
        pixels[:top_edge, :, :] = np.repeat(avg_edge, top_edge, axis=0)
        
        # Add gradient blend
        blend_width = min(32, top_edge)
        for i in range(blend_width):
            alpha = i / blend_width
            pixels[top_edge-blend_width+i, :, :] = (1-alpha) * pixels[top_edge-blend_width+i, :, :] + alpha * pixels[top_edge, :, :]
    
    # Fill in bottom side
    bottom_edge = (target_height + orig_height) // 2
    if bottom_edge < target_height:
        edge_pixels = pixels[bottom_edge-8:bottom_edge, :, :]
        avg_edge = np.mean(edge_pixels, axis=0, keepdims=True)
        pixels[bottom_edge:, :, :] = np.repeat(avg_edge, target_height - bottom_edge, axis=0)
        
        # Add gradient blend
        blend_width = min(32, target_height - bottom_edge)
        for i in range(blend_width):
            alpha = i / blend_width
            pixels[bottom_edge+i, :, :] = (1-alpha) * pixels[bottom_edge+i, :, :] + alpha * pixels[bottom_edge, :, :]
    
    return Image.fromarray(pixels.astype(np.uint8))

def parse_aspect_ratio(aspect_str):
    """Parse aspect ratio from string like '16:9' or a decimal like '1.78'"""
    if isinstance(aspect_str, float):
        return aspect_str
    
    # Check if input contains a colon (e.g., "16:9")
    if ":" in aspect_str:
        width, height = aspect_str.split(":")
        return float(width) / float(height)
    else:
        return float(aspect_str)

def main():
    parser = argparse.ArgumentParser(description="Expand a local image file")
    parser.add_argument("input_file", type=str, help="Path to the input image file")
    parser.add_argument("--aspect", type=str, default="16:9", 
                      help="Target aspect ratio (width:height) or decimal value (width/height)")
    parser.add_argument("--orientation", type=str, default="landscape", choices=["landscape", "portrait"],
                      help="Image orientation (landscape or portrait)")
    parser.add_argument("--output", type=str, help="Path to save output image (default: 'expanded_' + input filename)")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return
    
    # Load image
    try:
        image = Image.open(args.input_file)
        # Convert to RGB mode if it's not already
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return
    
    # Determine output filename
    if args.output:
        output_filename = args.output
    else:
        base_name = os.path.basename(args.input_file)
        name, ext = os.path.splitext(base_name)
        output_filename = f"expanded_{name}{ext}"
    
    # Calculate target dimensions
    orig_width, orig_height = image.size
    
    # Make sure the image isn't too large for memory constraints
    if max(orig_width, orig_height) > 1200:
        scale_factor = 1200 / max(orig_width, orig_height)
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        print(f"Resizing large image from {orig_width}x{orig_height} to {new_width}x{new_height}")
        image = image.resize((new_width, new_height), Image.LANCZOS)
        orig_width, orig_height = image.size
    
    # Parse aspect ratio
    aspect_ratio = parse_aspect_ratio(args.aspect)
    
    is_portrait = args.orientation.lower() == "portrait"
    if is_portrait:
        # Portrait mode
        target_width = orig_width
        target_height = int(orig_width * (1.0 / aspect_ratio))
        print(f"Using portrait orientation ({1.0/aspect_ratio:.2f}:1 or 1:{aspect_ratio:.2f})")
    else:
        # Landscape mode
        target_width = int(orig_height * aspect_ratio)
        target_height = orig_height
        print(f"Using landscape orientation ({aspect_ratio:.2f}:1)")
    
    print(f"Original dimensions: {orig_width}x{orig_height}")
    print(f"Target dimensions: {target_width}x{target_height} (aspect ratio: {target_width/target_height:.2f})")
    
    # Expand image
    print("Expanding image...")
    expanded_image = repeat_edge_pixels(image, target_width, target_height)
    expanded_image.save(output_filename)
    
    # Create comparison filename
    comparison_filename = f"comparison_{os.path.basename(output_filename)}"
    
    # Display comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")
    
    ax2.imshow(expanded_image)
    ax2.set_title(f"Expanded Image ({target_width}x{target_height})")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig(comparison_filename)
    plt.show()
    
    print(f"Images saved to '{output_filename}' and '{comparison_filename}'")

if __name__ == "__main__":
    main() 