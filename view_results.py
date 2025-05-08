"""
View all the expanded images in a visual report
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import glob

def main():
    # Find all PNG images in the current directory
    image_files = glob.glob("*.png")
    
    # Filter out the images
    comparison_images = sorted([f for f in image_files if f.startswith("comparison_")])
    
    # Get all expanded images but exclude ones used for visual galleries
    expanded_images = sorted([
        f for f in image_files 
        if (f.startswith("expanded_") or "ultrawide" in f or "vertical" in f or "cinematic" in f) 
        and not f in ["expanded_gallery.png", "comparison_gallery.png"]
    ])
    
    original_images = sorted([f for f in image_files if f.startswith("original_")])
    
    print(f"Found {len(comparison_images)} comparison images")
    print(f"Found {len(expanded_images)} expanded images")
    
    # Create figure with appropriate size for comparisons
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("E-DALL-E Image Expansion Comparisons", fontsize=16)
    
    # 1. Show some of the comparison images
    if comparison_images:
        print("\nComparison images:")
        for i, img_path in enumerate(comparison_images[:6]):  # Show up to 6 comparison images
            print(f" - {img_path}")
            try:
                img = Image.open(img_path)
                plt.subplot(3, 2, i+1)
                plt.imshow(img)
                plt.title(img_path)
                plt.axis('off')
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add some space for the suptitle
    plt.savefig("comparison_gallery.png")
    print("Saved comparison gallery to 'comparison_gallery.png'")
    
    # 2. Create a separate figure for expanded images gallery
    if expanded_images:
        fig2 = plt.figure(figsize=(15, 15))
        fig2.suptitle("Expanded Images Gallery", fontsize=16)
        
        num_images = len(expanded_images)
        cols = 2
        rows = (num_images + cols - 1) // cols  # Ceiling division
        
        print("\nExpanded images:")
        for i, img_path in enumerate(expanded_images):
            print(f" - {img_path}")
            try:
                img = Image.open(img_path)
                ax = plt.subplot(rows, cols, i+1)
                plt.imshow(img)
                
                # Get original dimensions
                width, height = img.size
                aspect_ratio = width / height
                
                plt.title(f"{img_path}\n{width}x{height} (Aspect: {aspect_ratio:.2f})")
                plt.axis('off')
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add some space for the suptitle
        plt.savefig("expanded_gallery.png")
        print("\nSaved expanded image gallery to 'expanded_gallery.png'")
    
    plt.show()

if __name__ == "__main__":
    main() 