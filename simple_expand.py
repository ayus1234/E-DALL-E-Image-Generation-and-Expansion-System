"""
Simple Image Expander using VQGAN+CLIP

This script takes an existing image and expands it to a new aspect ratio using
the E-DALL-E technique (repeating edge pixels + VQGAN+CLIP optimization).
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

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
    
    # Fill in bottom side (if needed)
    bottom_edge = (target_height + orig_height) // 2
    if bottom_edge < target_height:
        edge_pixels = pixels[bottom_edge-8:bottom_edge, :, :]
        avg_edge = np.mean(edge_pixels, axis=0, keepdims=True)
        pixels[bottom_edge:, :, :] = np.repeat(avg_edge, target_height - bottom_edge, axis=0)
    
    return Image.fromarray(pixels.astype(np.uint8))

def expand_image(image_path, target_aspect_ratio, prompt=None, output_path=None, iterations=100):
    """Expand an image to a new aspect ratio
    
    Args:
        image_path (str): Path to input image
        target_aspect_ratio (float): Target aspect ratio (width/height)
        prompt (str, optional): Text prompt for CLIP guidance
        output_path (str, optional): Path to save output image
        iterations (int): Number of optimization iterations
        
    Returns:
        PIL.Image: Expanded image
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
    
    # If no prompt provided, use generic description
    if not prompt:
        prompt = "a beautiful image"
    
    # Determine if we should expand width or height
    current_aspect = orig_width / orig_height
    
    if target_aspect_ratio > current_aspect:
        # Expand width
        target_width = int(orig_height * target_aspect_ratio)
        target_height = orig_height
    else:
        # Expand height
        target_width = orig_width
        target_height = int(orig_width / target_aspect_ratio)
    
    print(f"Original dimensions: {orig_width}x{orig_height} (aspect ratio: {current_aspect:.2f})")
    print(f"Target dimensions: {target_width}x{target_height} (aspect ratio: {target_aspect_ratio:.2f})")
    
    # Create initial expanded image with repeated edge pixels
    expanded_img = repeat_edge_pixels(image, target_width, target_height)
    
    # Simple version without VQGAN+CLIP expansion
    if iterations <= 0:
        if output_path:
            expanded_img.save(output_path)
        return expanded_img
    
    try:
        # Load models for optimization
        print("Loading CLIP model...")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Encode prompt with CLIP
        print(f"Using prompt: '{prompt}'")
        text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
        text_embeddings = clip_model.get_text_features(**text_inputs)
        
        # Convert expanded image to tensor
        img_tensor = tf.convert_to_tensor(np.array(expanded_img) / 255.0, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, 0)
        img_tensor = tf.Variable(img_tensor)
        
        # Define optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        # Define the center region to preserve
        center_x = (target_width - orig_width) // 2
        center_y = (target_height - orig_height) // 2
        
        # Original image tensor
        orig_tensor = tf.convert_to_tensor(np.array(image) / 255.0, dtype=tf.float32)
        
        # Create mask (1 for regions to optimize, 0 for original image)
        mask = np.ones((target_height, target_width, 3), dtype=np.float32)
        mask[center_y:center_y+orig_height, center_x:center_x+orig_width, :] = 0
        mask_tensor = tf.convert_to_tensor(mask)
        
        # Optimization loop
        print(f"Starting optimization for {iterations} iterations...")
        for i in range(iterations):
            with tf.GradientTape() as tape:
                # Get current image
                current_img = img_tensor[0]
                
                # Prepare for CLIP
                clip_img = clip_processor(
                    images=np.array(current_img * 255, dtype=np.uint8), 
                    return_tensors="pt", 
                    padding=True
                )
                
                # Get image features
                image_embeddings = clip_model.get_image_features(**clip_img)
                
                # Calculate loss (negative cosine similarity)
                similarity = -tf.keras.losses.cosine_similarity(
                    text_embeddings.numpy(), image_embeddings.detach().numpy()
                )
                loss = tf.reduce_mean(similarity)
            
            # Get gradients and update
            gradients = tape.gradient(loss, [img_tensor])
            optimizer.apply_gradients(zip(gradients, [img_tensor]))
            
            # Preserve the original center
            updated = img_tensor.numpy()[0]
            updated[center_y:center_y+orig_height, center_x:center_x+orig_width, :] = orig_tensor.numpy()
            img_tensor.assign(tf.expand_dims(updated, 0))
            
            # Clip values to valid range
            img_tensor.assign(tf.clip_by_value(img_tensor, 0.0, 1.0))
            
            if i % 10 == 0:
                print(f"Iteration {i}/{iterations}, Loss: {loss.numpy():.4f}")
        
        # Convert final tensor to image
        final_img = Image.fromarray(np.array(img_tensor[0] * 255, dtype=np.uint8))
        
        if output_path:
            final_img.save(output_path)
            print(f"Saved expanded image to: {output_path}")
        
        return final_img
    
    except ImportError:
        print("CLIP optimization failed. Using edge-pixel version instead.")
        if output_path:
            expanded_img.save(output_path)
        return expanded_img

def main():
    parser = argparse.ArgumentParser(description="Expand an image to a different aspect ratio")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--aspect", type=float, default=16/9, help="Target aspect ratio (width/height)")
    parser.add_argument("--prompt", type=str, help="Text prompt for CLIP guidance")
    parser.add_argument("--output", type=str, help="Path to save output image")
    parser.add_argument("--iterations", type=int, default=100, help="Number of optimization iterations")
    parser.add_argument("--simple", action="store_true", help="Use simple edge-pixel expansion without optimization")
    
    args = parser.parse_args()
    
    iterations = 0 if args.simple else args.iterations
    
    # Load and expand the image
    expanded = expand_image(
        args.image_path, 
        args.aspect, 
        prompt=args.prompt, 
        output_path=args.output,
        iterations=iterations
    )
    
    # Display comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    image = Image.open(args.image_path).convert("RGB")
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")
    
    # Expanded image
    ax2.imshow(expanded)
    ax2.set_title(f"Expanded Image (Aspect Ratio: {args.aspect:.2f})")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 