import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from transformers import CLIPProcessor, FlaxCLIPModel
from functools import partial
from flax.training.common_utils import shard_prng_key
from flax.jax_utils import replicate
from tqdm.auto import trange

# DALL-E Mini modules
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel

# For reproducibility
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

class DALLE_Generator:
    def __init__(self, model_type="mega"):
        """Initialize DALL-E Mini generator
        
        Args:
            model_type (str): Model size - "mini", "mega", or "mega-full"
        """
        # Choose model
        if model_type == "mini":
            self.dalle_model = "dalle-mini/dalle-mini/mini-1:v0"
        elif model_type == "mega":
            self.dalle_model = "dalle-mini/dalle-mini/mega-1-fp16:latest"
        else:  # mega-full
            self.dalle_model = "dalle-mini/dalle-mini/mega-1:latest"
        
        self.vqgan_repo = "dalle-mini/vqgan_imagenet_f16_16384"
        self.vqgan_commit_id = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
        
        # Load models
        print(f"Loading DALL-E model: {self.dalle_model}")
        self.model, self.params = DalleBart.from_pretrained(
            self.dalle_model, revision=None, dtype=jnp.float16, _do_init=False
        )
        
        print("Loading VQGAN model")
        self.vqgan, self.vqgan_params = VQModel.from_pretrained(
            self.vqgan_repo, revision=self.vqgan_commit_id, _do_init=False
        )
        
        # Replicate parameters for multiple devices
        self.params = replicate(self.params)
        self.vqgan_params = replicate(self.vqgan_params)
        
        # Load processor
        self.processor = DalleBartProcessor.from_pretrained(self.dalle_model)
        
        # Helper functions for parallel processing
        self.p_generate = jax.pmap(
            self._generate, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6)
        )
        self.p_decode = jax.pmap(self._decode, axis_name="batch")
        
        print(f"Model loaded successfully. Using {jax.device_count()} devices.")
    
    def _generate(self, tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
        """Generate image tokens from text prompt"""
        return self.model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )
    
    def _decode(self, indices, params):
        """Decode image tokens to image"""
        return self.vqgan.decode_code(indices, params=params)
    
    def generate_images(self, prompt, n_predictions=4, gen_top_k=None, gen_top_p=None, 
                        temperature=None, cond_scale=10.0):
        """Generate images from text prompt
        
        Args:
            prompt (str): Text prompt
            n_predictions (int): Number of images to generate
            gen_top_k, gen_top_p, temperature, cond_scale: Generation parameters
            
        Returns:
            List of PIL Image objects
        """
        tokenized_prompts = self.processor([prompt])
        tokenized_prompt = replicate(tokenized_prompts)
        
        # Create a random key
        seed = random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)
        
        # Generate images
        images = []
        for i in trange(max(n_predictions // jax.device_count(), 1)):
            # Get a new key
            key, subkey = jax.random.split(key)
            
            # Generate images
            encoded_images = self.p_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                self.params,
                gen_top_k,
                gen_top_p,
                temperature,
                cond_scale,
            )
            
            # Remove BOS token
            encoded_images = encoded_images.sequences[..., 1:]
            
            # Decode images
            decoded_images = self.p_decode(encoded_images, self.vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            
            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)
        
        return images

class ImageExpander:
    def __init__(self):
        """Initialize the VQGAN+CLIP based image expander"""
        # Load CLIP model for guidance
        self.clip_model_name = "openai/clip-vit-base-patch32"
        print(f"Loading CLIP model: {self.clip_model_name}")
        self.clip = FlaxCLIPModel.from_pretrained(self.clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        
        # Load VQGAN model for image expansion
        self.vqgan_repo = "dalle-mini/vqgan_imagenet_f16_16384"
        self.vqgan_commit_id = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
        print("Loading VQGAN model for expansion")
        self.vqgan, self.vqgan_params = VQModel.from_pretrained(
            self.vqgan_repo, revision=self.vqgan_commit_id
        )
    
    def repeat_edge_pixels(self, img, target_width, target_height):
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
        
        # Fill in top side
        top_edge = (target_height - orig_height) // 2
        if top_edge > 0:
            edge_pixels = pixels[top_edge:top_edge+8, :, :]
            avg_edge = np.mean(edge_pixels, axis=0, keepdims=True)
            pixels[:top_edge, :, :] = np.repeat(avg_edge, top_edge, axis=0)
        
        # Fill in bottom side
        bottom_edge = (target_height + orig_height) // 2
        if bottom_edge < target_height:
            edge_pixels = pixels[bottom_edge-8:bottom_edge, :, :]
            avg_edge = np.mean(edge_pixels, axis=0, keepdims=True)
            pixels[bottom_edge:, :, :] = np.repeat(avg_edge, target_height - bottom_edge, axis=0)
        
        return Image.fromarray(pixels.astype(np.uint8))
    
    def expand_image(self, image, prompt, target_aspect_ratio, num_iterations=100):
        """Expand an image to a new aspect ratio using VQGAN+CLIP
        
        Args:
            image (PIL.Image): Input image
            prompt (str): Text prompt for guiding expansion
            target_aspect_ratio (float): Target aspect ratio (width/height)
            num_iterations (int): Number of optimization iterations
            
        Returns:
            PIL.Image: Expanded image
        """
        orig_width, orig_height = image.size
        
        # Calculate target dimensions while preserving height
        target_width = int(orig_height * target_aspect_ratio)
        target_height = orig_height
        
        # Create initial expanded image with repeated edge pixels
        expanded_img = self.repeat_edge_pixels(image, target_width, target_height)
        
        # Encode prompt with CLIP
        text_inputs = self.clip_processor(text=[prompt], return_tensors="np", padding=True)
        text_embeddings = self.clip.get_text_features(**text_inputs)
        
        # Encode image with VQGAN
        vqgan_input = tf.keras.applications.inception_v3.preprocess_input(
            np.array(expanded_img.resize((target_width, target_height)), dtype=np.float32)
        )
        z, indices = self.vqgan.encode(vqgan_input[None, ...])
        
        # Get original image center portion in latent space
        orig_vqgan_input = tf.keras.applications.inception_v3.preprocess_input(
            np.array(image.resize((orig_width, orig_height)), dtype=np.float32)
        )
        orig_z, orig_indices = self.vqgan.encode(orig_vqgan_input[None, ...])
        
        # Initialize optimizer
        z = tf.Variable(z)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        
        # Define the latent center region (to preserve)
        center_width = orig_z.shape[2]
        center_height = orig_z.shape[1]
        center_x = (z.shape[2] - center_width) // 2
        center_y = (z.shape[1] - center_height) // 2
        
        # Optimization loop
        for i in range(num_iterations):
            with tf.GradientTape() as tape:
                # Decode latent vector to image
                generated_img = self.vqgan.decode_code(z)
                
                # Process for CLIP
                clip_img = tf.image.resize(generated_img, (224, 224))
                clip_img = (clip_img + 1.0) / 2.0  # Normalize to [0,1]
                
                # Get image features
                image_inputs = self.clip_processor(
                    images=np.array(clip_img[0]), return_tensors="np", padding=True
                )
                image_embeddings = self.clip.get_image_features(**image_inputs)
                
                # Calculate loss (cosine similarity)
                similarity = tf.keras.losses.cosine_similarity(
                    text_embeddings, image_embeddings, axis=1
                )
                loss = tf.reduce_mean(similarity)
            
            # Get gradients and update
            gradients = tape.gradient(loss, [z])
            optimizer.apply_gradients(zip(gradients, [z]))
            
            # Preserve the original center
            z_value = z.numpy()
            z_value[:, center_y:center_y+center_height, center_x:center_x+center_width, :] = orig_z.numpy()
            z.assign(z_value)
            
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss.numpy()}")
        
        # Generate final image
        final_image = self.vqgan.decode_code(z)
        final_image = tf.clip_by_value(final_image, -1.0, 1.0)
        final_image = (final_image + 1.0) / 2.0 * 255.0
        final_image = final_image.numpy()[0].astype(np.uint8)
        
        return Image.fromarray(final_image)

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate and expand images with DALL-E and VQGAN+CLIP')
    parser.add_argument('--prompt', type=str, default='a painting of rolling farmland', 
                        help='Text prompt for image generation')
    parser.add_argument('--num_images', type=int, default=4, 
                        help='Number of images to generate')
    parser.add_argument('--aspect_ratio', type=float, default=16/9, 
                        help='Target aspect ratio for expansion (width/height)')
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Number of optimization iterations for expansion')
    parser.add_argument('--model_type', type=str, default='mega', 
                        choices=['mini', 'mega', 'mega-full'],
                        help='DALL-E model size')
    args = parser.parse_args()
    
    # Initialize generators
    dalle_generator = DALLE_Generator(model_type=args.model_type)
    image_expander = ImageExpander()
    
    # Generate images with DALL-E Mini
    print(f"Generating {args.num_images} images for prompt: '{args.prompt}'")
    images = dalle_generator.generate_images(args.prompt, n_predictions=args.num_images)
    
    # Display generated images
    plt.figure(figsize=(16, 4))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.title(f"Image {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('dalle_output.png')
    
    # Select first image for expansion
    selected_img = images[0]
    
    # Expand image to target aspect ratio
    print(f"Expanding image to aspect ratio {args.aspect_ratio}")
    
    # Create repeated edge pixels version
    orig_width, orig_height = selected_img.size
    target_width = int(orig_height * args.aspect_ratio)
    target_height = orig_height
    repeated_edge_img = image_expander.repeat_edge_pixels(selected_img, target_width, target_height)
    
    # Expand with VQGAN+CLIP
    expanded_img = image_expander.expand_image(
        selected_img, args.prompt, args.aspect_ratio, args.iterations
    )
    
    # Display comparison
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(selected_img)
    plt.title("Original Image (1:1)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(repeated_edge_img)
    plt.title(f"Repeated Edge Pixels ({args.aspect_ratio:.2f}:1)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(expanded_img)
    plt.title(f"E-DALL-E Expanded ({args.aspect_ratio:.2f}:1)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('expanded_output.png')
    print("Done! Results saved to 'dalle_output.png' and 'expanded_output.png'")

if __name__ == "__main__":
    main() 