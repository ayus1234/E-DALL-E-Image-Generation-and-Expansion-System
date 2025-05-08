# E-DALL-E: Image Generation with Varying Aspect Ratios
# Based on the work by Robert A. Gonsalves

# 1. Install Dependencies
# Run these commands in separate cells in Colab:
# pip install -q tensorflow keras_cv matplotlib
# pip install -q git+https://github.com/borisdayma/dalle-mini.git
# pip install -q git+https://github.com/patil-suraj/vqgan-jax.git
# pip install -q ftfy unidecode emoji

# 2. Check GPU and Import Libraries
# Run in a separate cell:
# nvidia-smi -L

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

# Check devices
print(f"Number of available devices: {jax.local_device_count()}")

# 3. Select DALL-E Model
# Choose a model (you can change this)
DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

# Load DALL-E model
model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=None, dtype=jnp.float16, _do_init=False
)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)

# Replicate params to use multiple devices
params = replicate(params)
vqgan_params = replicate(vqgan_params)

# Create parallel functions for generation
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )

# Decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

# Load processor
processor = DalleBartProcessor.from_pretrained(DALLE_MODEL)

# 4. Generate Images with DALL-E Mini
# Set your text prompt
prompt = "a painting of rolling farmland"  # Change this to your desired prompt
n_predictions = 4  # Number of images to generate

# Generation parameters
gen_top_k = None
gen_top_p = None
temperature = None
cond_scale = 10.0

# Tokenize prompt
tokenized_prompts = processor([prompt])
tokenized_prompt = replicate(tokenized_prompts)

# Create random key
seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)

# Generate images
images = []
for i in trange(max(n_predictions // jax.device_count(), 1)):
    # Get a new key
    key, subkey = jax.random.split(key)
    
    # Generate images
    encoded_images = p_generate(
        tokenized_prompt,
        shard_prng_key(subkey),
        params,
        gen_top_k,
        gen_top_p,
        temperature,
        cond_scale,
    )
    
    # Remove BOS token
    encoded_images = encoded_images.sequences[..., 1:]
    
    # Decode images
    decoded_images = p_decode(encoded_images, vqgan_params)
    decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
    
    for decoded_img in decoded_images:
        img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        images.append(img)

# Display generated images
plt.figure(figsize=(15, 15))
for i, img in enumerate(images):
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.title(f"Image {i+1}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# Choose an image for expansion (0-based index)
selected_image_idx = 0  # Change this to select a different image
selected_image = images[selected_image_idx]

plt.figure(figsize=(6, 6))
plt.imshow(selected_image)
plt.title("Selected image for expansion")
plt.axis("off")
plt.show()

# 5. Create Initial Expanded Image with Repeated Edge Pixels
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
    
    return Image.fromarray(pixels.astype(np.uint8))

# Set target aspect ratio
target_aspect_ratio = 16/9  # Change this to your desired aspect ratio

# Calculate target dimensions
orig_width, orig_height = selected_image.size
target_width = int(orig_height * target_aspect_ratio)
target_height = orig_height

# Create expanded image with repeated edge pixels
expanded_img = repeat_edge_pixels(selected_image, target_width, target_height)

# Display
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(selected_image)
plt.title("Original Image (1:1)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(expanded_img)
plt.title(f"Repeated Edge Pixels ({target_aspect_ratio:.1f}:1)")
plt.axis("off")
plt.tight_layout()
plt.show()

# 6. Load Models for Image Expansion (VQGAN+CLIP)
# Load CLIP model
clip_model_name = "openai/clip-vit-base-patch32"
clip = FlaxCLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# 7. Expand Image with VQGAN+CLIP
# Number of optimization iterations
num_iterations = 100  # Increase for better quality, decrease for faster results

# Encode prompt with CLIP
text_inputs = clip_processor(text=[prompt], return_tensors="np", padding=True)
text_embeddings = clip.get_text_features(**text_inputs)

# Encode expanded image with VQGAN
vqgan_input = tf.keras.applications.inception_v3.preprocess_input(
    np.array(expanded_img.resize((target_width, target_height)), dtype=np.float32)
)
z, indices = vqgan.encode(vqgan_input[None, ...])

# Get original image center portion in latent space
orig_vqgan_input = tf.keras.applications.inception_v3.preprocess_input(
    np.array(selected_image.resize((orig_width, orig_height)), dtype=np.float32)
)
orig_z, orig_indices = vqgan.encode(orig_vqgan_input[None, ...])

# Initialize optimizer
z = tf.Variable(z)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

# Define the latent center region (to preserve)
center_width = orig_z.shape[2]
center_height = orig_z.shape[1]
center_x = (z.shape[2] - center_width) // 2
center_y = (z.shape[1] - center_height) // 2

# Record loss values for plotting
losses = []

# Optimization loop
for i in range(num_iterations):
    with tf.GradientTape() as tape:
        # Decode latent vector to image
        generated_img = vqgan.decode_code(z)
        
        # Process for CLIP
        clip_img = tf.image.resize(generated_img, (224, 224))
        clip_img = (clip_img + 1.0) / 2.0  # Normalize to [0,1]
        
        # Get image features
        image_inputs = clip_processor(
            images=np.array(clip_img[0]), return_tensors="np", padding=True
        )
        image_embeddings = clip.get_image_features(**image_inputs)
        
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
    
    # Record loss
    losses.append(loss.numpy())
    
    if i % 10 == 0 or i == num_iterations - 1:
        print(f"Iteration {i}, Loss: {loss.numpy()}")
        
        # Generate intermediate result for display
        if i % 20 == 0 or i == num_iterations - 1:
            curr_img = vqgan.decode_code(z)
            curr_img = tf.clip_by_value(curr_img, -1.0, 1.0)
            curr_img = (curr_img + 1.0) / 2.0 * 255.0
            curr_img = curr_img.numpy()[0].astype(np.uint8)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(Image.fromarray(curr_img))
            plt.title(f"Iteration {i}")
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.plot(losses)
            plt.title("Loss Curve")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.show()

# Generate final expanded image
final_image = vqgan.decode_code(z)
final_image = tf.clip_by_value(final_image, -1.0, 1.0)
final_image = (final_image + 1.0) / 2.0 * 255.0
final_image = final_image.numpy()[0].astype(np.uint8)
expanded_final = Image.fromarray(final_image)

# 8. Compare Results
# Display final results
plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.imshow(selected_image)
plt.title("Original Image (1:1)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(expanded_img)
plt.title(f"Repeated Edge Pixels ({target_aspect_ratio:.1f}:1)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(expanded_final)
plt.title(f"E-DALL-E Expanded ({target_aspect_ratio:.1f}:1)")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save images
selected_image.save("original_image.png")
expanded_img.save("repeated_edge_pixels.png")
expanded_final.save("expanded_final.png")

print("\nImages saved to:")
print("  - original_image.png")
print("  - repeated_edge_pixels.png")
print("  - expanded_final.png") 