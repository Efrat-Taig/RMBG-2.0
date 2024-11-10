
"""
This script generates images based on a list of text prompts using the BRIA 2.3-FAST model.

Steps:
1. Initializes the BRIA 2.3-FAST model for image generation with optimized settings.
2. Iterates through each prompt in the provided list.
3. For each prompt, generates a specified number of images with unique random seeds for diversity.
4. Saves each generated image in the specified output directory, with filenames based on the prompt and random seed.

Dependencies:
- diffusers
- torch
- PIL (Pillow)
- transformers

Usage:
Define the `output_path` for saving images and a list of `prompts`. Run the script to generate and save images based on each prompt.
"""

from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
import torch
import random
import os
import re



def sanitize_filename(prompt):
    # Convert prompt to a valid filename by keeping only alphanumeric characters and spaces
    # Replace spaces with underscores and limit the length
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', prompt)
    sanitized = sanitized.replace(' ', '_')
    return sanitized[:50]  # Limit filename length to 50 characters

def generate_images(output_path, prompts, num_images_per_prompt=3):
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Initialize the model
    unet = UNet2DConditionModel.from_pretrained("briaai/BRIA-2.3-FAST", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained("briaai/BRIA-2.3-BETA", unet=unet, torch_dtype=torch.float16)
    pipe.force_zeros_for_empty_prompt = False
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")  # Ensure the model is moved to GPU
    
    for idx, prompt in enumerate(prompts):
        for i in range(num_images_per_prompt):
            # Generate a random seed
            random_seed = random.randint(1, 1000000)
            torch.manual_seed(random_seed)
            
            # Generate image
            image = pipe(prompt, num_inference_steps=8, guidance_scale=1.0).images[0]
            
            # Create filename from prompt and seed
            filename = f"{idx+1}_{sanitize_filename(prompt)}_seed_{random_seed}.png"
            image_path = os.path.join(output_path, filename)
            image.save(image_path)
            
            print(f"Generated image for prompt {idx+1}: '{prompt}' with random seed {random_seed}\nSaved as: {filename}\n")

if __name__ == "__main__":
    output_path = "/home/ubuntu/spring/misc/efrat/rmbg/gen_ai_benchmark"  # Directory where images will be saved
    # Define prompts

    prompts = [
    'A close-up of a medieval knights helmet with intricate carvings and golden accents, sunlight reflecting off its polished surface. The background is a dark, stone-walled room with subtle torches illuminating the edges.',
    'A futuristic robotic hand reaching out to touch a glowing holographic butterfly, with neon colors and fine mechanical details. The style is cyberpunk, with a highly detailed metallic texture on the hand.',
    'A magical crystal ball floating in mid-air, surrounded by swirling mist and enchanted runes. The image has a mystical, dark fantasy feel, with glowing purple and blue hues radiating from the ball.',
    'An intricately carved wooden mask with tribal patterns and feathers attached, illuminated with warm, golden light. The background is dark, giving a feeling of depth and mystery.',
    'A vintage typewriter on an old wooden desk, covered with dust, a single piece of paper in it that has the word "Chapter One" written in classic typeface. The scene is lit with soft, sepia-toned lighting for a nostalgic feel.',
    'A whimsical tea kettle with a face and tiny, animated arms, pouring tea into a floating cup. The style is cartoonish, with bright colors and exaggerated features, evoking a sense of playful fantasy.',
    'A surrealistic floating clock with melting edges in a desert setting. The clock hands are twisted and turning, with a backdrop of a cloudy sky that shifts into abstract shapes. The style is inspired by surrealism.',
    'A beautifully crafted violin with delicate carvings of flowers along its body, lying on a silk fabric. The style is hyper-realistic, with attention to the wood grain, and a warm light highlighting its curves.',
    'A Victorian-style birdcage with intricate metalwork, containing a glowing, mystical orb instead of a bird. The image is detailed with antique metal textures and faint beams of light escaping from within the cage.',
    'A neon-colored, retro arcade joystick with bright, pixelated patterns and an 80s vibe. The joystick glows, with reflections of neon lights in the background that resemble an old-school gaming arcade.',
    'A vibrant, expressive portrait of a young woman with colorful face paint and wild, curly hair. Her eyes are closed, and she is surrounded by abstract shapes and brush strokes in a watercolor style.',
    'A radiant mermaid with scales that shimmer in shades of green and blue, holding a glowing pearl in her hands. The underwater scene is bathed in ethereal light, and fish swim around her, adding a magical feel.',
    'A bustling 1920s jazz club scene with elegantly dressed people dancing and enjoying live music. The style is vintage, with a focus on the intricate details of fashion and decor from the roaring twenties.',
    'A cheerful baker in a pastel-colored kitchen, holding a tray of colorful cupcakes. The scene has a playful, cartoonish vibe, with exaggerated features and a warm, inviting atmosphere.',
    'A heroic astronaut standing on a distant planet, looking at a massive ringed planet in the sky. The style is cinematic, with rich colors and dramatic lighting highlighting the astronauts suit and the alien landscape.',
    'A close-up of a young child blowing bubbles, with a vibrant rainbow reflecting off the bubbles. The style is photorealistic, capturing the innocence and joy on the childs face with beautiful lighting.',
    'A traditional Japanese tea ceremony setting with a geisha in a colorful kimono, holding a teacup. The style is serene and detailed, with a soft focus on the intricate patterns of the kimono and tea set.',
    'A stylish street artist spray-painting a mural on a city wall. The scene is vibrant, with splashes of color and a hint of urban grit, capturing the energy and creativity of street art.',
    'A futuristic samurai with a glowing katana and neon armor, standing in a rain-soaked cityscape at night. The cyberpunk style is dark and dramatic, with reflections of neon lights on the wet pavement.',
    'A majestic lion with a mane of flowers and butterflies resting in a lush, enchanted forest. The scene is whimsical and colorful, with rich details that give a magical, fairytale-like quality to the lion and its surroundings.',
    'A powerful wizard casting a spell in a dark forest, with glowing blue energy swirling around his hands. The style is fantasy, with dramatic shadows and vibrant magical effects lighting up the scene.',
    'A close-up of an ancient Egyptian pharaohs golden mask, with rich turquoise and lapis lazuli inlays. The style is historical, focusing on intricate details of the mask and its polished gold surface.',
    'A young ballerina practicing in an empty dance studio, with sunlight streaming through the windows. The scene is peaceful and detailed, capturing the grace and determination of the dancer.',
    'A steampunk-inspired airship floating above a Victorian cityscape, with gears and pipes protruding from the sides. The style is retro-futuristic, with a focus on metallic textures and warm tones.',
    'A wise owl wearing reading glasses and perched on a stack of books, in a cozy, candle-lit library. The style is whimsical and warm, with rich, textured details on the feathers and books.',
    'A pirate captain holding a glowing treasure map, with his crew around him on a ship deck. The scene is adventurous and intense, with dramatic lighting and detailed pirate attire.',
    'A medieval blacksmith forging a sword in a dimly lit workshop, sparks flying as he hammers the blade. The style is realistic, capturing the rugged textures of the workshop and the strength of the blacksmith.',
    'A futuristic scientist examining a holographic display of a human genome sequence, with a high-tech lab background. The style is sci-fi, with a focus on neon lighting and digital displays.',
    'A 1950s diner scene with a waitress serving a milkshake, a jukebox playing, and people in retro outfits. The style is vintage and colorful, capturing the nostalgic vibe of a classic diner.',
    'A mystical phoenix rising from ashes, with fiery wings spreading wide, casting an intense glow. The style is mythological, with vibrant reds, oranges, and yellows illuminating the scene.',
    'A retro-futuristic robot vacuuming a mid-century modern living room. The style is a mix of 1950s aesthetics with sci-fi elements, capturing the humor and quirkiness of a robot doing chores.',
    'An intricate mandala made of precious stones and flowers, with vibrant colors and fine details. The style is spiritual, evoking peace and harmony through symmetrical patterns.',
    'A jester juggling colorful orbs in a grand medieval court. The style is bright and playful, with detailed fabric textures on the jesters outfit and motion captured in the orbs.',
    'A young artist painting a large canvas in a sunlit studio, with splashes of paint around and a dreamy, focused expression. The style is impressionistic, capturing light and emotion.',
    'A mysterious detective in a trench coat and fedora, standing under a streetlight in a foggy, noir-style city. The style is cinematic, with dramatic shadows and a sense of suspense.',
    'A fantasy princess wearing a crown made of flowers, sitting in a meadow surrounded by glowing fireflies. The scene is ethereal, with soft, pastel colors creating a dream-like atmosphere.',
    'A samurai in traditional armor meditating under a cherry blossom tree, petals falling around. The style is serene and detailed, with a soft focus on the petals and a peaceful ambiance.',
    'A robot with a heart-shaped face screen displaying emotions, sitting alone in a busy city square at night. The style is cyberpunk, with neon lights and a sense of solitude.',
    'A celestial angel with glowing wings, holding a staff and floating among clouds in a bright, heavenly light. The style is divine, with radiant golds and whites highlighting the figure.',
    'A musician playing a saxophone on a rainy street corner, with reflections of city lights on the wet pavement. The style is moody and atmospheric, capturing the soul of jazz.',
    'A cartoonish dragon sitting atop a hoard of gold coins, with a playful, mischievous expression. The scene is colorful and fun, appealing to fantasy and adventure themes.',
    'A neon-lit barbershop with a barber cutting a customers hair, the scene captured from outside through a rain-streaked window. The style is cinematic, with reflections of neon lights.',
    'A young witch stirring a glowing potion in a cauldron, surrounded by spell books and mystical artifacts. The style is dark fantasy, with rich details on magical objects and a mysterious glow.',
    'A dapper fox wearing a bow tie and glasses, holding a walking cane in a Victorian-style portrait. The style is whimsical and humorous, with a soft, classical background.',
    'A traditional blacksmith forging a weapon in a medieval village, surrounded by tools and a warm fire glow. The style is highly detailed, capturing the rustic feel of a medieval workshop.',
    'A king chess piece towering over other pieces on a chessboard, with a dramatic perspective and dark shadows. The style is intense, emphasizing the power and dominance of the king.',
    'A fashion model posing in a futuristic, metallic outfit with bold, angular shapes, in front of a minimalist backdrop. The style is avant-garde, with sharp lighting and a modern aesthetic.',
    'A rustic farmer holding a basket of fresh vegetables, standing in front of a farmhouse. The style is realistic, capturing the simplicity and authenticity of farm life.',
    'A retro computer setup with a CRT monitor, floppy disks, and a mechanical keyboard, lit by a soft, nostalgic glow. The scene is 1980s-inspired, with vintage details.',
    'A powerful eagle in mid-flight with its wings spread wide, against a stormy sky. The style is majestic, with sharp focus on the feathers'
    'A noble wolf standing on a cliff under the full moon, howling, with its fur glowing softly in the moonlight. The scene is wild and majestic, with a detailed night sky filled with stars.',
    'A mysterious alchemist in a dark laboratory, surrounded by colorful, bubbling potions and ancient books. The style is fantasy, with a rich, warm glow highlighting the alchemists concentrated expression.',
    'A futuristic skyscraper covered in greenery and plants, in a city where nature has overtaken technology. The style is utopian, with bright, fresh colors and an emphasis on harmony between nature and modern architecture.',
    'A magical forest with giant, colorful mushrooms and glowing plants. A small fairy flies between them, casting a delicate light over the scene. The style is whimsical, with rich, saturated colors.',
    'A young girl holding a red balloon, walking through an abandoned amusement park at sunset. The scene is eerie yet nostalgic, with a soft glow illuminating the broken rides and faded colors.',
    'An intricate stained-glass window depicting a mythical scene of dragons and knights. The style is rich and vibrant, with light streaming through each colored panel, casting beautiful reflections.',
    'A high-tech robot chef preparing a meal in a futuristic kitchen, with tools and ingredients floating in the air. The style is clean and polished, with a playful touch of sci-fi elements.',
    'A giant octopus emerging from the ocean near a small boat, its tentacles curling dramatically. The style is thrilling and intense, capturing the power and mystery of the sea creature.',
    'A close-up of a detailed golden pocket watch with delicate engravings, showing the passing of time with a dramatic shadow. The style is antique and timeless, focusing on intricate details.',
    'A serene desert landscape with sand dunes and a single camel silhouetted against the setting sun. The style is minimalist, with warm hues and an emphasis on simplicity and tranquility.',
    'A sorceress summoning a spirit in a stone circle, with energy swirling around her. The scene is intense and mysterious, with magical symbols glowing in the air.',
    'A bustling market in an ancient Middle Eastern city, filled with vibrant fabrics, spices, and people from various cultures. The style is warm and detailed, capturing the lively energy of the scene.',
    'A snowy owl flying through a winter forest, its feathers blending with the snow-covered trees. The style is natural and serene, emphasizing the beauty of winter and wildlife.',
    'A young scientist conducting an experiment in a high-tech lab, surrounded by holographic data screens. The style is futuristic, with neon lighting and a focus on technology.',
    'A whimsical floating island in the sky, with a small cottage, a garden, and waterfalls spilling over the edge. The style is fantasy, with bright colors and a peaceful, dream-like atmosphere.',
    'A group of explorers discovering a hidden, ancient temple in the jungle, with vines and trees partially covering the entrance. The style is adventurous, with rich greenery and an air of mystery.',
    'A classical pianist performing on a grand stage, with a spotlight illuminating the keys as they play. The style is dramatic, with a focus on the pianists intensity and passion.',
    'A playful cat chasing a butterfly in a garden, with flowers and sunlight casting a soft glow. The style is cheerful and lively, capturing the joy of nature and movement.',
    'A king and queen in elaborate royal attire, standing on a balcony overlooking their kingdom. The style is regal, with intricate details on the clothing and a majestic atmosphere.',
    'A lone warrior standing at the edge of a cliff, overlooking a vast, misty valley with ruins scattered below. The scene is mystical, with dramatic lighting and a sense of solitude and bravery.',
    'An illustrated scene of a young wizard studying at a candle-lit desk, with stacks of magical books, potions, and a curious cat watching from the side. The style is whimsical and detailed, capturing the charm of a magical study.',
    'A colorful illustration of a vintage hot air balloon soaring above a quaint village, with small houses and trees below. The style is cheerful and nostalgic, with vibrant colors and charming details.',
    'An illustration of a young boy discovering a treasure chest in an enchanted forest, with glowing fireflies and mystical symbols around. The scene is magical, with soft, glowing colors and intricate details.',
    'A playful illustration of a friendly dinosaur in a tropical jungle, surrounded by colorful plants and flowers. The style is cartoonish and vibrant, capturing a joyful and adventurous feeling.',
    'An illustrated steampunk clock tower with gears and cogs visible, standing tall in the middle of a bustling city square. The style is vintage and detailed, with a touch of fantasy.',
    'A cozy illustration of a family gathered around a fireplace, with warm tones and soft lighting capturing a peaceful evening indoors. The style is homely and comforting, with gentle brushstrokes and a nostalgic feel.',
    'An illustrated bookshelf filled with magical items like potions, scrolls, and a crystal ball, with vines and flowers growing around. The style is detailed and fantastical, creating a sense of mystery.',
    'A detailed illustration of an ornate teapot with intricate floral patterns, set against a softly patterned background. The style is delicate and refined, focusing on the beauty of craftsmanship.',
    'A whimsical illustration of a fairy tale castle floating in the clouds, with bridges connecting to nearby floating islands. The style is dreamy and imaginative, with pastel colors and soft edges.',
    'An illustrated scene of a bustling medieval marketplace with vendors selling fruits, pottery, and fabrics. The style is lively and colorful, capturing the vibrancy and energy of a historic market.',
    'A minimalist logo of a majestic lions head with a geometric design, bold lines, and a golden gradient. The style is modern and clean, suitable for a luxury brand or a sports team.',
    'A 3D model of a vintage pirate ship with detailed sails, ropes, and cannons, floating on a realistic water surface. The style is hyper-realistic, ideal for a video game or animation.',
    'A childrens book illustration of a friendly dragon with big eyes and a cheerful smile, sitting in a meadow with flowers and butterflies around. The style is playful and colorful, capturing the charm of childrens art.',
    'A pixel art scene of a retro arcade with neon signs and characters holding joysticks, ready to play. The style is nostalgic, with pixelated details evoking the vibe of classic 8-bit video games.',
    'A surrealistic painting of a clock with melting numbers, floating in a cloudy dreamscape. The style is inspired by surrealism, with soft, dream-like colors and unusual shapes.',
    'A stylized Japanese ink wash painting of a koi fish swimming in a pond, with delicate water ripples and lily pads. The style is traditional and elegant, capturing the simplicity and flow of sumi-e art.',
    'A futuristic, cyberpunk-style logo of a robotic eagle with sharp, metallic features and neon accents. The style is bold and tech-inspired, ideal for a sci-fi game or brand.',
    'A photorealistic 3D render of a crystal gemstone with intricate facets, reflecting light in various rainbow colors. The style is polished and luxurious, capturing the beauty of precious gemstones.',
    'A vibrant, comic-style illustration of a superhero flying through a cityscape, with dynamic lines and bright colors. The style is inspired by classic comic books, with bold outlines and exaggerated poses.',
    'A watercolor painting of a quaint countryside cottage surrounded by wildflowers, with a soft, pastel palette. The style is gentle and charming, ideal for a storybook or greeting card.',
    ]
    generate_images(output_path, prompts, num_images_per_prompt=1)  # Generate 5 images per prompt
