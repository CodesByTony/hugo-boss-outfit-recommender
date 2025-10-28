import pandas as pd
import os
import random
from pathlib import Path

print("🖼️ Mapping real BOSS images to products...")

# Your folder structure
IMAGE_BASE = "images"

# Mapping your folders to our product categories
FOLDER_MAPPING = {
    'hoodies': ['Hoodies'],
    't-shirts': ['T-Shirts'],
    'shirts': ['Shirts'],
    'blazers': ['Blazers'],
    'jackets': ['Blazers'],  # Merge jackets into Blazers
    'jeans': ['Jeans'],
    'pants': ['Pants', 'Trousers'],  # Use pants images for both
    'shoes': ['Shoes'],
    'belts': ['Belts'],
    'ties': ['Ties'],
    'accessories': ['Ties', 'Belts'],  # Distribute accessories
    # 'uncertain': skip
}

# Load products
products = pd.read_csv('data/boss_products.csv')

print(f"📦 Loaded {len(products)} products\n")

# Scan your images folder and collect all images
category_images = {}

for folder_name, categories in FOLDER_MAPPING.items():
    folder_path = os.path.join(IMAGE_BASE, folder_name)
    
    if os.path.exists(folder_path):
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_files.extend(Path(folder_path).glob(ext))
        
        # Convert to relative paths
        image_paths = [str(img) for img in image_files]
        
        print(f"📁 {folder_name:15} → {len(image_paths)} images")
        
        # Assign to all mapped categories
        for category in categories:
            if category not in category_images:
                category_images[category] = []
            category_images[category].extend(image_paths)
    else:
        print(f"⚠️  {folder_name:15} → Folder not found")

print(f"\n📊 Category Image Counts:")
for category in sorted(category_images.keys()):
    print(f"  {category:15} {len(category_images[category])} images")

# Assign images to products
assigned_count = 0
placeholder_count = 0

for idx, row in products.iterrows():
    category = row['category']
    
    if category in category_images and len(category_images[category]) > 0:
        # Randomly select an image from this category
        image_path = random.choice(category_images[category])
        products.at[idx, 'image_url'] = image_path
        assigned_count += 1
    else:
        # Use placeholder for categories without images
        products.at[idx, 'image_url'] = f"https://via.placeholder.com/400x500/333/FFF?text={category}"
        placeholder_count += 1

# Save updated products
products.to_csv('data/boss_products.csv', index=False)

print(f"\n✅ Image Assignment Complete!")
print(f"  ✓ Real images: {assigned_count}")
print(f"  ⚠ Placeholders: {placeholder_count}")

# Show statistics per category
print(f"\n📈 Assignment Statistics:")
for category in sorted(products['category'].unique()):
    cat_products = products[products['category'] == category]
    real_images = len(cat_products[~cat_products['image_url'].str.contains('placeholder')])
    total = len(cat_products)
    
    if category in category_images and len(category_images[category]) > 0:
        reuse_factor = total / len(category_images[category])
        print(f"  {category:15} {real_images}/{total} products ({len(category_images[category])} unique images, ~{reuse_factor:.1f}x reuse)")
    else:
        print(f"  {category:15} {real_images}/{total} products (NO IMAGES - using placeholder)")

print("\n💡 Categories without images:")
no_image_cats = [cat for cat in products['category'].unique() if cat not in category_images or len(category_images[cat]) == 0]
if no_image_cats:
    for cat in no_image_cats:
        num_products = len(products[products['category'] == cat])
        print(f"  • {cat} ({num_products} products)")
    print("\n  Options:")
    print("  1. Remove these categories from products")
    print("  2. Keep with placeholders (current)")
    print("  3. Download more images for these categories")
else:
    print("  None! All categories have real images! 🎉")

# Show sample assignments
print("\n📦 Sample Product Image Assignments:")
for category in sorted(category_images.keys())[:5]:
    sample = products[products['category'] == category].head(1)
    if len(sample) > 0:
        prod = sample.iloc[0]
        print(f"\n  {category}:")
        print(f"    Product: {prod['name']}")
        print(f"    Image: {prod['image_url']}")