import pandas as pd

print("🧹 Cleaning up dataset - keeping only categories with real images...")

# Load products
products = pd.read_csv('data/boss_products.csv')

print(f"Original dataset: {len(products)} products")
print("\nOriginal categories:")
print(products['category'].value_counts().sort_index())

# Categories you HAVE images for (based on your download)
CATEGORIES_WITH_IMAGES = [
    'Hoodies',
    'T-Shirts', 
    'Shirts',
    'Blazers',
    'Jeans',
    'Pants',
    'Trousers',
    'Shoes',
    'Belts',
    'Ties'
]

# Filter to only keep products with real images
filtered = products[products['category'].isin(CATEGORIES_WITH_IMAGES)].copy()

# Also filter out any products still using placeholder images
filtered = filtered[~filtered['image_url'].str.contains('placeholder', na=False)]

print(f"\n✅ Cleaned dataset: {len(filtered)} products")
print(f"❌ Removed: {len(products) - len(filtered)} products\n")

print("Remaining categories:")
print(filtered['category'].value_counts().sort_index())

# Save cleaned dataset
filtered.to_csv('data/boss_products.csv', index=False)

print("\n✅ Dataset cleaned and saved!")

# Show what was removed
removed_categories = set(products['category'].unique()) - set(filtered['category'].unique())
if removed_categories:
    print(f"\n🗑️  Removed categories (no images):")
    for cat in sorted(removed_categories):
        count = len(products[products['category'] == cat])
        print(f"  • {cat} ({count} products)")