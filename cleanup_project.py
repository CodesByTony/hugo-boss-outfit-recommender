import os
import shutil

print("🧹 Cleaning up project files...\n")

# Files to KEEP
KEEP_FILES = {
    'app_with_local_images.py',  # Main app
    'build_smart_gnn.py',  # GNN builder
    'cleanup_dataset.py',  # Dataset cleaner
    'generate_outfits_clean.py',  # Outfit generator
    'map_real_boss_images.py',  # Image mapper
    'cleanup_project.py',  # This file
    'README.md',
    'LICENSE'
}

KEEP_FOLDERS = {
    'data',
    'models',
    'images',
    '.git'
}

# Get all files in root
all_files = [f for f in os.listdir('.') if os.path.isfile(f)]
all_folders = [f for f in os.listdir('.') if os.path.isdir(f)]

# Files to delete
delete_files = [f for f in all_files if f not in KEEP_FILES and not f.startswith('.')]

print("🗑️  Files to delete:")
for f in delete_files:
    print(f"  • {f}")

# Delete files
for f in delete_files:
    try:
        os.remove(f)
        print(f"  ✓ Deleted: {f}")
    except Exception as e:
        print(f"  ✗ Error deleting {f}: {e}")

print("\n✅ Cleanup complete!")

print("\n📁 Final project structure:")
print("hugo-boss-outfit-recommender/")
print("├── data/")
print("│   ├── boss_products.csv")
print("│   └── outfit_combinations.csv")
print("├── models/")
print("│   └── gnn_model.pt")
print("├── images/")
print("│   ├── hoodies/")
print("│   ├── t-shirts/")
print("│   ├── shirts/")
print("│   ├── blazers/")
print("│   ├── jeans/")
print("│   ├── pants/")
print("│   ├── shoes/")
print("│   ├── belts/")
print("│   └── ties/")
print("├── app_with_local_images.py  ← MAIN APP")
print("├── build_smart_gnn.py")
print("├── cleanup_dataset.py")
print("├── generate_outfits_clean.py")
print("├── map_real_boss_images.py")
print("├── README.md")
print("└── LICENSE")