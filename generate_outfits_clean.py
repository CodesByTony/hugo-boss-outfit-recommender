import pandas as pd
import random

print("👔 Generating outfit combinations (clean dataset)...")

# Load cleaned products
products = pd.read_csv('data/boss_products.csv')

print(f"📦 Loaded {len(products)} products")
print(f"Categories: {', '.join(sorted(products['category'].unique()))}\n")

# OUTFIT RULES - Only using categories we have images for
OUTFIT_COMBINATIONS = {
    'Business': {
        'anchor': ['Blazers', 'Trousers'],
        'must_have': ['Shirts', 'Shoes', 'Belts'],
        'optional': ['Ties']
    },
    'Smart-Casual': {
        'anchor': ['Blazers', 'Shirts'],
        'must_have': ['Pants', 'Shoes'],
        'optional': ['Belts']
    },
    'Casual': {
        'anchor': ['T-Shirts', 'Hoodies'],
        'must_have': ['Jeans', 'Pants'],
        'optional': ['Shoes']
    },
    'Formal': {
        'anchor': ['Blazers'],
        'must_have': ['Trousers', 'Shirts', 'Shoes'],
        'optional': ['Ties', 'Belts']
    }
}

# Color matching
COLOR_MATCHES = {
    'Black': ['White', 'Grey', 'Charcoal'],
    'Navy': ['White', 'Beige'],
    'Charcoal': ['White', 'Grey'],
    'Grey': ['White', 'Black', 'Navy'],
    'White': ['Black', 'Navy', 'Charcoal', 'Brown'],
    'Beige': ['Navy', 'White', 'Brown'],
    'Brown': ['Beige', 'White'],
    'Olive': ['Beige', 'White', 'Brown']
}

def get_complementary_colors(color):
    return COLOR_MATCHES.get(color, ['White', 'Black'])

def create_outfit(outfit_type, products_df):
    rules = OUTFIT_COMBINATIONS[outfit_type]
    outfit = {}
    
    # Select anchor item
    anchor_categories = rules['anchor']
    anchor_category = random.choice(anchor_categories)
    anchor_items = products_df[products_df['category'] == anchor_category]
    
    if len(anchor_items) == 0:
        return None
    
    anchor = anchor_items.sample(1).iloc[0]
    outfit['items'] = [anchor]
    
    base_color = anchor['color']
    matching_colors = get_complementary_colors(base_color)
    
    # Add must-have items
    for must_category in rules['must_have']:
        if must_category == anchor_category:
            continue
        
        candidates = products_df[products_df['category'] == must_category]
        
        if len(candidates) == 0:
            continue
        
        # Try to match colors
        color_matched = candidates[candidates['color'].isin(matching_colors + [base_color])]
        
        if len(color_matched) > 0:
            item = color_matched.sample(1).iloc[0]
        else:
            item = candidates.sample(1).iloc[0]
        
        outfit['items'].append(item)
    
    # Maybe add optional items
    if random.random() > 0.5 and len(rules.get('optional', [])) > 0:
        optional_category = random.choice(rules['optional'])
        optional_items = products_df[products_df['category'] == optional_category]
        
        if len(optional_items) > 0:
            color_matched = optional_items[optional_items['color'].isin(matching_colors + [base_color])]
            if len(color_matched) > 0:
                item = color_matched.sample(1).iloc[0]
            else:
                item = optional_items.sample(1).iloc[0]
            outfit['items'].append(item)
    
    return outfit

def generate_outfit_dataset(num_outfits=1000):
    outfits = []
    outfit_types = list(OUTFIT_COMBINATIONS.keys())
    
    for i in range(num_outfits):
        outfit_type = random.choice(outfit_types)
        outfit = create_outfit(outfit_type, products)
        
        if outfit is None:
            continue
        
        outfit_record = {
            'outfit_id': f'OUTFIT{i:05d}',
            'outfit_type': outfit_type,
            'num_items': len(outfit['items'])
        }
        
        for idx, item in enumerate(outfit['items']):
            outfit_record[f'item_{idx+1}_id'] = item['product_id']
            outfit_record[f'item_{idx+1}_category'] = item['category']
        
        outfits.append(outfit_record)
    
    return pd.DataFrame(outfits)

# Generate outfits
outfits_df = generate_outfit_dataset(num_outfits=1000)
outfits_df.to_csv('data/outfit_combinations.csv', index=False)

print(f"\n✅ Generated {len(outfits_df)} outfit combinations!")
print(f"📁 Saved to: data/outfit_combinations.csv\n")

print("📊 OUTFIT TYPE DISTRIBUTION:")
print(outfits_df['outfit_type'].value_counts())

print("\n👔 SAMPLE OUTFITS:")
for i in range(3):
    outfit = outfits_df.iloc[i]
    print(f"\n{outfit['outfit_id']} ({outfit['outfit_type']}):")
    for j in range(1, outfit['num_items']+1):
        if f'item_{j}_category' in outfit:
            print(f"  - {outfit[f'item_{j}_category']} ({outfit[f'item_{j}_id']})")