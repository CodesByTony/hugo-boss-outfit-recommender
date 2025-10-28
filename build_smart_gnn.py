import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np

print("🧠 Building Fashion-Smart GNN...")

# Load data
products = pd.read_csv('data/boss_products.csv')
outfits = pd.read_csv('data/outfit_combinations.csv')

print(f"📦 {len(products)} products, {len(outfits)} outfits\n")

# DEFINE FASHION SLOTS (This is the key!)
OUTFIT_SLOTS = {
    'TOPS': ['Hoodies', 'T-Shirts', 'Shirts', 'Blazers'],
    'BOTTOMS': ['Jeans', 'Pants', 'Trousers'],
    'SHOES': ['Shoes'],
    'ACCESSORIES': ['Belts', 'Ties']
}

# FORMALITY LEVELS
FORMALITY = {
    'CASUAL': ['Hoodies', 'T-Shirts', 'Jeans'],
    'SMART_CASUAL': ['Shirts', 'Pants', 'Trousers', 'Belts'],
    'FORMAL': ['Blazers', 'Trousers', 'Ties', 'Shoes']
}

# Add metadata to products
def get_slot(category):
    for slot, categories in OUTFIT_SLOTS.items():
        if category in categories:
            return slot
    return 'OTHER'

def get_formality(category):
    if category in FORMALITY['FORMAL']:
        return 'FORMAL'
    elif category in FORMALITY['SMART_CASUAL']:
        return 'SMART_CASUAL'
    elif category in FORMALITY['CASUAL']:
        return 'CASUAL'
    return 'CASUAL'

products['slot'] = products['category'].apply(get_slot)
products['formality'] = products['category'].apply(get_formality)

print("📊 Product Distribution by Slot:")
print(products['slot'].value_counts())
print("\n📊 Product Distribution by Formality:")
print(products['formality'].value_counts())

# Build graph
product_to_idx = {pid: idx for idx, pid in enumerate(products['product_id'])}
idx_to_product = {idx: pid for pid, idx in product_to_idx.items()}

edges = []
for _, outfit in outfits.iterrows():
    items = []
    for i in range(1, 10):
        col = f'item_{i}_id'
        if col in outfit and pd.notna(outfit[col]):
            items.append(outfit[col])
    
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            idx_i = product_to_idx[items[i]]
            idx_j = product_to_idx[items[j]]
            edges.append([idx_i, idx_j])
            edges.append([idx_j, idx_i])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Create features
categories = sorted(products['category'].unique())
colors = sorted(products['color'].unique())
slots = sorted(products['slot'].unique())
formalities = sorted(products['formality'].unique())

cat_to_idx = {c: i for i, c in enumerate(categories)}
color_to_idx = {c: i for i, c in enumerate(colors)}
slot_to_idx = {s: i for i, s in enumerate(slots)}
formality_to_idx = {f: i for i, f in enumerate(formalities)}

# One-hot encode
category_onehot = np.eye(len(categories))[products['category'].map(cat_to_idx)]
color_onehot = np.eye(len(colors))[products['color'].map(color_to_idx)]
slot_onehot = np.eye(len(slots))[products['slot'].map(slot_to_idx)]
formality_onehot = np.eye(len(formalities))[products['formality'].map(formality_to_idx)]

# Normalize price
price_norm = (products['price'] - products['price'].min()) / (products['price'].max() - products['price'].min())

# Combine features
features = np.hstack([
    category_onehot,
    color_onehot,
    slot_onehot,
    formality_onehot,
    price_norm.values.reshape(-1, 1)
])

x = torch.tensor(features, dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

print(f"\n📈 Graph: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_node_features} features")

# GNN Model
class FashionGNN(nn.Module):
    def __init__(self, num_features, hidden=128, output=64):
        super(FashionGNN, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden)
        self.conv2 = SAGEConv(hidden, output)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(output)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x

model = FashionGNN(num_features=data.num_node_features)

print("\n🎯 Generating embeddings...")
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index)

# Save everything
torch.save({
    'model_state_dict': model.state_dict(),
    'embeddings': embeddings,
    'product_to_idx': product_to_idx,
    'idx_to_product': idx_to_product,
    'categories': categories,
    'colors': colors,
    'slots': slots,
    'formalities': formalities
}, 'models/gnn_model.pt')

# Also save metadata to products
products.to_csv('data/boss_products.csv', index=False)

print(f"\n✅ Smart GNN ready!")
print("\n🧪 Testing recommendations...")

# Test function
def test_smart_recommendations(product_id):
    query = products[products['product_id'] == product_id].iloc[0]
    query_idx = product_to_idx[product_id]
    query_emb = embeddings[query_idx]
    
    print(f"\n🔍 Query: {query['name']}")
    print(f"   Slot: {query['slot']}, Formality: {query['formality']}, Color: {query['color']}")
    
    similarities = torch.nn.functional.cosine_similarity(
        query_emb.unsqueeze(0), embeddings, dim=1
    )
    
    # SMART FILTERING
    recommendations = []
    used_slots = {query['slot']}
    
    for idx in torch.argsort(similarities, descending=True):
        if len(recommendations) >= 3:
            break
        
        rec_id = idx_to_product[idx.item()]
        if rec_id == product_id:
            continue
        
        rec = products[products['product_id'] == rec_id].iloc[0]
        
        # RULE 1: Different slot
        if rec['slot'] in used_slots:
            continue
        
        # RULE 2: Match formality (or one level apart)
        formality_levels = ['CASUAL', 'SMART_CASUAL', 'FORMAL']
        query_level = formality_levels.index(query['formality'])
        rec_level = formality_levels.index(rec['formality'])
        if abs(query_level - rec_level) > 1:
            continue
        
        # RULE 3: Color compatibility (not too strict)
        # Accept if same color OR complementary
        
        recommendations.append(rec)
        used_slots.add(rec['slot'])
    
    print("\n✨ Smart Recommendations:")
    for rec in recommendations:
        print(f"   ✓ {rec['name']} ({rec['slot']}) - {rec['formality']} - {rec['color']}")
    
    return recommendations

# Test cases
print("\n" + "="*60)
print("TEST CASES")
print("="*60)

# Test 1: Jeans (Casual bottom)
jeans = products[products['category'] == 'Jeans'].sample(1).iloc[0]
test_smart_recommendations(jeans['product_id'])

# Test 2: Blazer (Formal top)
blazer = products[products['category'] == 'Blazers'].sample(1).iloc[0]
test_smart_recommendations(blazer['product_id'])

# Test 3: Hoodie (Casual top)
hoodie = products[products['category'] == 'Hoodies'].sample(1).iloc[0]
test_smart_recommendations(hoodie['product_id'])

print("\n✅ Model saved to: models/gnn_model.pt")