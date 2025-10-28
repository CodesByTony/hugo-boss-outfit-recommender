import streamlit as st
import pandas as pd
import torch
import os

st.set_page_config(page_title="BOSS | Men's Fashion", page_icon="🖤", layout="wide")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600&display=swap');
    * { font-family: 'Montserrat', sans-serif; }
    .boss-header { 
        background: #000; 
        color: #fff; 
        padding: 30px; 
        text-align: center; 
        font-size: 48px; 
        letter-spacing: 12px; 
        font-weight: 300;
        margin: -70px -100px 40px -100px;
    }
    .stButton>button {
        background: #000;
        color: #fff;
        border: none;
        padding: 12px;
        font-size: 11px;
        letter-spacing: 2px;
        width: 100%;
    }
    .stButton>button:hover { background: #333; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Helper function to display images
def show_image(image_path, width=None):
    """Display image from local path or URL"""
    if image_path.startswith('http'):
        st.image(image_path, use_container_width=(width is None), width=width)
    else:
        # Local file path
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=(width is None), width=width)
        else:
            # Fallback if image not found
            st.image("https://via.placeholder.com/400x500/333/FFF?text=No+Image", 
                    use_container_width=(width is None), width=width)

# Load data
@st.cache_data
def load_data():
    products = pd.read_csv('data/boss_products.csv')
    return products

@st.cache_resource
def load_gnn():
    data = torch.load('models/gnn_model.pt', map_location='cpu', weights_only=False)
    return data['embeddings'], data['product_to_idx'], data['idx_to_product']

products = load_data()
embeddings, product_to_idx, idx_to_product = load_gnn()

# Session state
if 'view' not in st.session_state:
    st.session_state.view = 'home'
if 'selected_id' not in st.session_state:
    st.session_state.selected_id = None
if 'cart' not in st.session_state:
    st.session_state.cart = []

# AI Recommendations
def get_recommendations(product_id, top_k=3):
    """Smart fashion recommendations with outfit slot logic"""
    
    query = products[products['product_id'] == product_id].iloc[0]
    query_idx = product_to_idx[product_id]
    query_emb = embeddings[query_idx]
    
    # Calculate similarities
    similarities = torch.nn.functional.cosine_similarity(
        query_emb.unsqueeze(0), embeddings, dim=1
    )
    
    # SMART FILTERING
    recommendations = []
    used_slots = {query['slot']}
    
    # Define formality levels
    formality_order = {'CASUAL': 0, 'SMART_CASUAL': 1, 'FORMAL': 2}
    query_formality = formality_order.get(query['formality'], 0)
    
    # Sort by similarity
    sorted_indices = torch.argsort(similarities, descending=True)
    
    for idx in sorted_indices:
        if len(recommendations) >= top_k:
            break
        
        rec_id = idx_to_product[idx.item()]
        if rec_id == product_id:
            continue
        
        rec = products[products['product_id'] == rec_id].iloc[0]
        
        # RULE 1: Never recommend same slot (no jeans + pants!)
        if rec['slot'] in used_slots:
            continue
        
        # RULE 2: Match formality level (±1 level OK)
        rec_formality = formality_order.get(rec['formality'], 0)
        if abs(query_formality - rec_formality) > 1:
            continue
        
        # Add to recommendations
        recommendations.append(rec)
        used_slots.add(rec['slot'])
    
    return pd.DataFrame(recommendations)
    query = products[products['product_id'] == product_id].iloc[0]
    query_idx = product_to_idx[product_id]
    query_emb = embeddings[query_idx]
    
    similarities = torch.nn.functional.cosine_similarity(
        query_emb.unsqueeze(0), embeddings, dim=1
    )
    
    # Exclude same category
    for idx, p in enumerate(products.itertuples()):
        if p.category == query['category'] or p.product_id == product_id:
            similarities[idx] = -1
    
    # Get diverse recommendations
    recs = []
    used_cats = {query['category']}
    sorted_idx = torch.argsort(similarities, descending=True)
    
    for idx in sorted_idx:
        if len(recs) >= top_k:
            break
        rec_id = idx_to_product[idx.item()]
        rec = products[products['product_id'] == rec_id].iloc[0]
        if rec['category'] not in used_cats:
            recs.append(rec)
            used_cats.add(rec['category'])
    
    return pd.DataFrame(recs)

# Header
st.markdown('<div class="boss-header">BOSS</div>', unsafe_allow_html=True)

# Navigation
col1, col2, col3 = st.columns([4, 1, 1])
with col2:
    if st.button("🏠 HOME"):
        st.session_state.view = 'home'
        st.rerun()
with col3:
    if st.button(f"🛒 CART ({len(st.session_state.cart)})"):
        st.session_state.view = 'cart'
        st.rerun()

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### FILTERS")
    
    # Only show categories that have real images
    available_categories = sorted(products['category'].unique().tolist())
    category = st.selectbox("Category", ["All"] + available_categories)
    
    # Color filter
    st.markdown("**Color**")
    colors = ["All"] + sorted(products['color'].unique().tolist())
    selected_color = st.selectbox("Filter by color", colors)
    
    # Apply color filter
    if category != "All":
        filtered = products[products['category'] == category]
    else:
        filtered = products
    
    if selected_color != "All":
        filtered = filtered[filtered['color'] == selected_color]
    
    if category != "All":
        filtered = products[products['category'] == category]
    else:
        filtered = products
    
    st.metric("Products", len(filtered))
    
    # Show how many have real images
    real_images = len(filtered[~filtered['image_url'].str.contains('placeholder', na=False)])
    st.metric("Real BOSS Images", real_images)

# Main Content
if st.session_state.view == 'home':
    st.markdown(f"## {category if category != 'All' else 'All Products'}")
    st.markdown(f"*{len(filtered)} items*")
    st.markdown("")
    
    # Product grid
    for row in range(min(10, (len(filtered) + 3) // 4)):
        cols = st.columns(4)
        for col_idx in range(4):
            idx = row * 4 + col_idx
            if idx < len(filtered):
                product = filtered.iloc[idx]
                with cols[col_idx]:
                    show_image(product['image_url'])
                    st.markdown(f"**{product['name']}**")
                    st.markdown(f"⭐ {product['rating']}")
                    st.markdown(f"### €{product['price']}")
                    if st.button("VIEW", key=f"v_{product['product_id']}"):
                        st.session_state.view = 'detail'
                        st.session_state.selected_id = product['product_id']
                        st.rerun()

elif st.session_state.view == 'detail':
    product = products[products['product_id'] == st.session_state.selected_id].iloc[0]
    
    if st.button("← Back"):
        st.session_state.view = 'home'
        st.rerun()
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_image(product['image_url'], width=500)
    
    with col2:
        st.markdown(f"# {product['name']}")
        st.markdown(f"⭐ {product['rating']}/5.0 ({product['reviews']} reviews)")
        st.markdown("---")
        st.markdown(f"# €{product['price']}")
        st.markdown("---")
        st.markdown(f"**Material:** {product['material']}")
        st.markdown(f"**Sizes:** {product['sizes']}")
        st.markdown("")
        
        if st.button("🛒 ADD TO CART", type="primary"):
            st.session_state.cart.append(product.to_dict())
            st.success("Added!")
            st.balloons()
        
        st.markdown("")
        if st.button("✨ COMPLETE THE LOOK (AI)"):
            st.session_state.view = 'outfit'
            st.rerun()
        
        st.markdown("---")
        st.markdown(product['description'])

elif st.session_state.view == 'outfit':
    product = products[products['product_id'] == st.session_state.selected_id].iloc[0]
    
    if st.button("← Back"):
        st.session_state.view = 'detail'
        st.rerun()
    
    st.markdown("## ✨ AI Complete The Look")
    st.markdown("*Powered by Graph Neural Networks*")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Your Selection")
        show_image(product['image_url'], width=300)
        st.markdown(f"**{product['name']}**")
        st.markdown(f"€{product['price']}")
    
    with col2:
        st.markdown("### 🤖 AI Recommendations")
        
        recs = get_recommendations(product['product_id'], top_k=3)
        
        if len(recs) > 0:
            rec_cols = st.columns(3)
            total = 0
            
            for idx, (_, rec) in enumerate(recs.iterrows()):
                with rec_cols[idx]:
                    show_image(rec['image_url'])
                    st.markdown(f"**{rec['name']}**")
                    st.markdown(f"€{rec['price']}")
                    st.markdown(f"*{rec['category']}*")
                    if st.button("Add", key=f"a_{rec['product_id']}"):
                        st.session_state.cart.append(rec.to_dict())
                        st.success("✅")
                    total += rec['price']
            
            st.markdown("---")
            st.markdown(f"### Total Outfit: €{product['price'] + total}")
            
            if st.button("🛒 ADD COMPLETE OUTFIT", type="primary"):
                st.session_state.cart.append(product.to_dict())
                for _, rec in recs.iterrows():
                    st.session_state.cart.append(rec.to_dict())
                st.success("✅ Outfit added!")
                st.balloons()

elif st.session_state.view == 'cart':
    st.markdown("## 🛒 Cart")
    st.markdown("---")
    
    if len(st.session_state.cart) == 0:
        st.info("Empty cart")
        if st.button("← Shop"):
            st.session_state.view = 'home'
            st.rerun()
    else:
        total = 0
        for idx, item in enumerate(st.session_state.cart):
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                show_image(item['image_url'], width=80)
            with col2:
                st.markdown(f"**{item['name']}**")
                st.markdown(f"€{item['price']}")
            with col3:
                if st.button("🗑️", key=f"rm_{idx}"):
                    st.session_state.cart.pop(idx)
                    st.rerun()
            st.markdown("---")
            total += item['price']
        
        st.markdown(f"### Total: €{total}")
        if st.button("✅ CHECKOUT", type="primary"):
            st.success("Order placed!")
            st.balloons()
            st.session_state.cart = []

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #999; font-size: 10px;'>BOSS | Real Product Images | AI-Powered by GNN</div>", unsafe_allow_html=True)