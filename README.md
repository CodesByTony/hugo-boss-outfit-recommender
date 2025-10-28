cat > README.md << 'EOF'
# BOSS Complete-The-Look AI

🤖 AI-powered outfit recommendation system for BOSS menswear using Graph Neural Networks.

## 🎯 Features

- **1,000 BOSS products** across 10 categories
- **Graph Neural Network** recommendations
- **Smart outfit logic** (no duplicate item types, formality matching)
- **Real BOSS product images** (120+ curated images)
- **Complete outfit suggestions** (Top + Bottom + Shoes + Accessories)

## 🧠 Technology Stack

- **Frontend:** Streamlit
- **AI Model:** PyTorch Geometric (Graph Neural Networks)
- **Training Data:** 1,000 synthetic outfit combinations
- **Recommendation Logic:** Fashion slot-based filtering + GNN embeddings

## 📊 How It Works

1. **Product Representation:** Each product is a node in a graph
2. **Outfit Relationships:** Products appearing together are connected
3. **GNN Learning:** Neural network learns 64-dimensional embeddings
4. **Smart Filtering:** Ensures variety (different slots, matching formality)


## 🎓 Academic Project

Master's thesis prototype demonstrating AI-driven fashion recommendations.

**Note:** Product images © HUGO BOSS, used for academic purposes only.

---

Built with  using Streamlit + PyTorch
EOF
