import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
@st.cache_data
def load_data():
    products = pd.read_csv("products.csv", dtype={"product_id": "int32", "product_name": "category"})
    orders = pd.read_csv("orders.csv", dtype={"order_id": "int32", "user_id": "int32"})

    chunks_prior = pd.read_csv("order_products__prior.csv", chunksize=500000, dtype={"order_id": "int32", "product_id": "int32", "reordered": "int8"})
    chunks_train = pd.read_csv("order_products__train.csv", chunksize=500000, dtype={"order_id": "int32", "product_id": "int32", "reordered": "int8"})

    order_products_prior = pd.concat(chunks_prior, ignore_index=True)
    order_products_train = pd.concat(chunks_train, ignore_index=True)
    order_products = pd.concat([order_products_prior, order_products_train], ignore_index=True)

    order_products = order_products.merge(products, on="product_id", how="left")
    order_products = order_products.merge(orders, on="order_id", how="left")

    return products, orders, order_products

products, orders, order_products = load_data()

# Create Optimized User-Item Sparse Matrix
@st.cache_data
def create_sparse_matrix(order_products):
    unique_users = order_products['user_id'].astype('category').cat.codes.values
    unique_products = order_products['product_id'].astype('category').cat.codes.values
    
    user_item_sparse = coo_matrix(
        (order_products['reordered'].values, (unique_users, unique_products))
    ).tocsr()
    
    return user_item_sparse, unique_users, unique_products

user_item_sparse, unique_users, unique_products = create_sparse_matrix(order_products)

# Train Optimized KNN Model
@st.cache_resource
def train_knn_model():
    user_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    user_knn.fit(user_item_sparse)
    return user_knn

knn_model = train_knn_model()

# Create User Index Mapping
user_codes = order_products[['user_id']].drop_duplicates().reset_index(drop=True)
user_idx_map = {u: i for i, u in enumerate(user_codes['user_id'])}

# Recommendation Function (Collaborative Filtering)
def recommend_knn(user_id, num_recommendations=5):
    if user_id not in user_idx_map:
        return "User not found!"
    
    user_idx = user_idx_map[user_id]
    distances, indices = knn_model.kneighbors(user_item_sparse[user_idx], n_neighbors=6)
    similar_users = indices.flatten()[1:]
    
    recommended_products = order_products[order_products['user_id'].isin(user_codes.iloc[similar_users]['user_id'])]
    
    user_purchased = set(order_products[order_products['user_id'] == user_id]['product_id'])
    top_products = [p for p in recommended_products['product_id'].value_counts().index if p not in user_purchased][:num_recommendations]
    
    return products[products['product_id'].isin(top_products)][['product_id', 'product_name']]

@st.cache_resource
def compute_sparse_cosine_similarity():
    product_features = pd.get_dummies(products[['aisle_id', 'department_id']])
    product_sparse = csr_matrix(product_features)

    nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    nn_model.fit(product_sparse)

    product_idx = {product: i for i, product in enumerate(products['product_id'])}

    return nn_model, product_sparse, product_idx

nn_model, product_sparse, product_idx = compute_sparse_cosine_similarity()

# Optimized Content-Based Recommendation Function
def recommend_content_based(product_id, num_recommendations=5):
    if product_id not in product_idx:
        return "Product not found!"
    
    idx = product_idx[product_id]
    distances, indices = nn_model.kneighbors(product_sparse[idx], n_neighbors=num_recommendations+1)
    
    recommended_indices = indices.flatten()[1:]
    recommended_products = products.iloc[recommended_indices][['product_id', 'product_name']]
    
    return recommended_products.reset_index(drop=True)


# Hybrid Recommendation System
def get_real_time_recommendations(user_id, product_id, method="hybrid"):
    if method == "collaborative":
        return recommend_knn(user_id)
    elif method == "content":
        return recommend_content_based(product_id)
    elif method == "hybrid":
        collab_recommendations = recommend_knn(user_id)
        content_recommendations = recommend_content_based(product_id)
        return pd.concat([collab_recommendations, content_recommendations]).drop_duplicates().reset_index(drop=True)
    else:
        return "Invalid method!"

st.title("Real-Time E-Commerce Recommendation System")

st.sidebar.header("User & Product Selection")
user_id = st.sidebar.number_input("Enter User ID", min_value=int(order_products['user_id'].min()), max_value=int(order_products['user_id'].max()), step=1)
product_id = st.sidebar.number_input("Enter Product ID", min_value=int(products['product_id'].min()), max_value=int(products['product_id'].max()), step=1)
method = st.sidebar.selectbox("Choose Recommendation Method", ["collaborative", "content", "hybrid"])

if st.sidebar.button("Get Recommendations"):
    recommendations = get_real_time_recommendations(user_id, product_id, method)
    st.subheader("🛍 Recommended Products")
    st.dataframe(recommendations)
