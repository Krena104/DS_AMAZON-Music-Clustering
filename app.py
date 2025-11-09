import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(
    page_title="Amazon Music Clustering Dashboard",
    layout="wide"
)

# Load preprocessed data from pickle
@st.cache_data
def load_preprocessed(path="cleaned_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

# Load all objects from pickle
data = load_preprocessed()

df_reference = data["df_reference"]          
df_standard_scaled = data["df_standard_scaled"] 
df_pca = data["df_pca"]                    
kmeans_labels = data["kmeans_labels"]           
cluster_profile = data["cluster_profile"]       
feature_columns = data["feature_columns"]       
sil_score = data.get("sil_score", None)        
db_index = data.get("db_index", None)        

# Ensure cluster column exists
df_reference['Cluster'] = kmeans_labels

# Sidebar - Tab selection
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Metrics", "Visualization", "Insights & Export"])

# Overview
if page == "Overview":
    st.title("Amazon Music Clustering Dashboard")
    st.markdown("""
    This project groups similar Amazon Music songs based on their audio features using **K-Means clustering**.
    Explore clusters, visualize patterns, and understand musical characteristics.
    """)
    
    st.subheader("Dataset Summary")
    st.write(f"Total Songs: {df_reference.shape[0]}")
    st.write(f"Audio Features Used: {len(feature_columns)}")
    st.write(f"Number of Clusters: {df_reference['Cluster'].nunique()}")
    
    st.subheader("Top 5 Songs")
    st.dataframe(df_reference[['name_song', 'name_artists', 'genres', 'Cluster']].head(5))
    
    st.subheader("Cluster Distribution")
    cluster_counts = df_reference['Cluster'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Songs")
    ax.set_title("Cluster Size Distribution")
    st.pyplot(fig)

# Metrics
elif page == "Metrics":
    st.title("Cluster Evaluation Metrics")
    
    # Show precomputed scores
    st.metric("Silhouette Score", round(sil_score, 4) if sil_score is not None else "Not available")
    st.metric("Davies-Bouldin Index", round(db_index, 4) if db_index is not None else "Not available")
    
    st.subheader("Cluster-wise Feature Mean")
    st.dataframe(cluster_profile)

# Visualization
elif page == "Visualization":
    st.title("Cluster Visualizations")
    
    # PCA 2D Scatter Plot
    st.subheader("PCA 2D Scatter Plot")
    pca_df = pd.DataFrame(df_pca[:, :2], columns=['PC1', 'PC2'])
    pca_df['Cluster'] = kmeans_labels
    
    fig, ax = plt.subplots(figsize=(10,7))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=50, ax=ax)
    ax.set_title("PCA Visualization of Clusters")
    ax.grid(True)
    st.pyplot(fig)
    
    # Heatmap of Cluster Feature Means
    st.subheader("Heatmap of Cluster Feature Means")
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(cluster_profile, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Cluster-wise Feature Comparison")
    st.pyplot(fig)
    
    # Key Feature Averages per Cluster
    st.subheader("Key Feature Averages per Cluster")
    features_to_compare = ['energy', 'acousticness', 'valence', 'instrumentalness', 'speechiness']
    fig, ax = plt.subplots(figsize=(10,6))
    cluster_profile[features_to_compare].plot(kind='bar', ax=ax)
    ax.set_ylabel("Scaled Feature Value")
    ax.set_xlabel("Cluster")
    ax.set_title("Key Features by Cluster")
    ax.grid(True)
    st.pyplot(fig)

# Insights & Export
elif page == "Insights & Export":
    st.title("Insights and Export")
    
    st.subheader("Top Songs per Cluster")
    cluster_select = st.selectbox("Select Cluster", sorted(df_reference['Cluster'].unique()))
    top_n = st.slider("Number of Top Songs to Display", 5, 20, 5)
    
    top_songs = df_reference[df_reference['Cluster'] == cluster_select][['name_song', 'name_artists', 'genres']].head(top_n)
    st.dataframe(top_songs)
    
    st.subheader("Cluster Interpretation (Example)")
    st.markdown("""
    - **Cluster 0**: High danceability, high energy → Party tracks  
    - **Cluster 1**: Low energy, high acousticness → Chill acoustic  
    - **Cluster 2**: Medium energy & valence → Balanced mood 
    - **Cluster 3**: Instrumental-heavy tracks → Relaxed/Focus tracks
    """)
    
    st.subheader("Download Final Dataset")
    csv = df_reference.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "AmazonMusic_Clustered.csv", "text/csv")