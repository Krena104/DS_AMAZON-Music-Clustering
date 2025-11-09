# DS_AMAZON-Music-Clustering
# 🎵 Music Genre Clustering using Audio Features

### 🔍 Project Overview
With millions of songs available on platforms like Amazon Music, manually categorizing tracks into genres or moods is nearly impossible.  
This project uses **unsupervised machine learning** to **automatically group similar songs** based on their **audio characteristics**, such as tempo, energy, and danceability.  

By leveraging clustering techniques like **K-Means**, **DBSCAN**, and **Hierarchical Clustering**, this project aims to identify patterns in sound features — forming natural clusters that can represent **genres, moods, or playlists** without any manual labeling.

---

## 🚀 Objectives
- Automatically group songs with similar sound characteristics.
- Discover natural clusters that may represent **genres, moods, or themes**.
- Support **recommendation systems** and **playlist generation**.
- Visualize and interpret clusters to understand distinct song types.

---

## 💡 Business Use Cases

| Use Case | Description |
|-----------|--------------|
| 🎧 **Personalized Playlist Curation** | Automatically group songs that sound similar to enhance playlist generation. |
| 🔎 **Improved Song Discovery** | Suggest similar tracks to users based on their preferred audio profile. |
| 🎤 **Artist Analysis** | Help artists and producers identify competitive songs within the same audio cluster. |
| 📊 **Market Segmentation** | Streaming platforms can analyze user listening patterns and optimize promotions or recommendations. |

---

## 🧠 Project Workflow

### **1️⃣ Data Exploration & Preprocessing**
- Load dataset: `single_genre_artists.csv`  
- Inspect data structure, datatypes, and missing values.  
- Remove unnecessary columns: `track_id`, `track_name`, `artist_name`.  
- Normalize numerical features using **StandardScaler** or **MinMaxScaler** for consistent scaling.  

### **2️⃣ Feature Selection**
Selected features that describe the **sound and emotion** of a song:


These features capture rhythm, energy, and mood — essential for clustering.

### **3️⃣ Dimensionality Reduction (for Visualization)**
- Use **PCA** to reduce to 2D/3D for visualization.  
- Optionally use **t-SNE** for capturing non-linear relationships.

### **4️⃣ Clustering Algorithms**
#### 🔹 **K-Means Clustering**
- Determine the optimal number of clusters `k` using:
  - **Elbow Method** (Inertia/SSE)
  - **Silhouette Score**
- Apply `KMeans(n_clusters=k)` and append cluster labels to dataset.

#### 🔹 **DBSCAN**
- Detect noise and arbitrary-shaped clusters.
- Tune parameters `eps` and `min_samples`.

#### 🔹 **Hierarchical Clustering**
- Create dendrograms for hierarchical relationships.
- Understand how clusters merge or split.

### **5️⃣ Cluster Evaluation & Interpretation**
Metrics used:
- **Silhouette Score** – Higher is better (range: -1 to 1)
- **Davies-Bouldin Index** – Lower is better
- **Inertia (for K-Means)** – Measures compactness

Interpret clusters based on average feature values:
- Example:  
  - 🎉 Cluster 1: High energy, high danceability → *“Party Tracks”*  
  - 🌙 Cluster 2: Low tempo, high acousticness → *“Chill Acoustic”*

### **6️⃣ Visualization**
- **2D PCA/t-SNE scatter plots** (color-coded by cluster)
- **Bar charts** for average feature values per cluster
- **Heatmaps** to compare clusters across all features

### **7️⃣ Final Analysis & Export**
- Add final cluster labels to the original dataset  
- Save results as `clustered_songs.csv`  
- Summarize each cluster's unique characteristics  

---

## 🧾 Results

✅ Formed distinct song clusters based on their acoustic and rhythmic features.  
✅ Each cluster represents a unique musical mood or genre.  
✅ Visualizations help interpret the sonic characteristics behind each cluster.  
✅ Clusters can be used to **recommend similar songs** or **generate playlists** automatically.

---

## 📈 Evaluation Metrics

| Metric | Description |
|---------|--------------|
| **Silhouette Score** | Measures how similar a song is to its own cluster compared to others. |
| **Davies-Bouldin Index** | Evaluates intra-cluster and inter-cluster separation. |
| **Cluster Visualization** | 2D PCA/t-SNE plots for interpretability. |
| **Cluster Size Balance** | Distribution and evenness across clusters. |
| **Feature Interpretability** | Clarity of dominant audio features per cluster. |

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|--------|
| **Programming Language** | Python |
| **Data Analysis** | Pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Clustering Algorithms** | K-Means, DBSCAN, Hierarchical |
| **Dimensionality Reduction** | PCA, t-SNE |

---

## 📂 Dataset Description

**File Name:** `Data_Set.csv`  
**Features Include:**

Future Enhancements

Integrate Spotify or Amazon Music API for real-time data.

Use Deep Learning (Autoencoders) for advanced feature extraction.

Deploy as a web-based recommendation system using Streamlit.

Add genre labeling using supervised learning on clustered data
