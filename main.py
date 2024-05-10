import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score


# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Data/preprocessed_data.csv')
    df = df.dropna()
    return df


def main():
    st.title('Uncovering Crime Patterns In Data Using Clustering')

    # Load Data
    df = load_data()

    # Convert the 'Year' column to datetime format
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')

    # Group the data by year and choose 100 random samples for each year
    sampled_data = df.groupby('Year', group_keys=False).apply(lambda x: x.sample(min(len(x), 70), random_state=42))

    # Reset the index of the sampled data
    sampled_data.reset_index(drop=True, inplace=True)

    # Display the DataFrame
    num_rows_to_display = st.number_input('Number of rows to display:', min_value=10, max_value=len(df))
    st.write(df.sample(num_rows_to_display, random_state=42))

    # Clustering
    # Define the Streamlit app layout
    st.subheader('Clustering')
    st.subheader('Hierarchical Clustering vs K-Medoids')

    # Define the numbers of clusters to test
    cluster_numbers = [2, 4, 8, 10]

    # User input for the number of clusters
    selected_num_clusters = st.selectbox('Select the number of clusters:', [2, 4, 8, 10])

    # Proceed only if the user input is valid
    if selected_num_clusters:
        # Initialize a dictionary to store silhouette scores
        silhouette_scores = {'Hierarchical Clustering': [], 'KMedoids Clustering': []}

        # Select latitude and longitude columns
        geographical_data = sampled_data[['LATITUDE', 'LONGITUDE']].dropna()

        # Apply t-SNE for dimensionality reduction
        tsne_model = TSNE(n_components=2, random_state=42)
        tsne_data = tsne_model.fit_transform(geographical_data)

        for num_clusters in [selected_num_clusters]:
            # Hierarchical Clustering
            hierarchical_clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
            hc_labels = hierarchical_clustering.fit_predict(tsne_data)
            silhouette_hc = silhouette_score(tsne_data, hc_labels)
            silhouette_scores['Hierarchical Clustering'].append(silhouette_hc)

            # K-Medoids Clustering
            kmedoids = KMedoids(n_clusters=num_clusters, metric='euclidean', init='heuristic', max_iter=300,
                                random_state=42)
            km_labels = kmedoids.fit_predict(tsne_data)
            silhouette_km = silhouette_score(tsne_data, km_labels)
            silhouette_scores['KMedoids Clustering'].append(silhouette_km)

            # Plot Hierarchical Clustering and K-Medoids Clustering side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Plot Hierarchical Clustering
            axes[0].clear()  # Clear previous plot
            axes[0].scatter(tsne_data[:, 0], tsne_data[:, 1], c=hc_labels, cmap='viridis', s=50, alpha=0.5)
            axes[0].set_title(f'Hierarchical Clustering with t-SNE with {selected_num_clusters} clusters')
            axes[0].set_xlabel('t-SNE Component 1')
            axes[0].set_ylabel('t-SNE Component 2')

            # Plot K-Medoids Clustering
            axes[1].clear()  # Clear previous plot
            axes[1].scatter(tsne_data[:, 0], tsne_data[:, 1], c=km_labels, cmap='viridis', s=50, alpha=0.5)
            axes[1].set_title(f'K-Medoids Clustering with t-SNE with {num_clusters} clusters')
            axes[1].set_xlabel('t-SNE Component 1')
            axes[1].set_ylabel('t-SNE Component 2')

            # Show the plots
            st.pyplot(fig)

    # Find the maximum average Silhouette Score and its corresponding clustering algorithm for t-SNE
    avg_silhouette_scores = {algorithm: np.mean(scores) for algorithm, scores in silhouette_scores.items()}
    best_algorithm = max(avg_silhouette_scores, key=avg_silhouette_scores.get)
    max_silhouette_score = avg_silhouette_scores[best_algorithm]

    # Plotting the Silhouette Scores
    plt.figure(figsize=(10, 6))

    # Initialize lists to store silhouette scores for each algorithm
    hc_silhouette_scores = []
    km_silhouette_scores = []

    for num_clusters in cluster_numbers:
        # Hierarchical Clustering
        hierarchical_clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
        hc_labels = hierarchical_clustering.fit_predict(tsne_data)
        silhouette_hc = silhouette_score(tsne_data, hc_labels)
        hc_silhouette_scores.append(silhouette_hc)

        # K-Medoids Clustering
        kmedoids = KMedoids(n_clusters=num_clusters, metric='euclidean', init='heuristic', max_iter=300,
                            random_state=42)
        km_labels = kmedoids.fit_predict(tsne_data)
        silhouette_km = silhouette_score(tsne_data, km_labels)
        km_silhouette_scores.append(silhouette_km)

    # Plot silhouette scores
    st.subheader('Silhouette Scores')
    plt.plot(cluster_numbers, hc_silhouette_scores, marker='o', label='Hierarchical Clustering')
    plt.plot(cluster_numbers, km_silhouette_scores, marker='o', label='KMedoids Clustering')

    plt.title('Average Silhouette Scores for Different Clustering Algorithms')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.xticks(cluster_numbers)
    plt.legend()
    plt.grid(True)

    # Display the plot in Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # Display silhouette scores
    st.write("Silhouette Scores:")
    st.write(silhouette_scores)

    st.write(
        f"The best clustering algorithm is {best_algorithm} with an average Silhouette Score of {max_silhouette_score}.")

if __name__ == '__main__':
    main()
