import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import RobustScaler
import sklearn
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from skfuzzy.cluster import cmeans
from minisom import MiniSom


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

    # Create legend list
    legend_list = {
        'Gender': {0: 'Male', 1: 'Female'},
        'Case_Status': {0: 'Closed', 1: 'To Be Investigated'},
        'Condition_of_the_Person_Involved': {1: 'Confined', 2: 'Temporary Released'},
        'Jurisdiction_Department_Name': {0: 'DECAP', 1: 'DEMACRO'},
        'Unfolding': {0: 'No further developments', 1: 'Further developments'},
        'Conduct': {0: 'No event', 1: 'Cargo', 2: 'Commercial', 3: 'Pedestrian', 4: 'Residence', 5: 'Transport',
                    6: 'Others'},
        'Person_Type': {1: 'Victim', 2: 'Suspect/Perpetrator', 3: 'Witness/Declarant', 4: 'Driver/Investigated',
                        5: 'Legal representation', 6: 'Child', 7: 'Missing/Found',
                        8: 'Parties involved'},
        'Race': {1: 'White', 2: 'Brown', 3: 'Black', 4: 'Others'},
        'Type Criminal Charge': {1: 'Theft', 2: 'Robbery', 3: 'Bodily Harm', 4: 'Homicide', 5: 'Simple Homicide',
                                 6: 'Drug Offenses', 7: 'Sexual Assault'},
        'Type Region': {1: 'Central São Paulo', 2: 'Northeast of Central São Paulo',
                        3: 'Northwest of Central São Paulo', 4: 'Southwest of Central São Paulo'},
        'Education_Level': {1: 'Complete Middle School', 2: 'Complete High School', 3: 'Complete Higher Education',
                            4: 'Incomplete Middle School',
                            5: 'Incomplete Higher Education', 6: 'Incomplete High School', 7: 'Illiterate'}
    }

    # EDA
    st.subheader('Exploratory Data Analysis')

    st.subheader('Doughnut Chart')
    # Column Distribution
    selected_column = st.selectbox('Select the column:',
                                   options=['Gender', 'Jurisdiction_Department_Name', 'Conduct', 'Person_Type', 'Race'])

    # Get legend for the selected column
    legend = legend_list[selected_column]

    # Generate the bar chart for the selected column
    plt.figure(figsize=(5, 5))
    sns.countplot(data=sampled_data, x=selected_column, palette="bright")
    plt.title(f'Distribution of {selected_column}')
    plt.xlabel(selected_column)
    plt.ylabel('Count')

    # Add legend to the plot
    plt.legend(labels=legend.values(), loc='upper right')

    # Display the plot
    st.pyplot(plt)

    # Doughnut Chart

    st.subheader('Doughnut Chart')
    selected_feature_doughnut = st.selectbox('Select the feature:',
                                             options=['Gender', 'Jurisdiction_Department_Name', 'Conduct',
                                                      'Person_Type', 'Race'])

    # Get legend for the selected feature
    legend = legend_list[selected_feature_doughnut]

    # Get the value counts for the selected feature
    value_counts_doughnut = sampled_data[selected_feature_doughnut].value_counts()

    # Generate the donut chart for the selected feature
    plt.figure(figsize=(8, 6))
    plt.pie(value_counts_doughnut, labels=value_counts_doughnut.index, startangle=90, radius=1, autopct='%1.1f%%',
            pctdistance=0.85, colors=plt.cm.tab10.colors)

    # Add legend to the plot
    legend_labels = [f'{legend[key]}' for key in value_counts_doughnut.index]
    plt.legend(labels=legend_labels, loc='upper right')

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Add inner circle (donut hole)
    centre_circle = plt.Circle((0, 0), 0.7, fc='white')
    plt.gca().add_artist(centre_circle)

    # Display the plot
    st.pyplot(plt)

    # 3D Histogram Distribution
    st.subheader('3D Histogram Distribution')
    selected_feature_x_3d = st.selectbox('Select the first feature:', options=['Gender'])
    selected_feature_y_3d = st.selectbox('Select the second feature:',
                                         options=['Education_Level', 'Type Criminal Charge', 'Person_Type', 'Race'])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x_3d = sampled_data[selected_feature_x_3d]
    y_3d = sampled_data[selected_feature_y_3d]
    hist_3d, xedges_3d, yedges_3d = np.histogram2d(x_3d, y_3d, bins=20)
    xpos_3d, ypos_3d = np.meshgrid(xedges_3d[:-1] + 0.25, yedges_3d[:-1] + 0.25, indexing="ij")
    xpos_3d = xpos_3d.ravel()
    ypos_3d = ypos_3d.ravel()
    zpos_3d = 0
    dx_3d = dy_3d = 0.5 * np.ones_like(zpos_3d)
    dz_3d = hist_3d.ravel()
    colors_3d = plt.cm.viridis((dz_3d - dz_3d.min()) / dz_3d.ptp())
    ax.bar3d(xpos_3d, ypos_3d, zpos_3d, dx_3d, dy_3d, dz_3d, zsort='average', alpha=0.8, color=colors_3d)
    ax.set_xlabel(selected_feature_x_3d)
    ax.set_ylabel(selected_feature_y_3d)
    ax.set_zlabel('Count')
    ax.set_title('3D Histogram')
    ax.view_init(elev=20, azim=30)
    st.pyplot(plt)

    # Geographic Chart
    st.subheader('Geographic Chart')

    def generate_geographic_chart(df):
        # Calculate the mean latitude and longitude of the sampled data
        mean_lat = sampled_data_geo['LATITUDE'].mean()
        mean_lon = sampled_data_geo['LONGITUDE'].mean()

        # Create a base map centered at the mean latitude and longitude
        map_brazil = folium.Map(location=[mean_lat, mean_lon], zoom_start=4)

        # Create a MarkerCluster object
        marker_cluster = MarkerCluster().add_to(map_brazil)

        # Add markers for each sampled data point to the MarkerCluster
        for lat, lon in zip(sampled_data_geo['LATITUDE'], sampled_data_geo['LONGITUDE']):
            folium.Marker([lat, lon]).add_to(marker_cluster)

        # Display the map
        return map_brazil

    # Sample a subset of data points
    sampled_data_geo = df.sample(n=10000, random_state=42)  # Use a fixed random_state for reproducibility

    # Call the function to generate the geographic chart
    geographic_chart = generate_geographic_chart(sampled_data_geo)

    # Display the geographic chart
    st.components.v1.html(geographic_chart._repr_html_(), width=700, height=500)

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
