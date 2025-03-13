import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Clustinator:
    def __init__(self, df, entity_column, interval_column,day_column, observation, n_clusters_list):
        """
        Initializes the Clustering object with data, entity column, and interval column.
        """
        self.df = df
        self.entity_column = entity_column
        self.interval_column = interval_column
        self.day_column = day_column
        self.observation = observation
        self.n_clusters_list = n_clusters_list
        self.matrix = None
        self.days = None
        self.unique_intervals = None
        self.cluster_results = {}
    
    def prepare_data(self):
        """
        Prepares the data matrix for clustering based on the provided entity and interval columns.
        """
        unique_intervals = self.df[self.interval_column].unique()
        unique_intervals.sort()
        n_intervals = len(unique_intervals)  # Determine number of intervals

        # First sort by calendar date, observation entity, and time interval
        self.df.sort_values([self.day_column, self.entity_column, self.interval_column], inplace=True)

        entities = np.unique(self.df[self.entity_column].values)
        days = np.unique(self.df[self.day_column].values)
        n_entities = len(entities)
        n_days = len(days)

        # Initialize the matrix with zeros, rows are the days and columns are the number of observation entities by the #of time intervals
        self.matrix = np.zeros((n_days, n_entities * n_intervals))
        self.matrix.fill(np.nan)

        # Group by day and observation entity
        day_entity_groups = self.df.groupby([self.day_column, self.entity_column])

        # Populate the matrix
        for i, day in enumerate(days):
            for j, interval in enumerate(unique_intervals):
                for k, entity in enumerate(entities):
                    if (day, entity) in day_entity_groups.groups:
                        df_t = day_entity_groups.get_group((day, entity))
                        row = df_t[df_t[self.interval_column] == interval]
                        if not row.empty:
                            self.matrix[i, j * n_entities + k] = row[self.observation].values[0]  # Using observation (speed or validations)
        
        # Replace NaNs with zeros
        self.matrix = np.nan_to_num(self.matrix, nan=0)
        
        # Store days and intervals for later use in visualization
        self.days = days
        self.unique_intervals = unique_intervals
        
        return self.matrix, days, unique_intervals, entities

    def fit(self,data,n_clusters_list):
        """
        Fits the clustering model for different numbers of clusters.
        """
        if self.matrix is None:
            raise ValueError("Data matrix is not prepared. Please call prepare_data() first.")
            
            # Step 1: Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(self.matrix)
    

        pca = PCA(n_components=2)  
        pca_data = pca.fit_transform(standardized_data)
        
        # Clustering step for different values of n_clusters
        for n_clusters in self.n_clusters_list:
            model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
            cluster_labels = model.fit_predict(pca_data)
            self.cluster_results[n_clusters] = cluster_labels
            
        return self.cluster_results
    
    def evaluate(self):
        """
        Evaluates the clustering using various metrics (silhouette, Davies-Bouldin, and Calinski-Harabasz).
        """
        if not self.cluster_results:
            raise ValueError("Clustering has not been performed. Please call fit() first.")
        
        evaluation_results = {}
        for n_clusters, labels in self.cluster_results.items():
            silhouette = silhouette_score(self.matrix, labels)
            db_index = davies_bouldin_score(self.matrix, labels)
            ch_score = calinski_harabasz_score(self.matrix, labels)
            evaluation_results[n_clusters] = {
                'silhouette': silhouette,
                'davies_bouldin': db_index,
                'calinski_harabasz': ch_score
            }
        return evaluation_results,n_clusters
    
    
    
    
    
