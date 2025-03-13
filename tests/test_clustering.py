import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from clustering.cluster import Clustering  

# Sample DataFrame for testing
def create_sample_dataframe():
    data = {
        "day": ["2019-09-16", "2019-09-16", "2019-09-17", "2019-09-17","2019-09-18","2019-09-18"],
        "streetName": ["Main St", "Main St", "Elm St", "Elm St","other","other"],
        "interval": [1, 2, 1, 2, 1, 2],
        "speed": [27, 40, 35, 45, 75, 90]
    }
    return pd.DataFrame(data)

# Test for preparing the data
def test_prepare_data():
    df = create_sample_dataframe()
    entity_column = "streetName"
    interval_column = "interval"
    day_column="day"
    observation="speed"
    clustering_model = Clustering(df, entity_column, interval_column,day_column,observation)
    matrix, days, unique_intervals, df = clustering_model.prepare_data()
    
    # Check matrix dimensions (should be 3 rows, 6 columns)
    assert matrix.shape == (3, 6), f"Expected matrix shape (3, 6), but got {matrix.shape}"
    
    # Check that the matrix does not contain NaN values
    assert np.all(np.isfinite(matrix)), "Matrix contains NaN values!"
    
    # Check if the unique intervals are correctly identified
    assert len(unique_intervals) == 2, f"Expected 2 unique intervals, but got {len(unique_intervals)}"

# Test for clustering
def test_clustering():
    df = create_sample_dataframe()
    entity_column = "streetName"
    interval_column = "interval"
    day_column="day"
    observation="speed"
    clustering_model = Clustering(df, entity_column, interval_column,day_column,observation)
    matrix, _, _, _ = clustering_model.prepare_data()
    
    # Test clustering with 2 clusters
    n_clusters_list = [3]
    cluster_results = clustering_model.fit(matrix, n_clusters_list)
    
    # Print the cluster results
    print("Cluster results:", cluster_results)  # <-- Add this line
    print("Cluster result keys:", cluster_results.keys() if isinstance(cluster_results, dict) else "Not a dict")

    
    # Check if the clustering results have the correct number of clusters
    assert len(np.unique(cluster_results[3])) == 3, "Clustering didn't return the expected number of clusters"
    
    # Check the result type
    assert isinstance(cluster_results, dict), f"Expected dict, but got {type(cluster_results)}"
