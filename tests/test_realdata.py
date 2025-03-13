import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from tomtom_processor.processor import TomTomProcessor  
from clustering.cluster import Clustering
from visualizer.visualizer import Visualizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Initialize your processor class with the input directory
input_directory = r"C:\Users\daca2\python-course\Project\data" 
processor = TomTomProcessor(input_directory)

# Step 1: Process all JSON files in the directory
df = processor.process_all_json()

# Step 4: Define clustering columns
entity_column = "streetName"
interval_column = "timeSet" 
day_column = "day"  
observation = "weighted_avg_speed"  

# Step 5: Create clustering model and prepare data
clustering_model = Clustering(df, entity_column, interval_column, day_column, observation)
matrix, days,unique_intervals, entities = clustering_model.prepare_data()

# Step 6: Apply clustering
n_clusters_list= [2,3]  
cluster_results = clustering_model.fit(matrix, n_clusters_list)
#%%

visualizer = Visualizer()

# Generate figures for all cluster numbers in n_clusters_list
visualizer.generate_figures(n_clusters_list, matrix, cluster_results, days)

scores = visualizer.internal_evaluation(n_clusters_list, matrix, cluster_results)
visualizer.plot_clustering_scores(cluster_results, scores)

