import os
import sys

# Ensure the script can find project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processinator.processinator import Processinator
from clustering.cluster import Clustinator
from visualizer.visualizer import Visualizer


processinator = Processinator(input_directory= r"C:\Users\daca2\python-course\Project\Medy\"")

df=processinator.process_Medy(file_path= r"C:\Users\daca2\python-course\Project\Medy\Medy.csv")

df_medy=processinator.prepare_Medy(df)

# Step 2: Define clustering columns
entity_column = "sensor_id"
interval_column = "interval" 
day_column = "CalendarDa"  
observation = "n_observat"  
n_clusters_list = [3,5,7]  

# Step 3: Create clustering model and prepare data
clustering_model = Clustinator(df_medy, entity_column, interval_column, day_column, observation, n_clusters_list)
matrix, days, unique_intervals, entities = clustering_model.prepare_data()

# Step 4: Apply clustering
cluster_results = clustering_model.fit(matrix, n_clusters_list)

# Step 5: Visualize results
visualizer = Visualizer()
visualizer.generate_figures(n_clusters_list, matrix, cluster_results, days)

# Step 6: Evaluate clustering
scores = visualizer.internal_evaluation(n_clusters_list, matrix, cluster_results)
visualizer.plot_clustering_scores(cluster_results, scores)
#%%
mfd,q=visualizer.create_mfd(df)


print("Processing and clustering completed successfully.")

