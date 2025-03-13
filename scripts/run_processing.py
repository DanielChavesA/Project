import os
import sys
import multiprocessing

# Ensure the script can find project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processinator.processinator import Processinator
from clustering.cluster import Clustinator
from visualizer.visualizer import Visualizer

def main():
    # Define input data directory
    input_directory = os.path.join(os.path.dirname(__file__), '..', 'data')
    processinator = Processinator(input_directory)

    # Step 1: Process all JSON files in the directory
    df = processinator.process_all_json_parallel()
    
    processinator.plot_missing_segments(df)

    # Step 2: Define clustering columns
    entity_column = "streetName"
    interval_column = "timeSet" 
    day_column = "day"  
    observation = "w_h_speed_avg"  
    n_clusters_list = [3,5,7]  
    
# Plot for all streets (averaged per day)
    processinator.plot_flow_density_per_day(df)

    # Step 3: Create clustering model and prepare data
    clustering_model = Clustinator(df, entity_column, interval_column, day_column, observation, n_clusters_list)
    matrix, days, unique_intervals, entities = clustering_model.prepare_data()

    # Step 4: Apply clustering
    cluster_results = clustering_model.fit(matrix, n_clusters_list)

    # Step 5: Visualize results
    visualizer = Visualizer()
    visualizer.generate_figures(n_clusters_list, matrix, cluster_results, days)

    # Step 6: Evaluate clustering
    scores = visualizer.internal_evaluation(n_clusters_list, matrix, cluster_results)
    visualizer.plot_clustering_scores(cluster_results, scores)

    # Step 7: Create MFD
    mfd, dens = visualizer.create_mfd(df)
    # Save DataFrame to a CSV file
    # Define the file path
    file_path = r"C:\Users\daca2\OneDrive - KTH\PhD\data\Söder_case_study\processed_data.csv"
    df.to_csv(file_path, index=False) 

    print("Processing and clustering completed successfully.")
    
    return df, matrix, cluster_results, mfd, dens

if __name__ == '__main__':
    multiprocessing.freeze_support()
    df, matrix, cluster_results, mfd, dens = main()