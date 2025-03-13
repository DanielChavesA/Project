import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from dateutil.relativedelta import relativedelta
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class Visualizer:   
    def __init__(self):
    # Define color palettes
        self.weekend_colors = [
        '#67001f', '#d6604d', '#fdae61', '#f46d43', 
        '#d53e4f', '#9e0142', '#f768a1', '#f1c232',
        '#b30059', '#ff4d94', '#ff6666', '#e80073' 
    ]
    
        self.mixed_colors = [
        '#4d4d4d', '#35978f', '#bababa', '#878787',
        '#6b6b6b', '#33cccc', '#a3a3a3', '#999999' 
    ]
    
        self.weekday_colors = [
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', 
        '#cab2d6', '#6a3d9a', '#b15928', '#8dd3c7', '#bebada',
        '#fb8072', '#b3de69', '#bc80bd', '#fccde5', '#ccebc5',
        '#35978f', '#80cdc1', '#ffb3e6', '#ffff99', '#ffcc99', '#c2c2f0' 
    ]

    def assign_colors(self, n_clusters, days, assignments):
        days_colors = []
        color_to_cluster = []
        style_to_cluster = []
        cluster_id_weekdays_share = []
        cluster_id_weekend_share = []
        cluster_id_all_days = []

        for i in range(n_clusters):
            color_to_cluster.append(None)
            style_to_cluster.append(None)
            cluster_id_weekdays_share.append(0)
            cluster_id_weekend_share.append(0)
            cluster_id_all_days.append(0)

        for i in range(len(days)):
            if assignments[i] is not None:
                cluster_id_all_days[assignments[i]] += 1
                if '-' in str(days[i]):
                    pomT = datetime.datetime.strptime(str(days[i]), '%Y-%m-%d')
                else:
                    pomT = datetime.datetime.strptime(str(days[i]), '%Y%m%d')

                if int(pomT.weekday()) < 5:
                    cluster_id_weekdays_share[assignments[i]] += 1
                else:
                    cluster_id_weekend_share[assignments[i]] += 1

        print('cluster_id_weekdays_share', cluster_id_weekdays_share)
        print('cluster_id_weekend_share', cluster_id_weekend_share)

        for i in range(len(days)):
            if assignments[i] is not None:
                cluster_idx = assignments[i]
                if '-' in str(days[i]):
                    pomT = datetime.datetime.strptime(str(days[i]), '%Y-%m-%d')
                else:
                    pomT = datetime.datetime.strptime(str(days[i]), '%Y%m%d')

                if color_to_cluster[assignments[i]] is None:
                    if cluster_id_weekend_share[cluster_idx] / float(cluster_id_all_days[cluster_idx]) > 0.6:
                        color_to_cluster[assignments[i]] = self.weekend_colors.pop()
                        style_to_cluster[assignments[i]] = ':'
                    elif cluster_id_weekdays_share[cluster_idx] / float(cluster_id_all_days[cluster_idx]) > 0.6:
                        color_to_cluster[assignments[i]] = self.weekday_colors.pop(0)
                        style_to_cluster[assignments[i]] = '-'
                    else:
                        color_to_cluster[assignments[i]] = self.mixed_colors.pop()
                        style_to_cluster[assignments[i]] = ':'

                days_colors.append(color_to_cluster[assignments[i]])
            else:
                days_colors.append(None)

        return days_colors, color_to_cluster, style_to_cluster

    def calmap(self, ax, year, data, days, assignments, n_clusters, days_colors, color_to_cluster, limit_graphics=False):
        ax.tick_params('x', length=0, labelsize="medium", which='major')
        ax.tick_params('y', length=0, labelsize="x-small", which='major')

        # Month borders
        xticks, labels = [], []
        available_months = set(datetime.datetime.strptime(str(day), '%Y-%m-%d' if '-' in str(day) else '%Y%m%d').month for day in days)

        start = datetime.datetime(year, 1, 1).weekday()

        for month in sorted(available_months):
            first = datetime.datetime(year, month, 1)
            last = first + relativedelta(months=1, days=-1)

            y0 = first.weekday()
            y1 = last.weekday()
            x0 = (int(first.strftime("%j")) + start - 1) // 7
            x1 = (int(last.strftime("%j")) + start - 1) // 7

            P = [(x0, y0), (x0, 7), (x1, 7), (x1, y1 + 1), (x1 + 1, y1 + 1),
                 (x1 + 1, 0), (x0 + 1, 0), (x0 + 1, y0)]

            xticks.append(x0 + (x1 - x0 + 1) / 2)
            labels.append(first.strftime("%b"))
            poly = Polygon(P, edgecolor="black", facecolor="None", linewidth=1, zorder=20, clip_on=False)
            ax.add_artist(poly)

        line = Line2D([0, 53], [5, 5], linewidth=1, zorder=20, color="black", linestyle='dashed')
        ax.add_artist(line)

        if not limit_graphics:
            ax.set_xticks(xticks)
            ax.set_xticklabels(labels)
            ax.set_yticks(0.5 + np.arange(7))
            ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            ax.set_title("{}".format(year), weight="semibold")
        else:
            plt.tick_params(
                axis='x', which='both', bottom=False, top=False, labelbottom=False)
            plt.tick_params(
                axis='y', which='both', left=False, right=False, labelleft=False)

        # Clearing first and last day from the data
        valid = datetime.datetime(year, 1, 1).weekday()
        data[:valid, 0] = np.nan
        valid = datetime.datetime(year, 12, 31).weekday()
        data[valid + 1:, x1] = np.nan

        # Populate the diagram
        for i in range(len(days)):
            if '-' in str(days[i]):
                pomT = datetime.datetime.strptime(str(days[i]), '%Y-%m-%d')
            else:
                pomT = datetime.datetime.strptime(str(days[i]), '%Y%m%d')
            week_number = int(pomT.strftime("%W"))
            day_of_week = int(pomT.weekday())
            data[day_of_week, week_number] = assignments[i]

        act_date = datetime.datetime(year, 1, 1)
        while act_date.year == year:
            week_number = int(act_date.strftime("%W"))
            day_of_week = int(act_date.weekday())
            doy_id = act_date.timetuple().tm_yday
            if doy_id < 5 and week_number > 53:
                week_number = 0
            act_date = act_date + datetime.timedelta(days=1)

        cmap = ListedColormap(color_to_cluster)
        bounds = [-0.1]
        step = 1
        for i in range(n_clusters):
            bounds.append(i - 0.1 + step)
        norm = BoundaryNorm(bounds, cmap.N)

        ax.imshow(data, extent=[0, 53, 0, 7], zorder=10, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)

    def make_calendar_visualization_figure(self, days, assignments, n_clusters, years, days_colors, color_to_cluster, save_figure=None, show_figure=True, limit_graphics=False):
        fig = plt.figure(figsize=(8, 1.5 * len(years)), dpi=100)
        X = np.linspace(-1, 1, 53 * 7)

        for i, obj in enumerate(years):
            pom_s = str(len(years)) + '1' + str(i + 1)
            print(pom_s)

            ax = plt.subplot(int(pom_s), xlim=[34, 44], ylim=[0, 7], frameon=False, aspect=1)
            I = 1.2 - np.cos(X.ravel()) + np.random.normal(0, .2, X.size)
            I = I.reshape(53, 7).T
            I.fill(np.nan)
            self.calmap(ax, int(obj), I.reshape(53, 7).T, days, assignments, n_clusters, days_colors, color_to_cluster, limit_graphics)

        if save_figure:
            plt.savefig(save_figure)

        if show_figure or save_figure is None:
            plt.tight_layout()
            plt.show()

    def make_figure_centroids(self, x, y, color_to_cluster, style_to_cluster, cluster_ids, minY=None, maxY=None, save_figure=None, show_figure=True):
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        for i in range(len(x)):
            ax.plot(x[i], y[i], style_to_cluster[i], color=color_to_cluster[i], label=str(cluster_ids[i]))
        ax.set_xlabel('Time of day (hours)')
        ax.set_ylabel('Average Speed (km/h)')
        if minY is not None and maxY is not None:
            ax.set_ylim([minY, maxY])
        plt.legend()

        if save_figure:
            plt.savefig(save_figure)

        if show_figure or save_figure is None:
            plt.tight_layout()
            plt.show()

    def generate_figures(self, cluster_list, matrix, cluster_results, days):
        x_axis_hours = np.arange(6, 22, 1, dtype=int).tolist()  # 20-hour intervals

        centroids = {}
        centroids_yy_daytypes_dict = {}

        for n_clusters_t in cluster_list:
            centroids_xx = []
            centroids_yy_daytypes = []
            cluster_ids = []

            for i in range(n_clusters_t):
                centroids_xx.append(x_axis_hours)
                centroid_yy = np.nanmean(
                    matrix[np.where(cluster_results[n_clusters_t] == i)[0], 6:22],
                    axis=0
                ).transpose()
                centroids_yy_daytypes.append(centroid_yy.flatten())
                cluster_ids.append(i)

                centroids[n_clusters_t] = {
                    "x": centroids_xx,
                    "y": centroids_yy_daytypes,
                    "cluster_ids": cluster_ids
                }

                centroids_yy_daytypes_dict[n_clusters_t] = list(centroids_yy_daytypes)

            days_colors, color_to_cluster, style_to_cluster = self.assign_colors(n_clusters_t, days, cluster_results[n_clusters_t])
            self.make_figure_centroids(centroids_xx, centroids_yy_daytypes, color_to_cluster, style_to_cluster, cluster_ids)
            self.make_calendar_visualization_figure(days, cluster_results[n_clusters_t], n_clusters_t, [2019], days_colors, color_to_cluster, save_figure=None)

        return centroids, centroids_yy_daytypes_dict

    def internal_evaluation(self, cluster_list, matrix, cluster_results):
        """
        Computes internal evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) for clustering results.

        Parameters:
        cluster_list (list): List of cluster numbers to evaluate.
        matrix (np.array): Data matrix used for clustering.
        cluster_results (dict): Dictionary where keys are cluster numbers and values are clustering results.

        Returns:
        scores (dict): Dictionary containing evaluation metrics for each cluster number.
        """
        scores = {
            'Silhouette Score': {},
            'Davies-Bouldin Score': {},
            'Calinski-Harabasz Score': {}
        }

        for n_clusters in cluster_list:
            labels = cluster_results[n_clusters]

            # Calculate evaluation metrics
            SC_score = silhouette_score(matrix, labels)
            DB_score = davies_bouldin_score(matrix, labels)
            CH_score = calinski_harabasz_score(matrix, labels)

            # Store the scores in separate dictionaries
            scores['Silhouette Score'][n_clusters] = SC_score
            scores['Davies-Bouldin Score'][n_clusters] = DB_score
            scores['Calinski-Harabasz Score'][n_clusters] = CH_score

            # Print the computed cluster quality scores
            print(f'Cluster count: {n_clusters}')
            print('Silhouette Score:', SC_score)
            print('Davies-Bouldin Score:', DB_score)
            print('Calinski-Harabasz Score:', CH_score)
            print('-' * 40)

        return scores

    def plot_clustering_scores(self, cluster_results, scores):
        """
        Plots the clustering evaluation scores (Silhouette, Davies-Bouldin, Calinski-Harabasz) for different cluster counts.

        Parameters:
        cluster_results (dict): Dictionary where keys are cluster numbers and values are clustering results.
        scores (dict): Dictionary containing three dictionaries for each evaluation metric.
        """
        clusters = sorted(cluster_results.keys())

        SC_values = [scores['Silhouette Score'][k] for k in clusters]
        #DB_values = [scores['Davies-Bouldin Score'][k] for k in clusters]
        #CH_values = [scores['Calinski-Harabasz Score'][k] for k in clusters]

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Plot the three scores
        ax1.plot(clusters, SC_values, marker='o', linestyle='-', color='b', label='Silhouette Score')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

      #  ax2 = ax1.twinx()
       # ax2.plot(clusters, DB_values, marker='s', linestyle='--', color='r', label='Davies-Bouldin Score')
        #ax2.plot(clusters, CH_values, marker='^', linestyle='-.', color='g', label='Calinski-Harabasz Score')
        #ax2.set_ylabel('Score Values')

        # Add legends
        ax1.legend(loc='upper left')
        #ax2.legend(loc='upper right')

        plt.title('Clustering Evaluation Metrics')
        plt.grid()
        plt.show()
        
    def create_mfd(self, aggregated_data):
      """
    Creates a Macroscopic Fundamental Diagram (MFD) from aggregated traffic data.

    Parameters:
        aggregated_data (pd.DataFrame): Aggregated data containing 'relative_density' and 'relative_flow'.

    Returns:
        None (displays the MFD plot).
    """
    # Ensure the required columns are present
      if 'density_avg' not in aggregated_data.columns or 'flow_avg' not in aggregated_data.columns:
          raise ValueError("Input data must contain 'relative_density' and 'relative_flow' columns.")
     
        # Create a new "hour" column based on timeSet (Mapping: hour = timeSet + 4)
      aggregated_data["hour"] = aggregated_data["timeSet"].astype(int) + 4
    # Initialize a list to store results for each hour
      hourly_results = []

    # Group data by hour
      grouped_data = aggregated_data.groupby(["day", "hour"]) 

    # Iterate over each hour group
      for (day,hour), group in grouped_data:
        # Calculate total network length (L_net) for this hour
          L_net = group["total_distance"].sum()

        # Calculate weighted flow and density for this hour
          weighted_flow = group["flow_avg"] * group["total_distance"]
          weighted_density = group["density_avg"] * group["total_distance"]

        # Calculate network-aggregated flow (Q_delta_T) and density (K_delta_T) for this hour
          Q_delta_T = weighted_flow.sum() / L_net
          K_delta_T = weighted_density.sum() / L_net

        # Append results for this hour to the list
          hourly_results.append({
            "day":day,
            "hour": hour,
            "Q_delta_T": Q_delta_T,
            "K_delta_T": K_delta_T
          })

    # Convert the list of results to a DataFrame
      h_results_df = pd.DataFrame(hourly_results)
      
     # Plot the MFD for each day using different colors
      plt.figure(figsize=(10, 6))
      sns.scatterplot(
      x="K_delta_T", 
      y="Q_delta_T", 
      hue="day",  # Color by day
      data=h_results_df, 
      alpha=0.6, 
      edgecolor=None
)

# Fit a trendline for each day (optional)
      for day in h_results_df["day"].unique():
          subset = h_results_df[h_results_df["day"] == day]
          if len(subset) > 1:  # Avoid fitting for a single point
              z = np.polyfit(subset["K_delta_T"], subset["Q_delta_T"], 2)
              p = np.poly1d(z)
              x_values = np.linspace(subset["K_delta_T"].min(), subset["K_delta_T"].max(), 100)
              plt.plot(x_values, p(x_values), label=f"Trendline {day}")

# Add labels and title
      plt.xlabel("Density (vehicles/lane-km)")
      plt.ylabel("Flow (vehicles/hour)")
      plt.title("Macroscopic Fundamental Diagram (MFD) - Per Day")
      plt.legend()
      plt.grid(True)
      plt.show()
  
    # Filter out invalid or outlier data
      filtered_data = aggregated_data[
          (aggregated_data["density_avg"] > 0) & 
          (aggregated_data["flow_avg"] > 0)
      ]
      
      print(filtered_data.columns)
      
      # Calculate weighted flow
      filtered_data["weighted_flow"] = filtered_data["flow_avg"] * filtered_data["total_distance"]
# Calculate weighted density
      filtered_data["weighted_density"] = filtered_data["density_avg"] * filtered_data["total_distance"]
  
    # Group by hour and calculate the mean flow, density, speeds, production, and accumulation
      grouped_df = filtered_data.groupby(["timeSet", "hour"]).agg(
        total_distance=("total_distance", "sum"),
        total_weighted_flow=("weighted_flow", "sum"),
        total_weighted_density=("weighted_density", "sum"),
        h_speed=("w_h_speed_avg", "mean"),
        avg_speed=("weighted_avg_speed_avg", "mean")
    ).reset_index()

      # Calculate network average flow and density
      grouped_df["avg_w_flow"] = grouped_df["total_weighted_flow"] / grouped_df["total_distance"]
      grouped_df["avg_w_density"] = grouped_df["total_weighted_density"] / grouped_df["total_distance"]
      
      # Calculate production (P_delta_T) (veh.m/s) and accumulation (n_delta_T) (veh)
      grouped_df["production"] = (grouped_df["total_distance"]) * (grouped_df["avg_w_flow"]/3600)
      grouped_df["accumulation"] = (grouped_df["total_distance"]/1000) * grouped_df["avg_w_density"]

      # Calculate space-mean speed (V_delta_T)
      grouped_df["space_mean_speed"] = grouped_df["avg_w_flow"] / grouped_df["avg_w_density"]
      
      # Group by hour and calculate the mean flow and density
      filtered_data1=grouped_df
      
    # Create the MFD plot
      plt.figure(figsize=(10, 6))
      sns.scatterplot(
          x="avg_w_density", 
          y="avg_w_flow", 
          data=filtered_data1, 
          alpha=0.6, 
          edgecolor=None
      )
      
      filtered_data1 = filtered_data1.dropna(subset=["avg_w_density", "avg_w_flow"])
      filtered_data1 = filtered_data1[(~filtered_data1["avg_w_density"].isin([np.inf, -np.inf])) & 
                                (~filtered_data1["avg_w_flow"].isin([np.inf, -np.inf]))]

      
      # Add a trendline (2nd-degree polynomial fit)
      z = np.polyfit(filtered_data1["avg_w_density"], filtered_data1["avg_w_flow"], 2)
      p = np.poly1d(z)
      x_values = np.linspace(filtered_data1["avg_w_density"].min(), filtered_data1["avg_w_density"].max(), 100)
      plt.plot(x_values, p(x_values), color="red", label="Trendline")

    # Annotate points with hour values
      for i, row in filtered_data1.iterrows():
          plt.text(row["avg_w_density"], row["avg_w_flow"], str(row["hour"]), 
                   fontsize=10, ha='right', va='bottom', color='black')
          
    # Add labels and title
      plt.xlabel("Density (vehicles/lane-km)")
      plt.ylabel("Flow (vehicles/hour)")
      plt.title("Macroscopic Fundamental Diagram (MFD)")
      plt.legend()
      plt.grid(True)
      plt.show()
      
      # Plot 1: Production vs. Accumulation
      plt.figure(figsize=(10, 6))
      sns.scatterplot(
        x="accumulation", 
        y="production", 
        data=filtered_data1, 
        alpha=0.6, 
        edgecolor=None
    )

    # Annotate points with hour values
      for i, row in filtered_data1.iterrows():
          plt.text(row["accumulation"], row["production"], str(row["hour"]), 
                   fontsize=10, ha='right', va='bottom', color='black')

    # Add labels and title
      plt.xlabel("Accumulation (veh)")
      plt.ylabel("Production (veh.m/s)")
      plt.title("Production vs. Accumulation")
      plt.grid(True)
      plt.show()

    # Plot 2: Space-Mean Speed vs. Accumulation
      plt.figure(figsize=(10, 6))
      sns.scatterplot(
          x="accumulation", 
          y="h_speed", 
          data=filtered_data1, 
          alpha=0.6, 
          edgecolor=None
      )
      
      # Add a trendline (2nd-degree polynomial fit)
      z = np.polyfit(filtered_data1["accumulation"], filtered_data1["h_speed"], 3)
      p = np.poly1d(z)
      x_values = np.linspace(filtered_data1["accumulation"].min(), filtered_data1["accumulation"].max(), 100)
      plt.plot(x_values, p(x_values), color="red", label="Trendline")


    # Annotate points with hour values
      # Annotate points with hour values and adjust positioning
      for i, row in filtered_data1.iterrows():
          plt.text(row["accumulation"], row["h_speed"], str(row["hour"]),
             fontsize=10, ha='center', va='center', color='black', 
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle="round,pad=0.2"))


    # Add labels and title
      plt.xlabel("Accumulation (veh)")
      plt.ylabel("Space-Mean Speed (km/h)")
      plt.title("Space-Mean Speed vs. Accumulation")
      plt.grid(True)
      plt.show()
      
      # Plot 4: Space-Mean Speed vs. Accumulation
      plt.figure(figsize=(10, 6))
      sns.scatterplot(
            x="avg_w_density", 
            y="h_speed", 
            data=filtered_data1, 
            alpha=0.6, 
            edgecolor=None
        )
      
      # Add a trendline (2nd-degree polynomial fit)
      z = np.polyfit(filtered_data1["avg_w_density"], filtered_data1["h_speed"], 3)
      p = np.poly1d(z)
      x_values = np.linspace(filtered_data1["avg_w_density"].min(), filtered_data1["avg_w_density"].max(), 100)
      plt.plot(x_values, p(x_values), color="red", label="Trendline")

      # Annotate points with hour values
      for i, row in filtered_data1.iterrows():
            plt.text(row["avg_w_density"], row["h_speed"], str(row["hour"]), 
                     fontsize=10, ha='right', va='bottom', color='black')

      # Add labels and title
      plt.xlabel("Density (veh/lane-km)")
      plt.ylabel("Harmonic Averge Speed (km/h)")
      plt.title("Harmonic Average Speed vs. Density")
      plt.grid(True)
      plt.show()
      
      return filtered_data1, h_results_df

