import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from dateutil.relativedelta import relativedelta
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

