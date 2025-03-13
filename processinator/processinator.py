import os
import pandas as pd
import orjson
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns


class Processinator:
    def __init__(self, input_directory):
        """Initialize with the folder containing JSON files."""
        self.input_directory = input_directory

    def extract_segment_data(self, json_data):
        """Extracts relevant data from JSON files."""
        segments = json_data.get("network", {}).get("segmentResults", [])
        extracted_data = []

        for segment in segments:
            street_name = segment.get("streetName")
            distance = segment.get("distance")
            speed_limit = segment.get("speedLimit")
            road_class = segment.get("frc")
            direction = "REV" if segment.get("segmentId", 1) < 0 else "FWD"

            date_ranges = json_data.get("dateRanges", [])
            if date_ranges:
                day = date_ranges[0].get("from", "").split("T")[0]  # Extract date (before 'T')

            for time_result in segment.get("segmentTimeResults", []):
                extracted_data.append({
                    "streetName": street_name,
                    "road_class": road_class,
                    "distance": distance,
                    "speedLimit": speed_limit,
                    "direction": direction,
                    "timeSet": time_result.get("timeSet"),
                    "hSpeed": time_result.get("harmonicAverageSpeed"),
                    "medianSpeed": time_result.get("medianSpeed"),
                    "averageSpeed": time_result.get("averageSpeed"),
                    "standardDeviationSpeed": time_result.get("standardDeviationSpeed"),
                    "averageTravelTime": time_result.get("averageTravelTime"),
                    "medianTravelTime": time_result.get("medianTravelTime"),
                    "travelTimeRatio": time_result.get("travelTimeRatio"),
                    "day": day,
                    "flow":time_result.get("sampleSize")
                })

        return extracted_data

    def aggregate_data(self, df):
        """Groups data by street name, direction, day, and timeSet, calculating weighted averages including travel time."""
        df_fil = df[~df["road_class"].isin([0,1,2,6,7])].copy()

        # Filter out invalid or outlier data
        df_fil = df_fil[
              (df_fil["averageSpeed"] > 0) & 
              (df_fil["averageTravelTime"] > 0) & 
              (df_fil["distance"] > 0)
          ]
        
        df_fil = df_fil[df_fil["streetName"].notna()] #handle none/Nan values
        
        # Filter out streets with names ending with "avfarten" or "påfarten"
        df_fil = df_fil[
                ~df_fil["streetName"].str.endswith("avfarten") & 
                ~df_fil["streetName"].str.endswith("påfarten")
                  ]
        
        # Adding weighted calculations
        df_fil["weighted_speed"] = df_fil["averageSpeed"] * df_fil["distance"]
        df_fil["w_h_speed"] = df_fil["hSpeed"] * df_fil["distance"]
        
    # Compute total distance per street across all times to find max distance
        max_distance_df = df_fil.groupby(["streetName","direction", "day", "timeSet",]).agg(total_distance=("distance", "sum")).reset_index()
    
    # Group by streetName, direction, day, and timeSet for aggregation
        grouped_df = df_fil.groupby(["streetName", "direction", "day", "timeSet"]).agg(
        weighted_avg_speed=("weighted_speed", "sum"),
        w_h_speed=("w_h_speed", "sum"),
        total_travel_time=("averageTravelTime", "sum"),
        flow=("flow", "sum"),
        speed_limit=("speedLimit", "mean"),
    ).reset_index()
    
# Merge correctly to maintain per-direction distance
        grouped_df = grouped_df.merge(max_distance_df, on=["streetName", "direction", "day", "timeSet"], how="left")
        
        # Define the number of lanes for known streets
        lanes_dict = {
    "Götgatan": 2,
    "Hornsgatan": 2,
    "Hornsgatsavfarten": 2,
    "Hornsgatspåfarten": 2,
    "Katarinavägen": 2,
    "Renstiernas gata": 2,
    "Ringvägen": 2,
    "Västerbron": 2
}
        
# Assign number of lanes to each row in the DataFrame
# If a street is not in the dictionary, default to 1 lane
        grouped_df["n_lanes"] = grouped_df["streetName"].map(lanes_dict).fillna(1)

        # Calculate weighted average speed for each group
        grouped_df["weighted_avg_speed"] /= grouped_df["total_distance"]
        # Calculate weighted average speed for each group
        grouped_df["w_h_speed"] /= grouped_df["total_distance"]        
        # Calculate density (vehicles per lane-km)
        grouped_df["density"]= grouped_df["flow"] / (grouped_df["w_h_speed"]*grouped_df["n_lanes"])

        # Now calculate the average speed and travel time between directions for each street, day, and timeSet
        # Calculate average speed and travel time for each day, street, and timeSet, disregarding direction
        avg_speed_between_directions = grouped_df.groupby(["streetName", "day", "timeSet"])["weighted_avg_speed"].mean().reset_index()
        avg_hspeed_between_directions = grouped_df.groupby(["streetName", "day", "timeSet"])["w_h_speed"].mean().reset_index()
        avg_travel_time_between_directions = grouped_df.groupby(["streetName", "day", "timeSet"])["total_travel_time"].mean().reset_index()
        avg_density_between_directions = grouped_df.groupby(["streetName", "day", "timeSet"])["density"].mean().reset_index()
        avg_flow_between_directions = grouped_df.groupby(["streetName", "day", "timeSet"])["flow"].mean().reset_index()

        # Merge these averages back into the grouped dataframe
        grouped_df = grouped_df.merge(avg_speed_between_directions, on=["streetName", "day", "timeSet"], suffixes=("", "_avg"))
        grouped_df = grouped_df.merge(avg_hspeed_between_directions, on=["streetName", "day", "timeSet"], suffixes=("", "_avg"))
        grouped_df = grouped_df.merge(avg_travel_time_between_directions, on=["streetName", "day", "timeSet"], suffixes=("", "_avg"))
        grouped_df = grouped_df.merge(avg_density_between_directions, on=["streetName", "day", "timeSet"], suffixes=("", "_avg"))
        grouped_df = grouped_df.merge(avg_flow_between_directions, on=["streetName", "day", "timeSet"], suffixes=("", "_avg"))
        
        # Now drop the direction column to get only one row per street per day per timeSet
        final_df = grouped_df.drop(columns=["direction","total_travel_time","weighted_avg_speed","flow","density", "w_h_speed"],errors='ignore')

        # Drop duplicates to ensure only one row per street per day per timeSet
        final_df = final_df.drop_duplicates(subset=["streetName", "day", "timeSet"])

        return final_df

    def process_all_json(self):
        """Processes all JSON files in the directory incrementally and returns a combined DataFrame."""
        aggregated_results = []

        for filename in os.listdir(self.input_directory):
            if filename.endswith(".json"):
                filepath = os.path.join(self.input_directory, filename)
                with open(filepath, "rb") as file:
                    json_data = orjson.loads(file.read())  # Faster than json.load
                    extracted_data = self.extract_segment_data(json_data)
                    df = pd.DataFrame(extracted_data)

                    # Aggregate data for this day
                    aggregated_df = self.aggregate_data(df)
                    aggregated_results.append(aggregated_df)

        # Combine all aggregated results into a final dataframe
        df = pd.concat(aggregated_results, ignore_index=True)
        return df

    def process_json_file(self, filepath):
        """Helper function to process a single JSON file."""
        with open(filepath, "rb") as file:
            json_data = orjson.loads(file.read())  # Faster than json.load
            extracted_data = self.extract_segment_data(json_data)
            df = pd.DataFrame(extracted_data)
        return df

    def process_all_json_parallel(self):
        """Processes all JSON files in the directory incrementally and returns a combined DataFrame, using parallel processing."""
        # List all the JSON files in the directory
        filepaths = [os.path.join(self.input_directory, filename) for filename in os.listdir(self.input_directory) if filename.endswith(".json")]

        # Use multiprocessing to process files concurrently
        with multiprocessing.Pool() as pool:
            processed_data = pool.map(self.process_json_file, filepaths)

        # Combine all aggregated results into a final dataframe
        df = pd.concat(processed_data, ignore_index=True)
        
        # Apply aggregation after merging all data
        df = self.aggregate_data(df)
        # Print unique values of streetName after final filtering
        print("Unique values of streetName:")
        print(df["streetName"].unique())
        
        return df
    


    def plot_missing_segments(self, df):
        """Plots streets with missing data based on expected observations."""
        expected_entries_per_street = df["streetName"].value_counts().max()
        actual_counts = df["streetName"].value_counts()
        missing_counts = expected_entries_per_street - actual_counts
        missing_counts = missing_counts[missing_counts > 0]  # Keep only streets with missing data

        if missing_counts.empty:
            print("✅ No missing segments detected!")
            return

        plt.figure(figsize=(12, 6))
        missing_counts.plot(kind="bar", color="red", alpha=0.7)
        plt.xlabel("Street Name")
        plt.ylabel("Number of Missing Entries")
        plt.title("Missing Data (Segments) Per Street")
        plt.xticks(rotation=90)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()





    
    
    
    
    
    
    
    
    
    
    
        
    def process_Medy(self,file_path):
        """Processess Medy files"""
        return pd.read_csv(file_path, sep=",")
    
    def prepare_Medy(self,df):
        """explores and cleans"""

# Convert datetime columns to proper datetime objects
        df['date_time_'] = pd.to_datetime(df['date_time_'])
        df['date_tim_1'] = pd.to_datetime(df['date_tim_1'])

# Extract date from datetime for daily aggregation
        df['date'] = df['date_time_'].dt.date

# Calculate the mean of n_observat across all sensors and days
        mean_n_observat = df.groupby('date_time_')['n_observat'].mean().reset_index()
        mean_n_observat.rename(columns={'n_observat': 'mean_n_observat'}, inplace=True)
 
# Merge the mean back into the original dataframe
        df = df.merge(mean_n_observat, on='date_time_', how='left')

# Plotting
        plt.figure(figsize=(15, 8))

# Plot n_observat for each sensor_id
        for sensor_id, group in df.groupby('sensor_id'):
            plt.plot(group['date_time_'], group['n_observat'], label=f'Sensor {sensor_id}', alpha=0.5)

# Plot the mean n_observat across all sensors and days
            plt.plot(mean_n_observat['date_time_'], mean_n_observat['mean_n_observat'], 
               color='black', linewidth=2, label='Mean across all sensors')

# Customize the plot
            plt.title('n_observat Over Time for Each Sensor')
            plt.xlabel('Time')
            plt.ylabel('n_observat')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
            plt.grid(True)
            plt.tight_layout()

# Show the plot
            plt.show()
            
        # Get the full date range
        full_date_range = pd.date_range(start=df['date_time_'].min(), 
                               end=df['date_time_'].max(), 
                               freq='D')

# Create a DataFrame with the full date range
        full_dates_df = pd.DataFrame(full_date_range, columns=['date'])

# Get unique sensor_ids
        sensor_ids = df['sensor_id'].unique()

# Check for missing dates per sensor
        missing_data_summary = []

        for sensor_id in sensor_ids:
    # Filter data for the current sensor
            sensor_data = df[df['sensor_id'] == sensor_id]
    
    # Get unique dates for the current sensor
            sensor_dates = sensor_data['date'].unique()
    
    # Find missing dates
            missing_dates = full_dates_df[~full_dates_df['date'].isin(sensor_dates)]
    
    # Add to summary
            missing_data_summary.append({
        'sensor_id': sensor_id,
        'missing_dates': missing_dates['date'].tolist(),
        'num_missing_dates': len(missing_dates)
    })

# Convert summary to a DataFrame
        missing_data_summary_df = pd.DataFrame(missing_data_summary)

# Display the summary
        print(missing_data_summary_df)

# Create a heatmap to visualize missing dates
        plt.figure(figsize=(10, 6))
        for i, row in missing_data_summary_df.iterrows():
            if row['num_missing_dates'] > 0:
               plt.scatter(row['missing_dates'], [row['sensor_id']] * len(row['missing_dates']), 
                    color='red', alpha=0.5, label='Missing Dates' if i == 0 else "")

        plt.title('Missing Dates per Sensor')
        plt.xlabel('Date')
        plt.ylabel('Sensor ID')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return df

    def plot_flow_density_per_day(self,df):
        # Create a new "hour" column based on timeSet (Mapping: hour = timeSet + 4)
        df["hour"] = df["timeSet"].astype(int) + 4
    # Ensure 'timeSet' is sorted properly for time series plotting
        df["hour"] = pd.Categorical(df["hour"], ordered=True, categories=sorted(df["hour"].unique()))

    # Aggregate by timeSet and day (across all streets)
        df_grouped = df.groupby(["day", "hour"],observed=False)[["w_h_speed_avg","flow_avg", "density_avg"]].mean().reset_index()

    # Create Flow Plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_grouped, x="hour", y="flow_avg", hue="day", marker="o", palette="tab10")
        plt.title("Flow Over Time for All Streets (Grouped by Day)")
        plt.xlabel("Time of Day")
        plt.ylabel("Flow (vehicles/hour)")
        plt.xticks(rotation=45)
        plt.legend(title="Day", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.show()

    # Create Density Plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_grouped, x="hour", y="density_avg", hue="day", marker="s", palette="tab10", linestyle="dashed")
        plt.title("Density Over Time for All Streets (Grouped by Day)")
        plt.xlabel("Time of Day")
        plt.ylabel("Density (vehicles/km/lane)")
        plt.xticks(rotation=45)
        plt.legend(title="Day", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.show()

    # Create Speed Plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_grouped, x="hour", y="w_h_speed_avg", hue="day", marker="s", palette="tab10", linestyle="dashed")
        plt.title("Speed Over Time for All Streets (Grouped by Day)")
        plt.xlabel("Time of Day")
        plt.ylabel("Speed (km/h)")
        plt.xticks(rotation=45)
        plt.legend(title="Day", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.show()

        
        
        
        
        