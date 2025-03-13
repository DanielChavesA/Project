import os
import json
import pandas as pd
import orjson

class TomTomProcessor:
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
                    "harmonicAverageSpeed": time_result.get("harmonicAverageSpeed"),
                    "medianSpeed": time_result.get("medianSpeed"),
                    "averageSpeed": time_result.get("averageSpeed"),
                    "standardDeviationSpeed": time_result.get("standardDeviationSpeed"),
                    "averageTravelTime": time_result.get("averageTravelTime"),
                    "medianTravelTime": time_result.get("medianTravelTime"),
                    "travelTimeRatio": time_result.get("travelTimeRatio"),
                    "day": day
                })

        return extracted_data

    def aggregate_data(self, df):
        """Groups data by street name, direction, day, and timeSet, calculating weighted averages including travel time."""
        df_fil = df[~df["road_class"].isin([0,1,2,6, 7])].copy()
        
        # Adding weighted calculations
        df_fil["weighted_speed"] = df_fil["averageSpeed"] * df_fil["distance"]
        
        # Group by streetName, direction, day, and timeSet for aggregation
        grouped_df = df_fil.groupby(["streetName", "direction", "day", "timeSet"]).agg(
            total_distance=("distance", "sum"),
            weighted_avg_speed=("weighted_speed", "sum"),
            total_travel_time=("averageTravelTime", "sum"),
            speed_limit=("speedLimit", "mean"),
        ).reset_index()

        # Calculate weighted average speed for each group
        grouped_df["weighted_avg_speed"] /= grouped_df["total_distance"]

        # Now calculate the average speed and travel time between directions for each street, day, and timeSet
        # Calculate average speed and travel time for each day, street, and timeSet, disregarding direction
        avg_speed_between_directions = grouped_df.groupby(["streetName", "day", "timeSet"])["weighted_avg_speed"].mean().reset_index()
        avg_travel_time_between_directions = grouped_df.groupby(["streetName", "day", "timeSet"])["total_travel_time"].mean().reset_index()

        # Merge these averages back into the grouped dataframe
        # First, merge the speed
        grouped_df = grouped_df.merge(avg_speed_between_directions, on=["streetName", "day", "timeSet"], suffixes=("", "_avg"))

        # Then, merge the travel time
        grouped_df = grouped_df.merge(avg_travel_time_between_directions, on=["streetName", "day", "timeSet"], suffixes=("", "_avg"))

        # Now drop the direction column to get only one row per street per day per timeSet
        final_df = grouped_df.drop(columns=["direction","total_travel_time","weighted_avg_speed"])

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

        
        
        