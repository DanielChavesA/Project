import pytest
import pandas as pd
from tomtom_processor.processor import TomTomProcessor

#test data directory
TEST_DATA_DIR = "tests/test_data"

@pytest.fixture
def processor():
    return TomTomProcessor(TEST_DATA_DIR)

# Test if the processor initializes correctly
def test_processor_init(processor):
    assert processor.input_directory == TEST_DATA_DIR

# Test if JSON files are read correctly
def test_read_json(processor):
    df = processor.process_all_json()
    assert isinstance(df, pd.DataFrame)  # Ensure it returns a DataFrame
    assert not df.empty  # Ensure the DataFrame is not empty

# Test if travel time is aggregated correctly (assuming weighted average)
def test_travel_time_aggregation(processor):
    df = processor.process_all_json()
    grouped_df = df.groupby("streetName").agg({"averageTravelTime": "mean"})  # Example aggregation
    assert not grouped_df.empty
    assert "averageTravelTime" in grouped_df.columns
