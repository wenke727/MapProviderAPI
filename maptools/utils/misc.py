import json
import platform
import pandas as pd
import platform
import matplotlib.pyplot as plt

def detect_os():
    os_name = platform.system()
    if os_name == "Darwin":
        return "Mac"
    elif os_name == "Windows":
        return "Windows"
    elif os_name == "Linux":
        return "Linux"
    else:
        return "Unknown OS"

def read_json_file(file_path):
    """
    Reads a JSON file and returns the data as a Python dictionary.

    Parameters:
        - file_path: A string representing the path to the JSON file.

    Returns:
        A dictionary containing the data from the JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def convert_timestamps(geodataframe, timestamp_column, unit='ms', timezone='Asia/Shanghai', strftime=None):
    """
    Convert a column of timestamps in a GeoDataFrame from specified unit since Unix epoch to
    datetime objects in a specified timezone and optionally format them as strings.

    Parameters:
    - geodataframe (GeoDataFrame): The GeoDataFrame containing the timestamp column.
    - timestamp_column (str): The name of the column with the timestamps.
    - unit (str): The unit of the input timestamps (e.g., 's' for seconds, 'ms' for milliseconds).
    - timezone (str): The timezone to convert the datetime objects to.
    - strftime (str, optional): The format string to convert datetime objects to strings.
                                If None, no string conversion is performed.

    Returns:
    GeoDataFrame: The GeoDataFrame with the converted datetime column.
                  If strftime is not None, an additional 'strftime' column is added
                  with formatted datetime strings.
    """
    # Convert to datetime, with specified unit
    geodataframe[timestamp_column] = pd.to_datetime(geodataframe[timestamp_column], unit=unit)

    # Convert timezone
    # Convert to datetime, with specified unit
    geodataframe[timestamp_column] = pd.to_datetime(geodataframe[timestamp_column], unit=unit)

    # Localize the timestamp to UTC and then convert to the desired timezone
    geodataframe[timestamp_column] = geodataframe[timestamp_column].dt.tz_localize('UTC').dt.tz_convert(timezone)

    # If strftime is provided, format datetime objects as strings and store in a new column
    if strftime is not None:
        geodataframe['strftime'] = geodataframe[timestamp_column].dt.strftime(strftime)

    return geodataframe


def set_chinese_font_style():
    # Detect the operating system
    system_name = platform.system()

    # Set font for Windows
    if system_name == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei']
    # Set font for Linux
    elif system_name == 'Linux':
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    # Set font for Mac
    elif system_name == 'Darwin':  # Darwin is the name of macOS's underlying system
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

    # Set the property to display the minus sign correctly
    plt.rcParams['axes.unicode_minus'] = False


if __name__ == "__main__":
    # Create a test GeoDataFrame
    from shapely import Point
    import geopandas as gpd
    
    data = {
        'timestamp': [1652396665466, 1704640862000], # example timestamps in milliseconds
        'geometry': [Point(1, 1), Point(2, 2)]
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    # Convert timestamps
    gdf_converted = convert_timestamps(gdf, 'timestamp', unit='ms', timezone='Asia/Shanghai', strftime='%Y-%m-%d %H:%M:%S')

    gdf_converted.head()

# %%
