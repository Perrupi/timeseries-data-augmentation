import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from wgan_generator import *


# Modifiable Parameters
start_time = pd.Timestamp("2023-01-01 00:00:00")  # Replace with your desired start time
time_interval = 1  # Time step length (1s or more)
nominal_dataset_path = datasets_folder + 'nominal_dataset.csv'


# Create an empty DataFrame to store the time series data
combined_df = pd.DataFrame()

# Iterate over CSV files in the folder and load the data
for csv_file in glob.glob(os.path.join(synthetic_data_folder, '*.csv')):
    # Load each CSV file into a DataFrame and append it to the combined DataFrame
    df = pd.read_csv(csv_file, sep=';', decimal=',')
    df.iloc[:,0] = df.iloc[:,0].rolling(window=8).mean()  # Use rolling avg to denoise synthetic data
    combined_df = pd.concat([combined_df, df], axis=1)

# Create a date range for the index
end_time = start_time + pd.Timedelta(seconds=(sequence_length - 1) * time_interval)
time_index = pd.date_range(start=start_time, end=end_time, freq=f"{time_interval}S")

# Plotting the time series data
plt.figure(figsize=(12, 6))

# Plot multiple time series datasets in light blue
for i, col in enumerate(combined_df.columns):  # Exclude the last column for red plotting
    plt.plot(time_index, combined_df[col], color='#004080', alpha=0.01, solid_capstyle='butt')
plt.plot([], [], color='#004080', label='Data generated by the WGAN')

# Add plot of real nominal timeseries
real_data = pd.read_csv(nominal_dataset_path, index_col=None)
real_data = real_data[column_of_interest][:sequence_length]

# Plot one additional time series in red
plt.plot(time_index, real_data, color='red', label='Real cleaning timeseries')

# Add labels and legend
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Show the plot
plt.show()