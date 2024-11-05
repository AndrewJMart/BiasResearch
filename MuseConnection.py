from pylsl import StreamInlet, resolve_byprop
import csv
import time
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# Resolve EEG stream by type
streams = resolve_byprop('type', 'EEG', timeout=2)
if streams:
    print("Muse EEG stream found.")
    inlet = StreamInlet(streams[0])

    # Open a CSV file to record data
    with open('eeg_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "Channel 1", "Channel 2", "Channel 3", "Channel 4"])

        print("Recording EEG data...")
        start_time = time.time()
        while time.time() - start_time < 20:  # Record for 20 seconds
            sample, timestamp = inlet.pull_sample()
            writer.writerow([timestamp] + sample)
        print("Recording complete. Data saved to 'eeg_data.csv'.")
else:
    print("No EEG stream found. Ensure Muse is connected and streaming via BlueMuse.")

# Load the EEG data
eeg_data = pd.read_csv('./eeg_data.csv')

# Define filter parameters for each frequency band
BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 40)
}
fs = 256  # Sampling rate of Muse device

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

# Apply bandpass filter to each channel and plot
fig, axes = plt.subplots(len(BANDS), 1, figsize=(10, 10), sharex=True)

for i, (band, (low, high)) in enumerate(BANDS.items()):
    filtered_channels = {}
    for col in eeg_data.columns[1:]:  # Skip the timestamp column
        filtered_channels[col] = bandpass_filter(eeg_data[col], low, high, fs)
    
    # Plot the filtered signals for each band
    for j, (channel, filtered_data) in enumerate(filtered_channels.items()):
        axes[i].plot(filtered_data, label=f"{channel}")
    
    axes[i].set_title(f"{band} Band ({low}-{high} Hz)")
    axes[i].legend(loc="upper right")
    axes[i].set_ylabel("Amplitude (µV)")

axes[-1].set_xlabel("Sample Number")
plt.tight_layout()
plt.show()
