import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.tree import DecisionTreeClassifier
# Adding to imports
from river import tree, preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

# Step 1: Define file path for dataset
# Directory containing data file received from silk after convertinging it
path = "./assets/csv"  # Specify local directory to fetch the data
# Combining directory and file name to creating complete path
data_file = os.path.join(path, "tcp_data.csv")

# Step 2: Loading data into DataFrame
# Reading data from CSV file using pipe '|' as separator
# Strip extra spaces from column names for consistency
data = pd.read_csv(data_file, sep='|', skipinitialspace=True)
data.columns = data.columns.str.strip()

# Removing unwanted column if it exists
if 'Unnamed: 6' in data.columns:
    data = data.drop(columns=['Unnamed: 6'])

# Removing rows containing missing values
data = data.dropna()

# Converting start and end times to datetime format for calculations
data['sTime'] = pd.to_datetime(data['sTime'], errors='coerce')
data['eTime'] = pd.to_datetime(data['eTime'], errors='coerce')

# Compute session duration in seconds
data['duration_sec'] = (data['eTime'] - data['sTime']).dt.total_seconds()

# Filter rows where duration is greater than zero
data = data[data['duration_sec'] > 0]

# Calculate packets processed per second
data['packets_per_sec'] = data['packets'] / data['duration_sec']

# Removing rows with missing or invalid computed values and reset index
data = data.dropna().reset_index(drop=True)

# Categorize data into groups based on packets per second (PPS)
data['class'] = pd.cut(
    data['packets_per_sec'], bins=[0, 100, 500, 1000, float('inf')],
    labels=['Low', 'Medium', 'High', 'Anomalous']
)

# Step 3: Display raw and processed data
# Print first ten rows from raw dataset
print("Loaded Data (First 10 Records):")
print(data.head(10).to_string(index=False))

# Print first ten rows from processed dataset
print("\nProcessed Data (First 10 Records):")
print(data[['sTime', 'eTime', 'packets', 'duration_sec', 'packets_per_sec', 'class']].head(10).to_string(index=False))

# Step 4: Creating visualization function
def style_plot(ax, title, xlabel, ylabel):
    # Appling background color to plot
    ax.set_facecolor("#f9f9f9")
    # Adding gridlines for readability
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    # Set plot title and axis labels with proper styling
    ax.set_title(title, fontsize=14, weight='bold', color="#333333")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

# Using modern plotting style for uniformity
plt.style.use('ggplot')

# Creating figure with multiple subplots for different graphs
fig, axes = plt.subplots(5, 2, figsize=(20, 25))

# Graph 1: Histogram showing PPS distribution
ax1 = axes[0, 0]
ax1.hist(data['packets_per_sec'], bins=50, color='#4F81BD', edgecolor='black')
ax1.axvline(1000, color='red', linestyle='--', label='Anomaly Threshold')
style_plot(ax1, "Histogram of PPS", "Packets/Sec", "Frequency")
ax1.legend()

# Graph 2: Bar chart showing distribution of classes
ax2 = axes[0, 1]
class_counts = data['class'].value_counts()
ax2.bar(class_counts.index, class_counts.values, color=['#4CAF50', '#FF9800', '#F44336', '#9C27B0'])
style_plot(ax2, "Class Distribution", "Class", "Count")

# Graph 3: Scatter plot visualizing PPS vs. Duration
ax3 = axes[1, 0]
ax3.scatter(data['duration_sec'], data['packets_per_sec'], c='blue', alpha=0.5)
ax3.axhline(1000, color='red', linestyle='--', label='Anomaly Threshold')
style_plot(ax3, "PPS vs. Duration", "Duration (Sec)", "PPS")
ax3.legend()

# Graph 4: Trend of PPS over time
ax4 = axes[1, 1]
data_sorted = data.sort_values(by='sTime')
ax4.plot(data_sorted['sTime'], data_sorted['packets_per_sec'], color='#FF5722', linewidth=1.5)
style_plot(ax4, "PPS Trend", "Time", "PPS")
ax4.xaxis.set_major_formatter(FuncFormatter(lambda x, _: pd.to_datetime(x).strftime('%H:%M')))

# Graph 5: Boxplot showing distribution of PPS
ax5 = axes[2, 0]
ax5.boxplot(data['packets_per_sec'], vert=False, patch_artist=True, boxprops=dict(facecolor="#D9E6F2"))
style_plot(ax5, "Boxplot of PPS", "PPS", "")

# Graph 6: Histogram showing PPS distribution for each class
ax6 = axes[2, 1]
for label, subset in data.groupby('class'):
    ax6.hist(subset['packets_per_sec'], bins=20, alpha=0.5, label=str(label))
style_plot(ax6, "PPS per Class", "PPS", "Frequency")
ax6.legend()

# Graph 7: Histogram showing session duration distribution
ax7 = axes[3, 0]
ax7.hist(data['duration_sec'], bins=50, color='#9C27B0', edgecolor='black')
style_plot(ax7, "Duration Distribution", "Duration (Sec)", "Frequency")

# Graph 8: Violin plot visualizing PPS distribution
ax8 = axes[3, 1]
ax8.violinplot(data['packets_per_sec'], vert=False, showmeans=True)
style_plot(ax8, "Violin Plot of PPS", "PPS", "")

# Graph 9: Scatter plot comparing PPS and Packets
ax9 = axes[4, 0]
ax9.scatter(data['packets'], data['packets_per_sec'], c='green', alpha=0.6)
style_plot(ax9, "PPS vs Packets", "Packets", "PPS")

# Graph 10: Boxplot comparing PPS across different classes
ax10 = axes[4, 1]
ax10.boxplot([data[data['class'] == cls]['packets_per_sec'] for cls in data['class'].cat.categories], patch_artist=True)
ax10.set_xticklabels(data['class'].cat.categories)
style_plot(ax10, "PPS per Class", "Class", "PPS")

# Adjust layout, save graphs as image, and display
plt.tight_layout()
plt.savefig("tcp_analytics_graphs.png", dpi=300)
plt.show()

# Step 5: Save processed data and anomalies to CSV files
# Write processed data to CSV
data.to_csv("./assets/csv/processed_tcp_data.csv", index=False)

# Extract rows classified as Anomalous and save separately
anomalies = data[data['class'] == 'Anomalous']
anomalies.to_csv("./assets/csv/anomalous_records.csv", index=False)

# Notify user about saved outputs
print("\nGraphs saved. Outputs available as:")
print("- Processed data: 'processed_tcp_data.csv'")
print("- Anomalies: 'anomalous_records.csv'")


# Step 6: Classification using VFDT
scaler = preprocessing.StandardScaler()
vfdt_model = tree.HoeffdingTreeClassifier()

# Train VFDT model
for _, row in data.iterrows():
    X = {'packets_per_sec': row['packets_per_sec']}
    y = row['class']
    vfdt_model = vfdt_model.learn_one(X, y)

# Step 7: Classification using On-Demand Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
data['packets_cluster'] = kmeans.fit_predict(data[['packets_per_sec']])

# Map clusters to class labels
cluster_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
data['cluster_class'] = data['packets_cluster'].map(cluster_labels)

# Step 8: Retrieve and query classified data
low_medium_nodes = data[data['class'].isin(['Low', 'Medium'])]
print("\nLow and Medium Nodes:")
print(low_medium_nodes)

# Step 9: Anomaly Detection
anomaly_threshold = 1000  # Adjust threshold based on observation
anomalies = data[data['packets_per_sec'] > anomaly_threshold]
print("\nAnomalous Nodes:")
print(anomalies)

# Save anomaly data for further analysis
anomalies.to_csv("./assets/csv/anomalies_nodes.csv", index=False)

# Step 10: Evaluate Classifier Accuracy
y_true = data['class']
y_pred_vfdt = [vfdt_model.predict_one({'packets_per_sec': x}) for x in data['packets_per_sec']]
print("\nVFDT Classification Report:")
print(classification_report(y_true, y_pred_vfdt))

y_pred_kmeans = data['cluster_class']
print("\nKMeans Classification Report:")
print(classification_report(y_true, y_pred_kmeans))
