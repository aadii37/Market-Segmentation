import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, jaccard_score
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler





# STEP 3 
df = pd.read_csv("mcdonalds.csv")

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

seg_vars = df.iloc[:, :11]

seg_matrix = seg_vars.applymap(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)

print("Binary Segmentation Matrix Preview:")
print(seg_matrix.head())

descriptor_vars = df.iloc[:, 11:]

combined_df = pd.concat([seg_matrix, descriptor_vars], axis=1)


pca = PCA(n_components=2)  
principal_components = pca.fit_transform(seg_matrix)


pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Visualize PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA of Segmentation Variables")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
plt.grid(True)
plt.show()

print("Explained Variance Ratio:", pca.explained_variance_ratio_)




# STEP 4


# Basic Structure Checks
print("Shape:", seg_matrix.shape)
print("Columns:", seg_matrix.columns.tolist())
print("Preview:\n", seg_matrix.head())
print("Data types:\n", seg_matrix.dtypes)

# Missing Value Check
missing_counts = seg_matrix.isnull().sum()
print("Missing values:\n", missing_counts)

# Summary Statistics
summary = seg_matrix.describe()
print("Summary Statistics:\n", summary)

# Plot mean values to understand dominant perceptions
seg_means = seg_matrix.mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=seg_means.index, y=seg_means.values, palette='viridis')
plt.xticks(rotation=45)
plt.ylabel("Proportion of 'YES' responses")
plt.title("Distribution of YES responses per perception")
plt.grid(True)
plt.tight_layout()
plt.show()

#Correlation Matrix
corr = seg_matrix.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Segmentation Variables")
plt.tight_layout()
plt.show()

#  Standardization (for PCA/clustering)
scaler = StandardScaler()
seg_matrix_scaled = scaler.fit_transform(seg_matrix)


print("Means (should be ~0):", np.mean(seg_matrix_scaled, axis=0))
print("Stds (should be ~1):", np.std(seg_matrix_scaled, axis=0))



# STEP 5

inertia = []
silhouette_scores = []

k_range = range(2, 9)

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(seg_matrix_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(seg_matrix_scaled, kmeans.labels_))

# Plot Scree plot (Inertia)
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Scree Plot (Inertia)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares')
plt.grid(True)
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(8, 4))
plt.plot(k_range, silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Scores for k-means')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

k_final = 4
kmeans_final = KMeans(n_clusters=k_final, n_init=10, random_state=42)
kmeans_final.fit(seg_matrix_scaled)
cluster_labels = kmeans_final.labels_

# Add cluster labels to original data
df['Cluster'] = cluster_labels
print("Cluster counts:\n", df['Cluster'].value_counts())


def compute_segment_stability(data, n_clusters=4, n_boot=20):
    # Fit base model
    base_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    base_model.fit(data)
    base_labels = base_model.labels_

    segment_stability = np.zeros(n_clusters)

    # For each cluster index
    for cluster_id in range(n_clusters):
        stability_scores = []

        for _ in range(n_boot):
            # Bootstrap sample
            boot_sample = resample(data)
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            model.fit(boot_sample)
            new_labels = model.predict(data)

            # For each point, get Jaccard similarity between cluster_id assignments
            base_cluster_mask = (base_labels == cluster_id).astype(int)
            new_cluster_mask = (new_labels == cluster_id).astype(int)

            # Jaccard similarity between cluster memberships
            jac = jaccard_score(base_cluster_mask, new_cluster_mask)
            stability_scores.append(jac)

        segment_stability[cluster_id] = np.mean(stability_scores)

    return segment_stability

# Run for final cluster count
segment_scores = compute_segment_stability(seg_matrix_scaled, n_clusters=4, n_boot=30)

# Plot Bar chart of stability per segment
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(segment_scores)+1), segment_scores, color='dodgerblue')
plt.ylim(0, 1)
plt.xlabel("Segment")
plt.ylabel("Average Jaccard Similarity")
plt.title("Figure A.8: Segment-wise Stability (Jaccard Index)")
plt.grid(True)
plt.show()


# STEP 6

# Merge segmentation matrix and cluster labels
profile_df = seg_matrix.copy()
profile_df['Cluster'] = df['Cluster']

# Group by cluster and calculate means
cluster_profiles = profile_df.groupby('Cluster').mean()

print("Cluster Profiles (Mean YES Response per Perception):\n", cluster_profiles)

# Plot as heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_profiles, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f")
plt.title("Segment Profiles Based on Segmentation Variables")
plt.ylabel("Cluster")
plt.xlabel("Perception Attribute")
plt.tight_layout()
plt.show()


# STEP 7 Describing Segments

# Separate descriptor variables
descriptor_df = df.iloc[:, 11:].copy()
descriptor_df['Cluster'] = df['Cluster']

# Loop through descriptors
for col in descriptor_df.columns:
    if col == 'Cluster':
        continue

    print(f"\n--- Descriptor: {col} ---")

    if descriptor_df[col].dtype == 'object' or descriptor_df[col].nunique() < 10:
        # Categorical descriptor → use Chi-squared test
        contingency_table = pd.crosstab(descriptor_df['Cluster'], descriptor_df[col])
        print("Contingency Table:\n", contingency_table)

        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-squared test: p-value = {p:.4f}")

        # Plot
        contingency_table.plot(kind='bar', stacked=True, figsize=(8, 4))
        plt.title(f"{col} by Segment")
        plt.xlabel("Cluster")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    else:
        groups = [group[col].values for name, group in descriptor_df.groupby('Cluster')]
        f_stat, p = f_oneway(*groups)
        print(f"ANOVA test: p-value = {p:.4f}")

        # Plot
        sns.boxplot(x='Cluster', y=col, data=descriptor_df)
        plt.title(f"{col} by Segment")
        plt.tight_layout()
        plt.show()





# STEP 8 Selecting the Target Segment

#Segment Sizes

segment_sizes = df['Cluster'].value_counts().sort_index()
print("Segment Sizes:\n", segment_sizes)

# Plot segment sizes
plt.figure(figsize=(6, 4))
sns.barplot(x=segment_sizes.index, y=segment_sizes.values, color='skyblue')
plt.title("Segment Sizes")
plt.xlabel("Segment")
plt.ylabel("Number of Respondents")
plt.grid(True)
plt.tight_layout()
plt.show()


# Map VisitFrequency to Numeric Scores

freq_map = {
    'Never': 0,
    'Once a year': 1,
    'Every three months': 2,
    'Once a month': 3,
    'Once a week': 4,
    'More than once a week': 5
}

# Apply mapping
df['VisitFrequencyNumeric'] = df['VisitFrequency'].map(freq_map)



avg_visit_freq = df.groupby('Cluster')['VisitFrequencyNumeric'].mean()
print("\nAverage Visit Frequency Score per Segment:\n", avg_visit_freq)

# Plot average numeric visit frequency
plt.figure(figsize=(6, 4))
sns.barplot(x=avg_visit_freq.index, y=avg_visit_freq.values, color='lightgreen')
plt.title("Average Visit Frequency Score by Segment")
plt.xlabel("Segment")
plt.ylabel("Visit Frequency Score (0–5 Scale)")
plt.grid(True)
plt.tight_layout()
plt.show()


