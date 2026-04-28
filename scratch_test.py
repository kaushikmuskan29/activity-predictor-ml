import joblib
import numpy as np

# Load models
kmeans = joblib.load('models/kmeans.pkl')
scaler = joblib.load('models/scaler.pkl')
mapping = joblib.load('models/cluster_activity_mapping.pkl')
features = joblib.load('models/important_features.pkl')

print("Loaded models successfully.")

# Test all zeros (default slider values)
full_input = np.zeros((1, 10))
scaled_input = scaler.transform(full_input)
distances = kmeans.transform(scaled_input)[0]
pred = kmeans.predict(scaled_input)[0]

print(f"Default (all 0.0) -> Cluster {pred} : {mapping.get(pred, 'Unknown')}")
print(f"Distances: {distances}")

# What if we give some movement values?
moving_input = np.array([[0.5, 0.5, -0.2, 0.1, 0.8, -0.4, 0.9, 0.9, 0.5, 0.5]])
scaled_moving = scaler.transform(moving_input)
pred2 = kmeans.predict(scaled_moving)[0]
print(f"Moving -> Cluster {pred2} : {mapping.get(pred2, 'Unknown')}")

# Print scaler means
print(f"Scaler means: {scaler.mean_}")
