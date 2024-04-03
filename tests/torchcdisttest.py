import torch

x = torch.tensor([[1.0, 1.0],[2.0, 2.0]])
distances = torch.cdist(x, x) # calculate the distance between each pair of points

# We'll use argsort to find the indices that would sort each row
sorted_indices = distances.argsort() # sort the distances
print(sorted_indices)