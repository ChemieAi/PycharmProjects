import random

num_instances = 8124
num_features = 22
value_range = (1, 10)

dataset = []

# Generate feature names
feature_names = ["Feature " + str(i) for i in range(1, num_features + 1)]
dataset.append(feature_names)

for _ in range(num_instances):
    instance = [random.randint(value_range[0], value_range[1]) for _ in range(num_features)]
    dataset.append(instance)

# Writing the dataset to a file
filename = "dataset2.csv"
with open(filename, 'w') as file:
    for instance in dataset:
        file.write(','.join(str(value) for value in instance))
        file.write('\n')

print("Dataset created and saved as", filename)
