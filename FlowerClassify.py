def load_data(filepath):
    with open(filepath, 'r') as file:
        data = []
        first_line = True
        for line in file:
            if first_line:  # Skip the first line if it's the header
                first_line = False
                continue
            features = line.strip().split(',')
            # Skip the 'Id' column, convert the next four entries to float, and map the species to an int
            if features[-1] == 'Iris-setosa':
                features[-1] = 0
            elif features[-1] == 'Iris-versicolor':
                features[-1] = 1
            elif features[-1] == 'Iris-virginica':
                features[-1] = 2
            data.append([float(x) for x in features[1:-1]] + [int(features[-1])])
    return data

filepath = 'Iris/Iris.csv'
dataset = load_data(filepath)

import random

def split_data(data, test_ratio):
    random.shuffle(data)  # Shuffle the data to ensure randomness
    split_idx = int(len(data) * (1 - test_ratio))  # Find the split point
    train_set = data[:split_idx]
    test_set = data[split_idx:]
    return train_set, test_set

# Example usage
train_set, test_set = split_data(dataset, 0.2)

from math import sqrt

def euclidean_distance(point1, point2):
    return sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def knn(training_data, test_instance, k):
    distances = []
    # Calculate distances from the test instance to all training points
    for data in training_data:
        dist = euclidean_distance(test_instance[:-1], data[:-1])
        distances.append((data, dist))
    # Sort by distance and get the nearest k neighbors
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]

    # Aggregate the most common class
    class_votes = {}
    for neighbor in neighbors:
        response = neighbor[-1]  # class is in the last column
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1

    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]  # Return the class with the most votes


test_instance = test_set[0]
predicted_class = knn(train_set, test_instance, 3)
print(f'Predicted class: {predicted_class}')
def evaluate_knn_model(train_set, test_set, k):
    correct = 0
    total = len(test_set)

    for test_instance in test_set:
        predicted_class = knn(train_set, test_instance, k)
        actual_class = test_instance[-1]
        if predicted_class == actual_class:
            correct += 1

    accuracy = correct / total
    return accuracy

k = 11
accuracy = evaluate_knn_model(train_set, test_set, k)
print(f'Accuracy of kNN classifier with k={k}: {accuracy:.2f}')
