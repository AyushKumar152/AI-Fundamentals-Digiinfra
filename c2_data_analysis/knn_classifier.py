import numpy as np

class ImageKNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict_single(self, image):
        distances = np.linalg.norm(self.X_train - image, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        values, counts = np.unique(k_labels, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X_test):
        return [self.predict_single(x) for x in X_test]


def calculate_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (np.sum(y_true == y_pred) / len(y_true)) * 100

if __name__ == "__main__":

    # Dummy training data (flattened images)
    X_train = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [10, 10, 10],
        [11, 11, 11]
    ])
    y_train = np.array([0, 0, 1, 1])

    # Dummy test data
    X_test = np.array([
        [2, 2, 3],
        [10, 9, 10]
    ])
    y_test = np.array([0, 1])

    # Create & train model
    knn = ImageKNNClassifier(k=3)
    knn.fit(X_train, y_train)

    # Predict
    predictions = knn.predict(X_test)

    # Accuracy
    acc = calculate_accuracy(y_test, predictions)

    print("Predictions:", predictions)
    print("Accuracy:", acc)
