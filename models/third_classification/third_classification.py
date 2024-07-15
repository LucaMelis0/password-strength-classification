from utils_and_libraries.libs import MLPClassifier


neural_network_classifier = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    learning_rate='constant',
)
