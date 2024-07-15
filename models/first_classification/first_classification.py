from utils_and_libraries.libs import RandomForestClassifier

random_forest_classifier = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_features=None,
    bootstrap=True,
    )
