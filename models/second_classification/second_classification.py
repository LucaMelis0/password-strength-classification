from utils_and_libraries.libs import XGBClassifier

xgb_classifier = XGBClassifier(
    n_estimators=100,
    max_depth=None,
    learning_rate=0.1,
)
