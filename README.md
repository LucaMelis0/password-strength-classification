## Password Strength Classification

A simple password strength classification program that aims at analyzing the performance of three different classifiers 
on a password strength dataset.

### Description

The program uses three different classifiers to classify the strength of a password based on the dataset provided. 

The used classifiers are:
1. Random Forest
2. XGBoost Classifier
3. Neural Network (MLP)

The used dataset is the Password Strength dataset from Kaggle. The dataset contains more than 650000 passwords with 
their corresponding strength level (from 0 to 2, depending on their reliability).

Once installed the requirements and retrieved the dataset, the program will extract the features from each password and 
display them in bar charts and dataframes. 

Once the features are extracted, the data is preprocessed before being split into training and testing sets. Since the
dataset is not balanced, the data is under-sampled, meaning that only part of these three subsets is considered, so that
the corresponding number of samples related to each class is the same.

After the data preprocessing phase, it is split into training and test sets; therefore, K-Fold cross-validation is 
implemented in order to assess the performance of the classifiers. The performance is thus evaluated considering the 
accuracy, fit time and prediction time on the validation set, different for each iteration of K-Fold.

Finally, the cross-validation is performed, the program will then train the classifiers on the training set and test them on
the test set. The performance of the classifiers is then evaluated looking at the accuracy, the confusion matrix and the
classification report.

### Dependencies

The project relies on several Python libraries for data manipulation, data visualization and machine learning.
The dependencies are listed in the `requirements.txt` file.

### Installing

To customize the project, therefore installing and running it, the following steps are needed:
1. Considering Python and Git installed on the machine, clone the repository as:
```bash
git clone https://github.com/LucaMelis0/password-strength-classification.git
```
2. Install the required dependencies by running:
```bash
pip install -r requirements.txt
```

### Acknowledgments
Special thanks to the creators and contributors of the [Password Strength Classifier Dataset](https://www.kaggle.com/datasets/bhavikbb/password-strength-classifier-dataset) on Kaggle. 
