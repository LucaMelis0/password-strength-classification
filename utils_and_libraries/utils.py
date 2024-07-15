from utils_and_libraries.libs import np, pd, display, math, plt
from utils_and_libraries.libs import shuffle, StandardScaler
from utils_and_libraries.libs import accuracy_score, confusion_matrix, classification_report


def get_data(dataset_path: str):
    """Read the dataset from the given path and return it as a pandas DataFrame."""
    skipped_lines = list()
    data = pd.read_csv(dataset_path, on_bad_lines=skipped_lines.append, engine='python')
    return data, pd.DataFrame(skipped_lines)


def display_data(data: pd.DataFrame, head: bool = True):
    """Display the given data. If head is True, display the first 5 rows, otherwise the entire data."""
    display(data.head()) if head else display(data)


def digits_count(password: str):
    """Return the number of digits in the given password."""
    return sum(c.isdigit() for c in password)


def special_characters_count(password: str):
    """Return the number of special characters in the given password."""
    return sum(not c.isalnum() for c in password)


def uppercase_letters_count(password: str):
    """Return the number of uppercase letters in the given password."""
    return sum(c.isupper() for c in password)


def lowercase_letters_count(password: str):
    """Return the number of lowercase letters in the given password."""
    return sum(c.islower() for c in password)


def unique_characters_count(password: str):
    """Return the number of unique characters in the given password."""
    return len(set(password))


def has_repeated_char(password: str):
    """Return True if the given password has (at least) a 3-time repeated character, False otherwise."""
    for i in range(len(password) - 2):
        if password[i] == password[i + 1]:
            return True
    return False


def calculate_entropy(password: str):
    """
    Calculate the entropy of the given password. The formula is: l * log2(n), where 'l' is the length of the password
    and 'n' is the number of possible characters in the password.
    """
    n = 0
    if any(c.islower() for c in password):
        n += 26
    if any(c.isupper() for c in password):
        n += 26
    if any(c.isdigit() for c in password):
        n += 10
    if any(not c.isalnum() for c in password):
        n += 32  # we consider the standard, 32 characters-long set of special characters

    return len(password) * math.log2(n)


def extract_password_features(data: pd.DataFrame):
    """
    Extract the password statistical values for each password in the given data. The extracted features are:

    - length: the length of the password
    - numbers: the number of digits in the password
    - special_characters: the number of special characters in the password
    - uppercase_letters: the number of uppercase letters in the password
    - lowercase_letters: the number of lowercase letters in the password
    - unique_characters: the number of unique characters in the password
    - entropy: the entropy of the password
    - has_repeated_characters: True if the password has (at least) a 3-time repeated character, False otherwise

    :param data: the data containing the passwords.
    :return: a DataFrame containing the statistical values for each password.
    """
    parameters_for_each_password = dict()

    for password in data['password']:
        password = str(password)

        parameters_for_each_password[password] = {
            'length': len(password),
            'digits': digits_count(password),
            'special_characters': special_characters_count(password),
            'uppercase_letters': uppercase_letters_count(password),
            'lowercase_letters': lowercase_letters_count(password),
            'unique_characters': unique_characters_count(password),
            'entropy': calculate_entropy(password),
            'has_repeated_characters': has_repeated_char(password)
        }

    return pd.DataFrame.from_dict(parameters_for_each_password, orient='index')


def display_strength_distribution(data: pd.DataFrame):
    """Display the distribution of the password strength in the given data."""
    data['strength'].value_counts().sort_index().plot(kind='bar')
    plt.title('Password Strength Distribution')
    plt.xlabel('Strength')
    plt.ylabel('Count')
    plt.grid()
    plt.show()


def display_digits_distribution(data: pd.DataFrame):
    """
    Display the distribution of the frequency of numbers in the passwords,
    then display the minimum and maximum entropy for each password length.
    """
    data['digits'].value_counts().sort_index().plot(kind='bar')
    plt.title('Digits Frequency Distribution')
    plt.xlabel('Number of Digits')
    plt.ylabel('Count')
    plt.grid()
    plt.show()


def display_entropy_function(data: pd.DataFrame):
    """Display the distribution of the entropy of the passwords as a function of the length."""
    data.groupby('length')['entropy'].mean().plot(kind='line')
    plt.title('Evolution of Entropy Based on Password Length')
    plt.xlabel('Length')
    plt.ylabel('Entropy')
    plt.grid()
    plt.show()

    display(data.groupby('length')['entropy'].agg(['min', 'max', 'count']))


def balance_data_samples(data: pd.DataFrame):
    """
    Balance the data by under-sampling the two majority classes to match the number of samples of the minority class.

    :param data: the data to balance, as a Dataframe.
    :return: the balanced data.
    """
    # The class with the lower number of samples is class 2, so the number of samples related to the other classes
    # have to be equal to the number of samples of class 2
    class_0_samples = data[data['strength'] == 0]
    class_1_samples = data[data['strength'] == 1]
    class_2_samples = data[data['strength'] == 2]

    class_0_samples = shuffle(class_0_samples, random_state=42).iloc[:len(class_2_samples)]
    class_1_samples = shuffle(class_1_samples, random_state=42).iloc[:len(class_2_samples)]

    balanced_data = pd.concat([class_0_samples, class_1_samples, class_2_samples])

    # Display the new distribution of the password strength
    print("New, balanced label distribution: ")
    display_strength_distribution(balanced_data)

    return balanced_data


def prepare_features(password_dataset: pd.DataFrame, balance_data: bool = True):
    """
    Prepare the features and labels for the classification task. If balance_data is True, this function balances the
    data by under-sampling the two majority classes to match the number of samples of the minority class.
    It then normalizes the features using StandardScaler.

    :param password_dataset: the dataset containing the passwords.
    :param balance_data: whether to balance the data or not.
    :return: the features, labels, and password names.
    """
    # Balance the data if required
    if balance_data:
        data = balance_data_samples(password_dataset)
    else:
        data = password_dataset

    # Extract passwords, features, and labels
    passwords = data['password']
    y = data['strength']
    x = extract_password_features(data)

    # Convert x and y from dataframes to numpy arrays
    x = x.to_numpy()
    y = y.to_numpy()

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    return x, y, passwords


def display_cv_results(cross_validation_metrics: dict):
    """
    Display the cross-validation metrics in a DataFrame.

    :param cross_validation_metrics: the cross-validation metrics.
    """
    # Convert to DataFrame, transpose, and rename index
    cross_val_df = pd.DataFrame(cross_validation_metrics).T
    cross_val_df.columns = ['KFold Iteration: n.1',
                            'KFold Iteration: n.2',
                            'KFold Iteration: n.3',
                            'KFold Iteration: n.4',
                            'KFold Iteration: n.5']
    cross_val_df.index = ['Fit Time', 'Score Time', 'Validation Score']
    display(cross_val_df)


def get_metrics(y_test: np.ndarray, y_pred: np.ndarray):
    """
    Evaluate the accuracy, confusion matrix, and classification report for the given true and predicted labels.

    :param y_test: the true labels.
    :param y_pred: the predicted labels.
    :return: the accuracy, confusion matrix, and classification report.
    """
    accuracy_result = accuracy_score(y_test, y_pred)
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    classification_report_result = classification_report(
        y_test,
        y_pred,
        target_names=['weak', 'medium', 'strong'],
        output_dict=True
    )

    return accuracy_result, confusion_matrix_result, classification_report_result


def display_metrics(y_test: np.ndarray, y_pred: np.ndarray):
    """
    Display accuracy, confusion matrix, and classification report for the given true and predicted labels.

    :param y_test: the true labels.
    :param y_pred: the predicted labels.
    """
    accuracy, confusion_matrix_result, classification_report_result = get_metrics(y_test, y_pred)

    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    plt.matshow(confusion_matrix_result, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print('Classification Report:')
    display(pd.DataFrame(classification_report_result).T)
