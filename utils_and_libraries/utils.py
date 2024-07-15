from utils_and_libraries.libs import pd, display, math


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
