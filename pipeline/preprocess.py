from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Split and return x_train, x_test, y_train, y_test.
    """
    x = df.drop("loan_default", axis=1)
    y = df["loan_default"]
    return train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
