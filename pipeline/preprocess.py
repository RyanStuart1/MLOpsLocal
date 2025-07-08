from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Split into train / validation / test sets:
        train: 0.60
        val:   0.20
        test:  0.20
    Returns: x_train, x_val, x_test, y_train, y_val, y_test
    """
    x = df.drop("loan_default", axis=1)
    y = df["loan_default"]
    # First split off test (20%)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    # From the remaining 80%, split out validation (20% of total = 25% of 80%)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=42
    )
    return x_train, x_val, x_test, y_train, y_val, y_test