import pandas as pd

def fill_demean_scale(df, column):
    span = df[column].max() - df[column].min()
    df[column] = df[column].fillna(df[column].mean())
    df[column] = df[column] - df[column].mean()
    df[column] = (df[column] / span).astype(float)
    return df[column]

def prepare_train_and_test_data(train_path, test_path):
    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)
    cols_to_ignore = "Survived,Pclass,Embarked,Name,Ticket".split(",")
    features = [col for col in train_df.columns if col not in cols_to_ignore]
    X_train = train_df[features]
    y_train = train_df.Survived
    X_test = test_df[features]
    return X_train, y_train, X_test
