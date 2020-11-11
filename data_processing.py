def fill_demean_scale(df, column):
    span = df[column].max() - df[column].min()
    df[column] = df[column].fillna(df[column].mean())
    df[column] = df[column] - df[column].mean()
    df[column] = (df[column] / span).astype(float)
    return df[column]