import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_chess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset cargado exitosamente desde: {file_path}")
        print(f"Forma del dataset: {df.shape}")
        print(df.head(), "\n")
        return df
    except FileNotFoundError:
        print(f"Error: El archivo no se encontró en la ruta: {file_path}")
        return None
    except Exception as e:
        print(f"Ocurrió un error al cargar el dataset: {e}")
        return None

def calculate_total_ply(moves_series):
    return moves_series.apply(lambda x: len(str(x).split(' ')) if pd.notna(x) else np.nan)

def extract_time_controls(df):

    def parse_code(code):
        try:
            parts = str(code).split('+')
            initial_time = int(parts[0])
            increment = int(parts[1])
            return initial_time, increment
        except:
            return np.nan, np.nan

    df[['initial_time_minutes', 'increment_seconds']] = df['increment_code'].apply(
        lambda x: pd.Series(parse_code(x))
    )

    df.drop(columns=['increment_code'], inplace=True)

    return df

def normalize_features(X_train, X_test, normalizer=None):
    if normalizer is None:
        normalizer = MinMaxScaler()

    normalizer_fitted = normalizer.fit(X_train)

    X_train_norm = normalizer_fitted.transform(X_train)
    X_test_norm = normalizer_fitted.transform(X_test)

    X_train_norm = pd.DataFrame(X_train_norm, columns=X_train.columns, index=X_train.index)
    X_test_norm = pd.DataFrame(X_test_norm, columns=X_test.columns, index=X_test.index)

    return X_train_norm, X_test_norm, normalizer_fitted
