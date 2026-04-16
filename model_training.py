import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, encoding='latin1')
    print(f"Total records: {len(df)}")

    df = df.rename(columns={'state_ut': 'stateut'})

    cols = ['stateut', 'district', 'Disease', 'day', 'mon', 'year',
            'Latitude', 'Longitude', 'preci', 'LAI', 'Temp', 'Cases']

    df = df[cols]

    df['Cases'] = pd.to_numeric(df['Cases'], errors='coerce')
    df = df.dropna(subset=['Cases'])

    df = df.sort_values(by=['stateut', 'district', 'Disease', 'year', 'mon', 'day'])

    # Fill climate
    for col in ['Temp', 'preci', 'LAI']:
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    return df


def create_features(df):
    print("Creating lag features...")
    df['caseslastweek'] = df.groupby(['stateut', 'district', 'Disease'])['Cases'].shift(1)
    df['caseslastmonth'] = df.groupby(['stateut', 'district', 'Disease'])['Cases'].shift(4)
    df = df.dropna(subset=['caseslastweek', 'caseslastmonth'])
    return df


def encode_and_scale(df):
    print("Encoding and scaling...")

    le_state = LabelEncoder()
    df['stateut_enc'] = le_state.fit_transform(df['stateut'])

    le_district = LabelEncoder()
    df['district_enc'] = le_district.fit_transform(df['district'])

    le_disease = LabelEncoder()
    df['disease_enc'] = le_disease.fit_transform(df['Disease'])

    scaler = StandardScaler()
    df[['Temp_scaled', 'preci_scaled', 'LAI_scaled']] = scaler.fit_transform(
        df[['Temp', 'preci', 'LAI']]
    )

    return df, scaler, le_state, le_district, le_disease


def build_lstm(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def prepare_lstm_data(X, y, sequence_length=2):   # 👈 reduced to 2
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    return np.array(X_seq), np.array(y_seq)


if __name__ == "__main__":

    df = load_and_preprocess_data('Final_data_large.csv')
    df = create_features(df)
    df, scaler, le_state, le_district, le_disease = encode_and_scale(df)

    feature_cols = ['day', 'mon', 'year',
                    'Latitude', 'Longitude',
                    'Temp_scaled', 'preci_scaled', 'LAI_scaled',
                    'caseslastweek', 'caseslastmonth',
                    'stateut_enc', 'district_enc', 'disease_enc']

    X = df[feature_cols].values
    y = df['Cases'].values

    print("\nPreparing LSTM sequences...")
    seq_len = 2   # 👈 important
    X_seq, y_seq = prepare_lstm_data(X, y, seq_len)

    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    print("\nTraining LSTM...")
    model = build_lstm((seq_len, X.shape[1]))

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=40,
              batch_size=32,
              callbacks=[early_stop],
              verbose=1)

    y_pred = model.predict(X_test, verbose=0)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("\n--- LSTM Metrics ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}")

    pipeline = {
        'scaler': scaler,
        'le_state': le_state,
        'le_district': le_district,
        'le_disease': le_disease,
        'model': model,
        'features': feature_cols,
        'sequence_length': seq_len
    }

    joblib.dump(pipeline, 'best_disease_model.pkl')
    model.save('best_disease_model.keras')

    print("Done! Model saved successfully.")