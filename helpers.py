import pandas as pd
import yfinance as yf
import numpy as np

# ====================================
# ADX (Average Directional Index)
# ====================================

def calculate_adx(dataset, period=14):

    df = dataset.copy()
    # 1. Calcular o True Range (TR)
    df['h-l'] = df['High'] - df['Low']
    df['h-pc'] = abs(df['High'] - df['Close'].shift(1))
    df['l-pc'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)

    # 2. Calcular o Movimento Direcional (+DM e -DM)
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']

    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    # 3. Suavização (Wilder's Smoothing)
    # Wilder usava uma média específica, mas a EMA (Média Móvel Exponencial) é a substituta padrão
    df['TR_smooth'] = df['TR'].rolling(window=period).mean() # Ou use .ewm(span=period)
    df['+DM_smooth'] = df['+DM'].rolling(window=period).mean()
    df['-DM_smooth'] = df['-DM'].rolling(window=period).mean()

    # 4. Calcular +DI e -DI
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])

    # 5. Calcular o DX e finalmente o ADX
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].rolling(window=period).mean()

    # Limpeza das colunas auxiliares para não sujar o dataset
    df.drop(['h-l', 'h-pc', 'l-pc', 'up_move', 'down_move', 'TR', 'DX'], axis=1, inplace=True)
    
    return df


# ====================================
# Bandas de Bollinger
# ====================================

def calculate_bollinger_bands(dataset, period=20):

    df = dataset.copy()

    df['EMA_20'] = df['Close'].ewm(span=period, adjust=False).mean()
    df['SMA_20'] = df['Close'].rolling(window=period, min_periods=1).mean()
    df['STD_20'] = df['Close'].rolling(window=period, min_periods=1).std()
    df['Bollinger+'] = df['SMA_20'] + (2 * df['STD_20'])
    df['Bollinger-'] = df['SMA_20'] - (2 * df['STD_20'])

    return df


def calculate_macd(dataset, fast=12, slow=26, signal=9):

    df = dataset.copy()
    df['SMA_fast'] = df['Close'].rolling(window=fast, min_periods=1).mean()
    df['SMA_slow'] = df['Close'].rolling(window=slow, min_periods=1).mean()
    df['MACD'] = df['SMA_fast'] - df['SMA_slow']
    df['MACD_Signal'] = df['MACD'].rolling(window=signal, min_periods=1).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    return df


def calculate_dataframe_features (dataset):

    df = dataset.copy()

    columns_to_drop = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    # Calcular o True Range (TR)
    df['h-l'] = df['High'] - df['Low']
    df['h-pc'] = abs(df['High'] - df['Close'].shift(1))
    df['l-pc'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)

    # Calcular o Movimento Direcional (+DM e -DM)
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']

    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    # Suavização (Wilder's Smoothing)
    # Wilder usava uma média específica, mas a EMA (Média Móvel Exponencial) é a substituta padrão
    df['TR_smooth'] = df['TR'].rolling(window=14).mean() # Ou use .ewm(span=period)
    df['+DM_smooth'] = df['+DM'].rolling(window=14).mean()
    df['-DM_smooth'] = df['-DM'].rolling(window=14).mean()

    # Calcular +DI e -DI
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])

    # Calcular o DX e finalmente o ADX
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].rolling(window=14).mean()

    # Limpeza das colunas auxiliares para não sujar o dataset
    df.drop(['h-l', 'h-pc', 'l-pc', 'up_move', 'down_move', 'TR', 'DX', '+DI','-DI','TR_smooth', '+DM_smooth', '-DM_smooth', '+DM', '-DM'], axis=1, inplace=True)


    # Médias
    # O .squeeze() garante que 'Close' seja tratado como uma única coluna
    close_series = df['Close'].squeeze()
    
    df['EMA_9'] = close_series.ewm(span=9).mean()
    df['EMA_21'] = close_series.ewm(span=21).mean()
    df['SMA_20'] = close_series.rolling(window=20).mean()
    
    df['Distance_SMA_20'] = close_series - df['SMA_20']

    # Bollinger
    df['STD_20'] = close_series.rolling(window=20, min_periods=1).std()
    df['Bollinger+'] = df['SMA_20'] + (2 * df['STD_20'])
    df['Bollinger-'] = df['SMA_20'] - (2 * df['STD_20'])
    
    # Cálculo do %B (Normalizado)
    df['Bollinger_thickness'] = (close_series - df['Bollinger-']) / (df['Bollinger+'] - df['Bollinger-'])

    # Limpeza
    df.drop(columns=['STD_20', 'Bollinger+', 'Bollinger-'], inplace=True)

    # Adicionando a coluna Target
    df['Target'] = df['Close'].shift(-1)

    df.drop(columns=columns_to_drop, inplace=True)
    df.dropna(inplace=True)

    return df


