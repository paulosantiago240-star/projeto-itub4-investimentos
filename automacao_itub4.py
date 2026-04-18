import yfinance as yf
import pandas as pd
import joblib
import requests
import os
import sys

def executar_automacao():
    # Encontra o caminho da pasta onde o script atual está salvo
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_modelo = os.path.join(diretorio_atual, 'modelo_itub4_regressao.pkl')
    
    print(f"Buscando modelo em: {caminho_modelo}")
    
    try:
        modelo = joblib.load(caminho_modelo)
        print("✅ Modelo carregado com sucesso!")
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo .pkl não foi encontrado em: {diretorio_atual}")
        return
    
# Importando sua função do helpers.py
try:
    from Códigos.helpers import calculate_dataframe_features
except ImportError:
    print("Erro: O arquivo helpers.py deve estar na mesma pasta.")

# --- CONFIGURAÇÕES DO TELEGRAM ---
TELEGRAM_TOKEN = "8622738844:AAEVS4mrWxmh-jueYxVWGCbos0I8ymsO7M4"
TELEGRAM_CHAT_ID = "7344559332"

def enviar_telegram(mensagem):
    """Envia uma notificação instantânea para o seu celular"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensagem, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
        print("Notificação enviada ao Telegram!")
    except Exception as e:
        print(f"Erro ao conectar com Telegram: {e}")

def executar_automacao():
    print("🤖 Iniciando Robô de Investimentos ITUB4...")
    
    # 1. Carregar modelo
    try:
        modelo = joblib.load('modelo_itub4_regressao.pkl')
    except:
        print("Erro: Certifique-se de que o arquivo .pkl está na pasta.")
        return

    # 2. Baixar dados
    data = yf.download("ITUB4.SA", period="60d", progress=False)
    preco_atual = float(data['Close'].iloc[-1].squeeze())

    # 3. Gerar Features
    df_features = calculate_dataframe_features(data)
    
    # --- AJUSTE CRÍTICO: Compatibilidade de 14 colunas ---
    # Remova o 'Target' e garanta que as colunas batam com o treino
    X_input = df_features.drop(columns=['Target']).tail(1)
    
    # Se o erro de '14 features' persistir, você deve listar aqui as 
    # colunas exatas que usou no treino do notebook:
    # colunas_treino = ['EMA_9', 'EMA_21', 'SMA_7', 'ADX', '+DI', '-DI', ...]
    # X_input = X_input[colunas_treino]

    # 4. Previsão
    try:
        previsao = modelo.predict(X_input)[0]
    except ValueError as e:
        print(f"❌ Erro de dimensionalidade: {e}")
        return

    # 5. Lógica de Decisão
    variacao = ((previsao / preco_atual) - 1) * 100
    print(f"Previsão: R$ {previsao:.2f} ({variacao:.2f}%)")

    # Gatilho: Se a previsão for de alta acima de 0.7%
    if variacao > 0.01:
        alerta = (
            f"🚀 *Sinal de Compra: ITUB4*\n\n"
            f"• Preço Atual: R$ {preco_atual:.2f}\n"
            f"• Alvo Estimado: R$ {previsao:.2f}\n"
            f"• Potencial: {variacao:.2f}%\n\n"
            f"✅ Modelo validado via Regressão Linear."
        )
        enviar_telegram(alerta)
    else:
        print("Sinal neutro. Nenhuma mensagem enviada.")

if __name__ == "__main__":
    executar_automacao()