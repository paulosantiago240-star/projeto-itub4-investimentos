# Previsão de Ativos Financeiros com Machine Learning (ITUB4)

Este projeto utiliza **Regressão Linear** para prever o preço de fechamento das ações do Itaú Unibanco (ITUB4) e automatiza alertas de compra via **Telegram**.

## Estrutura do Projeto
O projeto foi desenvolvido em três camadas:
1. **Análise e Treinamento:** Notebook Jupyter para EDA (Análise Exploratória de Dados) e validação do modelo.
2. **Engenharia de Atributos:** Biblioteca modular (`helpers.py`) para cálculo de indicadores técnicos (MACD, ADX, Bandas de Bollinger).
3. **Automação:** Script Python configurado para execução diária com notificações em tempo real.

## Tecnologias Utilizadas
- **Python 3.14**
- **Bibliotecas:** Pandas, Matplotlib, Numpy, Scikit-Learn, YFinance, Joblib.
- **Indicadores Técnicos:** Médias Móveis Exponenciais (EMA) e Simples (SMA), ADX, Bollinger Bands.

## O Modelo
O modelo de Regressão Linear foi treinado com foco em **momentum e volatilidade**, utilizando a distância do preço em relação às médias móveis e a espessura das bandas de Bollinger como preditores para evitar o superajuste ao preço nominal.

## Automação
O script `automacao_itub4.py` realiza o download dos dados após o fechamento do mercado, processa os indicadores e, caso o modelo identifique um potencial de alta superior a X%, dispara um alerta via API do Telegram.

---
**Desenvolvido por Luis Paulo Loubet** *Graduando em Engenharia Aeronáutica - EESC-USP*
