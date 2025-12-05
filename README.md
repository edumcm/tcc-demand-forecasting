# 📦 Projeto de Previsão de Demanda — Olist (TCC)

## 📁 Estrutura do Projeto

tcc-demand-forecasting/  
├─ LICENSE  
├─ README.md  
├─ requirements.txt  
├─ configs/  
│  └─ data.yaml — Caminhos e diretórios do projeto  
├─ data/  
│  ├─ raw/ — Dados originais da Olist (CSV)  
│  ├─ interim/ — Dados intermediários (merged, agregados, lags)  
│  └─ processed/ — Dados finais prontos para modelagem  
├─ src/  
│  ├─ data/  
│  │  ├─ loader.py — Carregamento das bases  
│  │  ├─ preprocessing.py — Limpeza, merges e padronização  
│  │  ├─ aggregation.py — Agregação temporal  
│  │  └─ imputation.py — Regras de imputação  
│  ├─ features/  
│  │  └─ build.py — Lags, médias móveis, features temporais  
│  ├─ training_schema/  
│  │  └─ split_rolling.py — Validação temporal  
│  ├─ models/  
│  │  ├─ lgbm_fit_predict.py  
│  │  ├─ lgbm_tunning.py  
│  │  ├─ prophet_fit_predict.py  
│  │  ├─ sarima_fit_predict.py  
│  │  └─ lstm_fit_predict.py  
│  └─ evaluations/  
│     ├─ models_metrics.py — Métricas de modelos  
│     ├─ feature_analysis.py — Análise de features  
│     └─ plot_real_pred.py — Gráficos Real vs Predito  
├─ notebooks/  
│  ├─ 00_colab_bootstrap.ipynb  
│  ├─ 01_eda_overview.ipynb  
│  ├─ 02_preprocessamento.ipynb  
│  ├─ 03_analise_features.ipynb  
│  └─ 04_experiments.ipynb  
├─ artifacts/  
│  ├─ oof/  
│  ├─ predictions/  
│  ├─ metrics/  
│  ├─ figures/  
│  ├─ hpo/  
│  └─ config_snapshots/  
└─ reports/  
   ├─ tables/  
   └─ figures/  

---

# 📘 Descrição do Projeto

Este projeto implementa um pipeline completo de previsão de demanda utilizando o dataset público da Olist.  
O objetivo é prever as vendas totais ao longo do tempo, aplicando modelos estatísticos e de machine learning.

---

# 🎯 Objetivos

- Prever vendas totais agregadas ao longo do tempo.  
- Construir pipeline modular e completo.  
- Comparar modelos: LightGBM, Prophet, SARIMA, LSTM.  
- Avaliar métricas: RMSE, MAPE, WAPE, MASE, RMSSE.  
- Identificar features relevantes via Pearson e MI.

---

# 🔄 Fluxo do Projeto

## 1. Carregamento dos Dados  
Leitura de CSVs em data/raw usando configs/data.yaml.

## 2. Pré-processamento  
Merge, limpeza e geração de olist_merged.parquet.

## 3. Agregação Temporal  
Geração de vendas totais semanais.

## 4. Feature Engineering  
Lags, médias móveis, janelas deslizantes, calendário.

## 5. Imputação  
Regras específicas (zero, NaN, flags).

## 6. Validação Temporal  
Esquema rolling/expanding window.

## 7. Treinamento  
Modelos LightGBM, Prophet, SARIMA e LSTM.

## 8. Avaliação  
Cálculo de RMSE, MAPE, WAPE, MASE, RMSSE.

## 9. Análise de Features  
Pearson, MI, ranking e figuras.

## 10. Geração de Gráficos  
Real vs predito em reports/figures.

---

# 📒 Execução via Notebooks

- 00_colab_bootstrap.ipynb  
- 01_eda_overview.ipynb  
- 02_preprocessamento.ipynb  
- 03_analise_features.ipynb  
- 04_experiments.ipynb  

---

# ⚙️ Instalação

pip install -r requirements.txt

---

# 📁 Pastas de Destaque

- data/raw — dados brutos  
- data/interim — dados tratados  
- artifacts — estrutura para experimentos  
- reports — resultados finais  

---

# 📄 Licença

MIT License
