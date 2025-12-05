# src/evaluation/plot_real_pred.py

import matplotlib.pyplot as plt
import numpy as np

 # ---------- gráfico predito x real ----------
def plot_real_pred(real: np.array, pred: np.array, date: np.array, title: str, fig_path):
  """
  Plota gráfico de linhas com valores reais vs o predito
  """
  plt.figure(figsize=(10, 7))
  plt.plot(date, real, label="true")
  plt.plot(date, pred, label="pred")
  plt.title(title)
  plt.legend()
  plt.ylim(ymin=0)
  plt.xticks(rotation=45)
  plt.savefig(fig_path, dpi=160)
  plt.show()
  plt.close()