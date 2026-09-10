import pandas as pd

# Caminho do relatório gerado no workspace do Thermal Vision AI
caminho_relatorio = "thermal_vision_workspace/reports/classification_report.csv"

# Lê o arquivo CSV
df = pd.read_csv(caminho_relatorio)

# Exibe o relatório formatado
print(df)