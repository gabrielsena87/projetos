import pandas as pd

# Caminho do relatório gerado na pasta workspace
caminho_relatorio = "workspace/reports/classification_report.csv"

# Lê o arquivo CSV
df = pd.read_csv(caminho_relatorio)

# Exibe o relatório formatado
print(df)