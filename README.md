# Thermal Vision AI 🔥

### Classificação inteligente de equipamentos elétricos em imagens térmicas

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit--learn](https://img.shields.io/badge/scikit-learn-1.0%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-active%20development-2EA44F)](https://github.com/gabrielsena87/projetos)
[![Licença](https://img.shields.io/badge/licenca-MIT-2EA44F)](LICENSE)

O **Thermal Vision AI** transforma imagens infravermelhas em uma classificação objetiva de equipamentos elétricos. O pipeline combina a representação visual do **DINOv2**, descritores locais **SIFT** e um ensemble de modelos clássicos para reconhecer cinco categorias de ativos.

Em outras palavras: a imagem entra quente, o vetor de características faz o trabalho pesado e o modelo devolve uma classe com probabilidade. ⚡

## Por que este projeto é especial? ✨

Porque ele não depende de uma única lente para enxergar o problema. O DINOv2 captura contexto e semântica visual; o SIFT preserva pistas locais e geométricas; o stacking combina classificadores com comportamentos diferentes. Essa composição torna o pipeline mais explícito, auditável e fácil de experimentar do que uma caixa-preta isolada.

O projeto também separa as etapas de dados, extração, otimização e inferência. Assim, cada decisão pode ser inspecionada, reproduzida e melhorada sem desmontar o restante do sistema.

## Resultado de referência 📊

Em uma execução com **893 imagens válidas**, foram observados:

| Métrica | Resultado | Observação |
| --- | ---: | --- |
| Acurácia média na validação cruzada | 99,10% | 5 divisões estratificadas |
| Acurácia global | 99,10% | Predições out-of-fold |
| Cohen's Kappa | 0,9887 | Medida de concordância |
| F1 macro | 0,9910 | Média equilibrada entre classes |

Esses números são uma referência do conjunto disponível, não uma garantia universal. Eles podem variar com a versão das dependências, o hardware, o dataset e o estado dos artefatos em cache.

## Classes reconhecidas

- `Circuit Breakers`
- `Disconnectors`
- `Power Transformers`
- `Surge Arresters`
- `Wave Traps`

## Arquitetura técnica 🧠

```text
Imagem térmica (JPG / JPEG / PNG)
                │
                ▼
┌──────────────────────────────────────┐
│ DataManager                           │
│ valida, converte RGB e cria metadata  │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│ FeatureExtractor                      │
│ SIFT (128) + DINOv2 ViT-B/14 (768)    │
│ vetor híbrido: 896 dimensões          │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│ PipelineOptimizer                     │
│ StandardScaler → VarianceThreshold    │
│ → SelectKBest → PCA                   │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│ StackingClassifier                    │
│ KNN + Random Forest + SVM             │
│ meta-modelo: Logistic Regression      │
└──────────────────┬───────────────────┘
                   ▼
Classe prevista + probabilidade + relatórios
```

### Componentes

| Componente | Tecnologia | Responsabilidade |
| --- | --- | --- |
| Ingestão | Python, Pillow, pandas | Validar imagens e gerar `metadata.csv` |
| Embedding profundo | PyTorch, DINOv2 ViT-B/14 | Extrair 768 características visuais |
| Descritor local | OpenCV SIFT | Extrair 128 características locais médias |
| Engenharia de atributos | scikit-learn | Escalonamento, seleção e PCA |
| Classificação | KNN, Random Forest, SVM, Logistic Regression | Produzir a decisão final por stacking |
| Persistência | NumPy, joblib, JSON | Salvar features, modelo e configuração |
| Relatórios | pandas, Matplotlib, Seaborn, HTML | Registrar métricas e visualizações |

## Estrutura do projeto

```text
.
├── VIT.py                         # Pipeline principal e classe Predictor
├── vit_dinov2.py                  # DINOv2, embeddings e fine-tuning
├── testar_predictor.py            # Inferência de uma imagem
├── testar_6_imagens.py            # Geração de painel de inferência
├── relatorio.py                   # Leitura de artefatos do modelo
├── teste.py                       # Utilitário local de inspeção
├── Infrared Power Equipment Dataset/  # Dataset organizado por classe
├── workspace/
│   ├── cleaned/                   # Imagens válidas e metadata.csv
│   ├── features/                  # X.npy e y.npy
│   ├── models/                    # Modelo, classes e configuração
│   └── reports/                   # CSVs, gráficos e dashboard
└── README.md
```

## Requisitos

- Windows 10/11 ou Linux
- Python 3.12
- 8 GB de RAM; 16 GB ou mais é recomendado
- GPU NVIDIA com driver atualizado é recomendada, mas CPU é suportada
- Cerca de 5 GB livres para dependências, cache do Torch Hub e artefatos
- Acesso à internet na primeira execução para baixar o DINOv2, caso ele não esteja em cache

## Instalação 🚀

### 1. Obter o projeto e criar o ambiente

No PowerShell:

```powershell
Set-Location "D:\hd_central\projetos"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 2. Instalar o PyTorch

Para uma GPU compatível com CUDA 12.8:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio `
  --index-url https://download.pytorch.org/whl/cu128
```

Para CPU, use as wheels padrão:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio
```

### 3. Instalar as dependências do projeto

```powershell
.\.venv\Scripts\python.exe -m pip install `
  numpy pandas pillow opencv-python matplotlib seaborn `
  scikit-learn joblib tqdm
```

### 4. Validar o ambiente

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

O driver NVIDIA é suficiente para as wheels oficiais do PyTorch; não é obrigatório instalar o CUDA Toolkit completo.

## Dataset 🗂️

O caminho padrão está configurado em `Config.dataset_dir`, no arquivo `VIT.py`:

```python
dataset_dir: Path = Path(r"D:\hd_central\projetos\Infrared Power Equipment Dataset")
```

Organize as imagens desta forma:

```text
Infrared Power Equipment Dataset/
├── Circuit Breakers/
├── Disconnectors/
├── Power Transformers/
├── Surge Arresters/
└── Wave Traps/
```

São aceitos arquivos `.jpg`, `.jpeg` e `.png`. Cada imagem é verificada, convertida para RGB e copiada para `workspace/cleaned/` antes da extração.

## Uso

### Treinar o pipeline híbrido

```powershell
Set-Location "D:\hd_central\projetos"
.\.venv\Scripts\python.exe -u .\VIT.py
```

Para acompanhar os logs e salvar uma cópia:

```powershell
.\.venv\Scripts\python.exe -u .\VIT.py 2>&1 |
  Tee-Object -FilePath .\pipeline.log
```

O processo valida o dataset, extrai as features, otimiza `SelectKBest` e PCA, treina o ensemble, calcula a validação cruzada e salva o modelo final.

### Inferir uma imagem em Python

```python
from VIT import Predictor

predictor = Predictor("workspace/models")
classe, probabilidade = predictor.predict("caminho/para/imagem.jpg")

print(f"Classe: {classe}")
print(f"Confiança: {probabilidade:.2%}")
```

### Inferir pelo terminal

```powershell
.\.venv\Scripts\python.exe .\testar_predictor.py `
  "caminho\para\imagem.jpg" `
  --models-dir workspace/models `
  --save workspace/reports/prediction_{stem}.png
```

Use `--show` para abrir o gráfico após a classificação. Sem o caminho da imagem, o script solicita o valor de forma interativa.

### Fine-tuning supervisionado

O módulo `vit_dinov2.py` possui `DINOv2Classifier` e `ViTTrainer` para fine-tuning completo ou linear probing. O treinamento usa split estratificado 80/20 e salva o melhor checkpoint em `workspace/models/dinov2_best.pt`.

```python
import pandas as pd
from vit_dinov2 import DINOv2Classifier, ViTTrainer

metadata = pd.read_csv("workspace/cleaned/metadata.csv")
classes = sorted(metadata["label"].unique())
label_to_idx = {label: index for index, label in enumerate(classes)}

model = DINOv2Classifier(
    variant="dinov2_vitb14",
    num_classes=len(classes),
    freeze_backbone=False,  # True para linear probing
)

trainer = ViTTrainer(
    model=model,
    metadata=metadata,
    label_to_idx=label_to_idx,
    epochs=20,
    batch_size=32,
    lr=1e-4,
    save_path="workspace/models",
)
history = trainer.train()
```

O checkpoint de fine-tuning é independente do `hybrid_stacking_model.joblib` usado pelo `Predictor` do pipeline híbrido.

## Artefatos gerados

| Caminho | Conteúdo |
| --- | --- |
| `workspace/cleaned/metadata.csv` | Caminhos das imagens válidas e seus rótulos |
| `workspace/features/X.npy` | Matriz de features híbridas |
| `workspace/features/y.npy` | Rótulos correspondentes |
| `workspace/models/hybrid_stacking_model.joblib` | Pipeline treinado e persistido |
| `workspace/models/class_names.joblib` | Ordem das classes usadas na inferência |
| `workspace/models/pipeline_config.json` | Dimensões e normalização da entrada |
| `workspace/reports/classification_report.csv` | Precisão, recall e F1 por classe |
| `workspace/reports/cv_results.csv` | Resultados das divisões da validação cruzada |
| `workspace/reports/confusion_matrix.png` | Matriz de confusão |
| `workspace/reports/dashboard.html` | Painel consolidado dos resultados |

### Cache de features

Se o backbone ou a implementação da extração mudar, remova `workspace/features/X.npy` e `workspace/features/y.npy` antes de treinar novamente. O pipeline reutiliza esses arquivos quando eles existem.

## Troubleshooting 🔧

### `CUDA out of memory`

Reduza `batch_size` no `ViTTrainer`, encerre outros processos que usam a GPU ou execute a extração em CPU.

### Download do DINOv2 falha

Verifique a conexão e tente novamente. O Torch Hub mantém o repositório em cache; se o cache estiver corrompido, remova a pasta correspondente e repita a execução.

### `WinError 1455`

O pipeline já usa `n_jobs=1` na validação cruzada para reduzir o consumo de memória virtual no Windows. Evite aumentar esse valor em máquinas com pouca RAM.

### Aviso sobre `xFormers`

`UserWarning: xFormers is not available` indica apenas que certas otimizações não estão instaladas. O DINOv2 continua funcional.

## Contribuição 🤝

Contribuições são bem-vindas, especialmente melhorias de reprodutibilidade, avaliação e desempenho.

1. Faça um fork e crie uma branch descritiva:

   ```powershell
   git checkout -b feature/nova-avaliacao
   ```

2. Faça uma alteração pequena e documente decisões relevantes.
3. Execute os testes ou validações disponíveis antes de abrir o PR.
4. Atualize o README quando mudar comandos, artefatos ou requisitos.
5. Abra um Pull Request com contexto, resultados antes/depois e limitações conhecidas.

Exemplos de áreas para evolução:

- API REST com FastAPI ou interface com Streamlit;
- suporte a mais backbones, como EfficientNet e ConvNeXt;
- quantização FP16/INT8 e otimização para edge;
- mapas de atenção e explicabilidade;
- testes automatizados para ingestão, cache e inferência.

## Licença 📄

Este projeto é distribuído sob a **MIT License**. Consulte o arquivo [LICENSE](LICENSE).

## Créditos

- [DINOv2](https://github.com/facebookresearch/dinov2), Meta AI Research;
- [OpenCV SIFT](https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html);
- [PyTorch](https://pytorch.org/) e [scikit-learn](https://scikit-learn.org/).

---

Feito para aproximar inspeção térmica, engenharia de atributos e aprendizado de máquina em um fluxo que dá para entender, medir e melhorar. 🔥
