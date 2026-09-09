"""
vit_dinov2.py

Módulo de ViT treinável baseado em DINOv2 (Meta AI), projetado para ser
plugado no hybrid_fusion_pipeline.py como substituto ou complemento ao
SqueezeNet. Suporta extração de embeddings, fine-tuning completo e
fine-tuning com cabeça linear congelada (linear probing).

Compatibilidade: torch >= 2.0, torchvision >= 0.15
Backbone disponíveis: dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

logger = logging.getLogger("HybridPipeline.ViT")


# ==============================================================================
# Constantes e mapeamento de dimensão por backbone
# ==============================================================================

DINOV2_VARIANTS = {
    "dinov2_vits14": 384,   # ViT-Small  — rápido, leve
    "dinov2_vitb14": 768,   # ViT-Base   — equilíbrio ideal (recomendado)
    "dinov2_vitl14": 1024,  # ViT-Large  — alta capacidade
    "dinov2_vitg14": 1536,  # ViT-Giant  — máxima acurácia, VRAM intensivo
}


# ==============================================================================
# Dataset interno para fine-tuning
# ==============================================================================

class ThermalImageDataset(Dataset):
    """
    Dataset PyTorch leve que consome o DataFrame de metadados já gerado
    pelo DataManager (colunas: 'filepath', 'label').
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        label_to_idx: dict,
        transform: transforms.Compose,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.metadata.iloc[idx]
        with Image.open(row["filepath"]) as img:
            img = img.convert("RGB")
        tensor = self.transform(img)
        label = self.label_to_idx[row["label"]]
        return tensor, label


# ==============================================================================
# Backbone DINOv2 com cabeça classificadora treinável
# ==============================================================================

class DINOv2Classifier(nn.Module):
    """
    ViT-DINOv2 com cabeça MLP treinável.

    Modos de operação
    -----------------
    - freeze_backbone=True  → linear probing: apenas a cabeça é treinada.
    - freeze_backbone=False → fine-tuning completo de todo o modelo.
    - num_classes=0         → extrator puro de embeddings (sem cabeça).

    Parâmetros
    ----------
    variant : str
        Um dos DINOV2_VARIANTS. Padrão: 'dinov2_vitb14'.
    num_classes : int
        Número de classes do dataset. 0 desativa a cabeça classificadora.
    freeze_backbone : bool
        Congela os pesos do ViT e treina apenas a cabeça.
    dropout : float
        Dropout aplicado antes da camada linear final.
    """

    def __init__(
        self,
        variant: str = "dinov2_vitb14",
        num_classes: int = 0,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        if variant not in DINOV2_VARIANTS:
            raise ValueError(
                f"Backbone '{variant}' inválido. "
                f"Escolha entre: {list(DINOV2_VARIANTS)}"
            )

        self.variant = variant
        self.embed_dim = DINOV2_VARIANTS[variant]
        self.num_classes = num_classes

        logger.info(f"Carregando backbone DINOv2: {variant} (embed_dim={self.embed_dim})")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            variant,
            pretrained=True,
        )

        if freeze_backbone:
            logger.info("Backbone congelado → linear probing ativado.")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Cabeça classificadora opcional
        self.head: Optional[nn.Sequential] = None
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(p=dropout),
                nn.Linear(self.embed_dim, num_classes),
            )
            logger.info(
                f"Cabeça classificadora criada: {self.embed_dim} → {num_classes} classes."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna logits (com cabeça) ou o embedding [CLS] bruto (sem cabeça).
        """
        # DINOv2 expõe forward_features para obter o token [CLS]
        features = self.backbone.forward_features(x)
        cls_token = features["x_norm_clstoken"]  # shape: (B, embed_dim)

        if self.head is not None:
            return self.head(cls_token)
        return cls_token

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Sempre retorna o embedding [CLS], ignorando a cabeça."""
        with torch.no_grad():
            features = self.backbone.forward_features(x)
        return features["x_norm_clstoken"]

    def unfreeze_last_n_blocks(self, n: int = 4):
        """
        Descongela os últimos N blocos transformer do backbone para
        fine-tuning parcial (útil quando freeze_backbone=True inicialmente).
        """
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        logger.info(f"Últimos {n} blocos do ViT descongelados para fine-tuning parcial.")


# ==============================================================================
# Extrator de features DINOv2 (interface com o pipeline existente)
# ==============================================================================

class ViTFeatureExtractor:
    """
    Substituto direto do bloco SqueezeNet no FeatureExtractor original.

    Uso básico (drop-in replacement)
    ---------------------------------
    Instancie esta classe e chame `extract_embedding(img)` no lugar de
    `_extract_squeezenet(img)`. A dimensão de saída é `embed_dim` do
    backbone escolhido (768 para vitb14).

    Integração com o pipeline híbrido
    -----------------------------------
    No FeatureExtractor original, substitua:
        sq_feats = self._extract_squeezenet(img)   # 1000-dim
    por:
        sq_feats = self.vit_extractor.extract_embedding(img)  # 768-dim
    e ajuste a soma no vetor híbrido: 128 (SIFT) + 768 (ViT) = 896 atributos.
    """

    def __init__(
        self,
        variant: str = "dinov2_vitb14",
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = DINOv2Classifier(variant=variant, num_classes=0)
        self.model.eval()
        self.model.to(self.device)

        # DINOv2 foi treinado com patch de 14 px; múltiplos de 14 são ideais.
        # 224 (16×14) e 518 (37×14) são opções comuns.
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    @torch.no_grad()
    def extract_embedding(self, img: Image.Image) -> np.ndarray:
        """Retorna o embedding [CLS] como array numpy (embed_dim,)."""
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        embedding = self.model.get_embedding(tensor)
        return embedding.squeeze(0).cpu().numpy()

    def extract_from_path(self, img_path: str) -> np.ndarray:
        """Conveniência: abre a imagem pelo caminho e extrai o embedding."""
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            return self.extract_embedding(img)


# ==============================================================================
# Treinador supervisionado (fine-tuning completo)
# ==============================================================================

class ViTTrainer:
    """
    Loop de treinamento supervisionado para fine-tuning do DINOv2.

    Parâmetros
    ----------
    model : DINOv2Classifier
        Instância com num_classes > 0.
    metadata : pd.DataFrame
        DataFrame com colunas 'filepath' e 'label'.
    label_to_idx : dict
        Mapeamento string → inteiro das classes.
    epochs : int
        Número de épocas de treinamento.
    batch_size : int
        Tamanho do mini-batch.
    lr : float
        Taxa de aprendizagem inicial (AdamW).
    weight_decay : float
        Regularização L2 do AdamW.
    save_path : str | Path
        Diretório onde o melhor checkpoint será salvo.
    device : torch.device | None
        GPU/CPU. Detectado automaticamente se None.
    """

    def __init__(
        self,
        model: DINOv2Classifier,
        metadata: pd.DataFrame,
        label_to_idx: dict,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        save_path: Union[str, Path] = "workspace/models",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.epochs = epochs
        self.save_path = Path(save_path)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Transforms distintos para treino (com augmentação) e validação
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Split 80/20 estratificado por classe
        train_meta, val_meta = self._stratified_split(metadata, label_to_idx)

        train_ds = ThermalImageDataset(train_meta, label_to_idx, self.train_transform)
        val_ds = ThermalImageDataset(val_meta, label_to_idx, self.val_transform)

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

    @staticmethod
    def _stratified_split(
        metadata: pd.DataFrame, label_to_idx: dict, val_frac: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split 80/20 estratificado preservando distribuição de classes."""
        train_rows, val_rows = [], []
        for label in label_to_idx:
            subset = metadata[metadata["label"] == label]
            n_val = max(1, int(len(subset) * val_frac))
            val_rows.append(subset.sample(n=n_val, random_state=42))
            train_rows.append(subset.drop(val_rows[-1].index))
        return pd.concat(train_rows), pd.concat(val_rows)

    def train(self) -> List[dict]:
        """
        Executa o loop de treinamento e retorna o histórico de métricas.

        Retorna
        -------
        history : list[dict]
            Lista de dicionários com 'epoch', 'train_loss', 'val_loss', 'val_acc'.
        """
        best_val_acc = 0.0
        history = []
        ckpt_path = self.save_path / "dinov2_best.pt"
        self.save_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            # ── Treino ───────────────────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)

            # ── Validação ────────────────────────────────────────────────────
            val_loss, val_acc = self._evaluate()

            self.scheduler.step()

            logger.info(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

            # Salva o melhor checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_acc": val_acc,
                        "variant": self.model.variant,
                        "num_classes": self.model.num_classes,
                    },
                    ckpt_path,
                )
                logger.info(f"  → Checkpoint salvo (val_acc={val_acc:.4f})")

        logger.info(f"Treinamento concluído. Melhor val_acc: {best_val_acc:.4f}")
        return history

    @torch.no_grad()
    def _evaluate(self) -> Tuple[float, float]:
        """Calcula loss e acurácia no conjunto de validação."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(self.val_loader), correct / total


# ==============================================================================
# Utilitário: carregamento de checkpoint para inferência
# ==============================================================================

def load_finetuned_vit(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> DINOv2Classifier:
    """
    Carrega um checkpoint treinado e retorna o modelo em modo eval.

    Parâmetros
    ----------
    checkpoint_path : str | Path
        Caminho para o arquivo .pt salvo pelo ViTTrainer.
    device : torch.device | None
        Dispositivo alvo. Detectado automaticamente se None.

    Retorna
    -------
    model : DINOv2Classifier
        Modelo carregado e pronto para inferência.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = DINOv2Classifier(
        variant=ckpt["variant"],
        num_classes=ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    logger.info(
        f"Checkpoint carregado: {checkpoint_path} | "
        f"epoch={ckpt['epoch']} | val_acc={ckpt['val_acc']:.4f}"
    )
    return model


# ==============================================================================
# Patch de integração no FeatureExtractor original
# ==============================================================================

def patch_feature_extractor(feature_extractor_instance, variant: str = "dinov2_vitb14"):
    """
    Injeta o ViTFeatureExtractor como atributo `vit_extractor` em uma
    instância já criada do FeatureExtractor original, e faz monkey-patch
    do método `_extract_squeezenet` para usar o DINOv2.

    Uso
    ---
    extractor = FeatureExtractor(cfg)
    patch_feature_extractor(extractor, variant="dinov2_vitb14")
    # A partir daqui, extract_hybrid_features usa DINOv2 no lugar do SqueezeNet.

    Parâmetros
    ----------
    feature_extractor_instance : FeatureExtractor
        Instância do extrator existente no pipeline.
    variant : str
        Backbone DINOv2 desejado.
    """
    vit = ViTFeatureExtractor(variant=variant, device=feature_extractor_instance.device)
    feature_extractor_instance.vit_extractor = vit

    def _extract_squeezenet_dino(self, img: Image.Image) -> np.ndarray:
        return self.vit_extractor.extract_embedding(img)

    import types
    feature_extractor_instance._extract_squeezenet = types.MethodType(
        _extract_squeezenet_dino, feature_extractor_instance
    )

    # Atualiza o comentário interno de dimensão (informativo)
    feature_extractor_instance._hybrid_dim = 128 + DINOV2_VARIANTS[variant]
    logger.info(
        f"Patch aplicado: SqueezeNet substituído por DINOv2 ({variant}). "
        f"Dim híbrida: 128 (SIFT) + {DINOV2_VARIANTS[variant]} (ViT) = "
        f"{feature_extractor_instance._hybrid_dim}"
    )


# ==============================================================================
# Exemplo de uso independente
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ── 1. Extração de embeddings (sem treinamento) ───────────────────────────
    extractor = ViTFeatureExtractor(variant="dinov2_vitb14")
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    emb = extractor.extract_embedding(dummy_img)
    print(f"Embedding shape: {emb.shape}")   # (768,)

    # ── 2. Fine-tuning completo ───────────────────────────────────────────────
    metadata = pd.read_csv("workspace/cleaned/metadata.csv")
    classes = sorted(metadata["label"].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}

    model = DINOv2Classifier(
        variant="dinov2_vitb14",
        num_classes=len(classes),
        freeze_backbone=False,   # fine-tuning completo
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
    # Salva o melhor checkpoint em workspace/models/dinov2_best.pt
    model = load_finetuned_vit("workspace/models/dinov2_best.pt")
    # modelo em eval() automaticamente

    # ── 3. Linear probing (backbone congelado) ────────────────────────────────
    # model_lp = DINOv2Classifier(
    #     variant="dinov2_vitb14",
    #     num_classes=len(classes),
    #     freeze_backbone=True,    # linear probing
    # )
    # trainer_lp = ViTTrainer(model_lp, metadata, label_to_idx, epochs=10)
    # history_lp = trainer_lp.train()

    # ── 4. Patch direto no pipeline existente ────────────────────────────────
    # from VIT import FeatureExtractor, Config
    # cfg = Config()
    # extractor_pipeline = FeatureExtractor(cfg)
    # patch_feature_extractor(extractor_pipeline, variant="dinov2_vitb14")
    # X, y = extractor_pipeline.process_dataset(metadata)