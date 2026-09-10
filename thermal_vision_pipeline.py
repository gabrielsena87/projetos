"""
hybrid_fusion_pipeline.py

Pipeline completo de classificação de equipamentos elétricos utilizando
imagens térmicas, com fusão híbrida (SqueezeNet + SIFT), seleção supervisionada
de características, redução de dimensionalidade e Stacking Ensemble.
"""

import os
import zipfile
import logging
import json
import joblib
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, List, Dict, Any, Optional, Union
from dinov2_models import patch_feature_extractor

if TYPE_CHECKING:
    from pandas import DataFrame

import importlib

try:
    cv2 = importlib.import_module("cv2")
except ImportError as exc:
    raise ImportError(
        "OpenCV não está instalado. Instale-o com 'pip install opencv-python'."
    ) from exc
import numpy as np
try:
    pd = importlib.import_module("pandas")
except ImportError as exc:
    raise ImportError(
        "Pandas não está instalado. Instale-o com 'pip install pandas'."
    ) from exc
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.squeezenet import SqueezeNet1_1_Weights

try:
    _sklearn_preprocessing = importlib.import_module("sklearn.preprocessing")
    StandardScaler = _sklearn_preprocessing.StandardScaler
    LabelEncoder = _sklearn_preprocessing.LabelEncoder
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, cross_val_predict
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score, precision_score,
        recall_score, f1_score, confusion_matrix, classification_report,
    )
    from sklearn.pipeline import Pipeline
except ImportError as exc:
    raise ImportError(
        "scikit-learn não está instalado. Instale-o com 'pip install scikit-learn'."
    ) from exc


# ==============================================================================
# Configurações Globais
# ==============================================================================

@dataclass
class Config:
    """Configurações e hiperparâmetros globais do pipeline."""
    # Diretórios
    workspace_dir: Path = Path("thermal_vision_workspace")
    raw_dir: Path = workspace_dir / "raw"
    extracted_dir: Path = workspace_dir / "extracted"
    cleaned_dir: Path = workspace_dir / "cleaned"
    features_dir: Path = workspace_dir / "features"
    models_dir: Path = workspace_dir / "models"
    reports_dir: Path = workspace_dir / "reports"
    
    # Arquivos
    dataset_dir: Path = Path(r"D:\hd_central\projetos\thermal_equipment_dataset")
    dataset_zip: str = "dataset.zip"
    metadata_csv: Path = cleaned_dir / "metadata.csv"
    X_npy: Path = features_dir / "X.npy"
    y_npy: Path = features_dir / "y.npy"
    
    # Pré-processamento e Modelos
    img_size: Tuple[int, int] = (224, 224)
    imagenet_mean: List[float] = (0.485, 0.456, 0.406)
    imagenet_std: List[float] = (0.229, 0.224, 0.225)
    
    # Espaço de Busca
    kbest_candidates: Tuple[int, ...] = (50, 100, 150, 200, 300, 500)
    pca_candidates: Tuple[float, ...] = (0.90, 0.95, 0.97, 0.99)
    
    knn_neighbors: Tuple[int, ...] = (3, 5, 7, 9, 11)
    rf_estimators: Tuple[int, ...] = (100, 200, 300, 500)
    svm_c: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    
    random_state: int = 42

def setup_logging() -> logging.Logger:
    """Configura o logger padrão da aplicação."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("HybridPipeline")

logger = setup_logging()


class ZipArchiveBase(ABC):
    """Interface de alto nível para manipular arquivos ZIP com segurança."""

    @abstractmethod
    def list_files(self) -> List[str]:
        """Retorna a lista de caminhos dentro do arquivo compactado."""
        raise NotImplementedError

    @abstractmethod
    def read_file_bytes(self, member_name: str) -> bytes:
        """Lê um arquivo interno em bytes."""
        raise NotImplementedError

    @abstractmethod
    def extract_all(self, destination: Union[str, Path]) -> List[Path]:
        """Extrai o conteúdo do zip em um diretório seguro."""
        raise NotImplementedError


class SafeZipFile(ZipArchiveBase):
    """Implementação concreta e segura do acesso a ZIPs para uso no pipeline."""

    def __init__(self, archive_path: Union[str, os.PathLike], mode: str = 'r'):
        self.archive_path = Path(archive_path)
        self.mode = mode
        self._zip = zipfile.ZipFile(self.archive_path, mode=self.mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._zip.close()

    def list_files(self) -> List[str]:
        return self._zip.namelist()

    def read_file_bytes(self, member_name: str) -> bytes:
        return self._zip.read(member_name)

    def extract_all(self, destination: Union[str, Path]) -> List[Path]:
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        extracted: List[Path] = []

        for member in self._zip.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise ValueError(
                    f"Entrada ZIP fora do diretório de destino: {member.filename!r}"
                ) from exc

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with self._zip.open(member, 'r') as source, target.open('wb') as dest:
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
            extracted.append(target)

        return extracted


# ==============================================================================
# Gerenciamento de Dados
# ==============================================================================

class DataManager:
    """Responsável por extrair, validar, limpar e organizar o dataset de imagens."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._create_directories()

    def _create_directories(self):
        """Cria a árvore de diretórios do workspace."""
        dirs = [
            self.cfg.workspace_dir, self.cfg.raw_dir, self.cfg.extracted_dir,
            self.cfg.cleaned_dir, self.cfg.features_dir, self.cfg.models_dir,
            self.cfg.reports_dir
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Estrutura do workspace verificada e inicializada.")

    def extract_dataset(self):
        """Prepara o dataset, usando a pasta configurada ou um ZIP como fallback."""
        if self.cfg.dataset_dir.exists():
            if not self.cfg.dataset_dir.is_dir():
                raise NotADirectoryError(
                    f"O caminho do dataset não é uma pasta: '{self.cfg.dataset_dir}'."
                )
            logger.info(f"Dataset encontrado em {self.cfg.dataset_dir}. Extração não necessária.")
            return

        zip_path = Path(self.cfg.dataset_zip)
        if not zip_path.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado em '{self.cfg.dataset_dir}' "
                f"e arquivo ZIP '{zip_path}' também não existe."
            )
        
        logger.info(f"Extraindo {zip_path}...")
        destination = self.cfg.extracted_dir.resolve()
        with SafeZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract_all(destination)
        logger.info("Extração concluída com sucesso.")

    def clean_and_organize(self) -> "DataFrame":
        """Limpa imagens corrompidas e gera o metadado relacional estruturado."""
        logger.info("Verificando integridade das imagens...")
        metadata = []
        
        source_dir = (
            self.cfg.dataset_dir
            if self.cfg.dataset_dir.exists()
            else self.cfg.extracted_dir
        )

        # Resolve o dataset root caso o ZIP tenha criado uma subpasta extra
        dataset_root: Optional[Path] = None
        for root, dirs, files in os.walk(source_dir):
            if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
                dataset_root = Path(root).parent
                break
                 
        if dataset_root is None:
            dataset_root = source_dir

        class_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
        class_names = sorted([d.name for d in class_dirs])
        
        joblib.dump(class_names, self.cfg.models_dir / "class_names.joblib")
        logger.info(f"Classes identificadas automaticamente: {class_names}")

        for cls_name in class_names:
            cls_out_dir = self.cfg.cleaned_dir / cls_name
            cls_out_dir.mkdir(exist_ok=True)
            
            cls_in_dir = dataset_root / cls_name
            valid_idx = 0
            
            for img_path in cls_in_dir.glob("*.*"):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                    
                try:
                    # Validar a integridade sem carregar toda a imagem para RAM
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    # Reabrir e aplicar normalização básica de formato
                    with Image.open(img_path) as img:
                        img_rgb = img.convert('RGB')
                        new_filename = f"{cls_name}_{valid_idx:04d}.jpg"
                        new_filepath = cls_out_dir / new_filename
                        img_rgb.save(new_filepath)
                        
                        metadata.append({'filepath': str(new_filepath), 'label': cls_name})
                        valid_idx += 1
                        
                except Exception as e:
                    logger.warning(f"Imagem corrompida removida: {img_path}. Causa: {str(e)}")
                    
        df = pd.DataFrame(metadata)
        df.to_csv(self.cfg.metadata_csv, index=False)
        logger.info(f"Metadata criado: {len(df)} imagens válidas registradas.")
        
        return df


# ==============================================================================
# Extração Híbrida de Características
# ==============================================================================

class FeatureExtractor:
    """Implementa o pipeline híbrido extraindo de descritores profundos e locais."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Deep Feature Extractor inicializado no device: {self.device}")
        
        # Carregamento SqueezeNet 1.1
        weights = SqueezeNet1_1_Weights.IMAGENET1K_V1
        self.sq_model = models.squeezenet1_1(weights=weights)
        self.sq_model.eval()
        self.sq_model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize(self.cfg.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.imagenet_mean, std=self.cfg.imagenet_std)
        ])
        
        # Inicialização do Extrator SIFT Local
        self.sift = cv2.SIFT_create()

    def _extract_squeezenet(self, img: Image.Image) -> np.ndarray:
        """Extrai 1000 features provenientes das ativações profundas."""
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.sq_model(tensor)
        return output.cpu().numpy().flatten()

    def _extract_sift(self, img: Image.Image) -> np.ndarray:
        """Extrai as 128 features através de descritores keypoints (SIFT)."""
        img_resized = img.resize(self.cfg.img_size)
        img_cv = np.array(img_resized)
        
        img_bgr = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(128)
        
        return descriptors.mean(axis=0)

    def extract_hybrid_features(self, img_path: str) -> np.ndarray:
        """Concatenação: 128 (SIFT) + 1000 (SqueezeNet) = 1128 atributos."""
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                sq_feats = self._extract_squeezenet(img)
                sift_feats = self._extract_sift(img)
                
                # Fusão Híbrida Serial
                hybrid_feats = np.concatenate([sift_feats, sq_feats])
                return hybrid_feats
        except Exception as e:
            logger.error(f"Erro durante extração da imagem {img_path}: {e}")
            return np.zeros(1128)

    def process_dataset(self, metadata: "DataFrame") -> Tuple[np.ndarray, np.ndarray]:
        """Itera o dataset e salva os blobs multidimensionais."""
        logger.info("Processamento de Features em lote iniciado...")
        X, y = [], []
        
        for idx, row in metadata.iterrows():
            feats = self.extract_hybrid_features(row['filepath'])
            X.append(feats)
            y.append(row['label'])
            
            if (idx + 1) % 50 == 0 or (idx + 1) == len(metadata):
                logger.info(f"Extraído: {idx + 1}/{len(metadata)}")
                
        X = np.array(X)
        y = np.array(y)
        
        np.save(self.cfg.X_npy, X)
        np.save(self.cfg.y_npy, y)
        logger.info(f"Features consolidadas. Matriz de Atributos (X): {X.shape}")
        
        return X, y


# ==============================================================================
# Otimizador de Pipeline
# ==============================================================================

class PipelineOptimizer:
    """Implementa toda a engenharia, seleção, redução e o Stacking Ensemble."""
    
    def __init__(self, cfg: Config, X: np.ndarray, y: np.ndarray):
        self.cfg = cfg
        self.X = X
        self.y = y
        self.le = LabelEncoder()
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.cfg.random_state)

    def optimize(self) -> Tuple[Pipeline, np.ndarray]:
        """Aplica busca baseada na acurácia média de Validação Cruzada."""
        y_enc = self.le.fit_transform(self.y)
        
        logger.info("Aplicando Scaling e Variance Threshold...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        var_thresh = VarianceThreshold(threshold=0.001)
        X_var = var_thresh.fit_transform(X_scaled)
        
        best_k = self._optimize_kbest(X_var, y_enc)
        kbest = SelectKBest(score_func=mutual_info_classif, k=best_k)
        X_kbest = kbest.fit_transform(X_var, y_enc)
        
        best_pca_var = self._optimize_pca(X_kbest, y_enc)
        pca = PCA(n_components=best_pca_var, random_state=self.cfg.random_state)
        pca.fit(X_kbest)
        self._plot_explained_variance(pca)
        
        stacking_clf = self._build_stacking_classifier()
        
        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('variance', VarianceThreshold(threshold=0.001)),
            ('kbest', SelectKBest(score_func=mutual_info_classif, k=best_k)),
            ('pca', PCA(n_components=best_pca_var, random_state=self.cfg.random_state)),
            ('classifier', stacking_clf)
        ])
        
        return final_pipeline, y_enc

    def _optimize_kbest(self, X: np.ndarray, y: np.ndarray) -> int:
        """Busca O(N) do melhor K que maximiza acurácia com modelo rápido."""
        logger.info("Otimizando número de atributos (SelectKBest)...")
        results = []
        base_clf = LogisticRegression(max_iter=1000, random_state=self.cfg.random_state)
        
        for k in self.cfg.kbest_candidates:
            if k > X.shape[1]:
                continue
                
            kb = SelectKBest(score_func=mutual_info_classif, k=k)
            X_tmp = kb.fit_transform(X, y)
            
            scores = cross_val_score(base_clf, X_tmp, y, cv=self.cv, scoring='accuracy')
            mean_acc = np.mean(scores)
            results.append({'k': k, 'mean_accuracy': mean_acc})
            logger.info(f"Testando K={k} -> Accuracy: {mean_acc:.4f}")
            
        df_res = pd.DataFrame(results)
        df_res.to_csv(self.cfg.reports_dir / "feature_selection_results.csv", index=False)
        
        best_k = df_res.loc[df_res['mean_accuracy'].idxmax()]['k']
        logger.info(f"Otimização concluída. Melhor K: {int(best_k)}")
        return int(best_k)

    def _optimize_pca(self, X: np.ndarray, y: np.ndarray) -> float:
        """Identifica a variância limite ótima do PCA."""
        logger.info("Otimizando componentes PCA...")
        results = []
        base_clf = LogisticRegression(max_iter=1000, random_state=self.cfg.random_state)
        
        for var in self.cfg.pca_candidates:
            pca = PCA(n_components=var, random_state=self.cfg.random_state)
            X_tmp = pca.fit_transform(X)
            
            scores = cross_val_score(base_clf, X_tmp, y, cv=self.cv, scoring='accuracy')
            mean_acc = np.mean(scores)
            results.append({'variance': var, 'n_components': X_tmp.shape[1], 'mean_accuracy': mean_acc})
            logger.info(f"Testando PCA var={var} (C={X_tmp.shape[1]}) -> Accuracy: {mean_acc:.4f}")
            
        df_res = pd.DataFrame(results)
        df_res.to_csv(self.cfg.reports_dir / "pca_results.csv", index=False)
        
        best_var = df_res.loc[df_res['mean_accuracy'].idxmax()]['variance']
        logger.info(f"Otimização concluída. Variância PCA Ótima: {best_var}")
        return best_var

    def _plot_explained_variance(self, pca_model: PCA):
        """Exporta gráfico da razão de variância."""
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(pca_model.explained_variance_ratio_), marker='o', linestyle='--')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Variância Explicada Acumulada')
        plt.title('Análise de Componentes Principais (PCA)')
        plt.grid(True)
        plt.savefig(self.cfg.reports_dir / "explained_variance.png")
        plt.close()

    def _build_stacking_classifier(self) -> StackingClassifier:
        """Configura os Modelos Base e Nível 1 do Stacking Ensemble."""
        logger.info("Construindo e Configurando Classificador Stacking Ensemble...")
        
        # Para um ambiente produtivo real, GridSearchCV seria rodado aqui.
        # Estamos provisionando classificadores com parâmetros representativos robustos.
        level0 = [
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('rf', RandomForestClassifier(n_estimators=300, random_state=self.cfg.random_state)),
            ('svm', SVC(kernel='rbf', C=10.0, probability=True, random_state=self.cfg.random_state))
        ]
        
        level1 = LogisticRegression(random_state=self.cfg.random_state)
        
        return StackingClassifier(estimators=level0, final_estimator=level1, cv=5)


# ==============================================================================
# Avaliador e Relatórios de Desempenho
# ==============================================================================

class ModelEvaluator:
    """Extrai métricas operacionais e processa o armazenamento dos artefatos finais."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.cfg.random_state)

    def evaluate_and_save(self, pipeline: Pipeline, X: np.ndarray, y_enc: np.ndarray, label_encoder: LabelEncoder):
        logger.info("Executando K-Fold Validação Estratificada em todo o Pipeline...")
        
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        cv_results = cross_validate(pipeline, X, y_enc, cv=self.cv, scoring=scoring, n_jobs=1)
        
        pd.DataFrame(cv_results).to_csv(self.cfg.reports_dir / "cv_results.csv", index=False)
        
        mean_acc = np.mean(cv_results['test_accuracy'])
        logger.info(f"[+] Acurácia Pipeline Média: {mean_acc:.4f}")
        
        # Computando Matriz e Métricas Estáticas Globalizadas
        y_pred = cross_val_predict(pipeline, X, y_enc, cv=self.cv, n_jobs=1)
        class_names = label_encoder.classes_
        
        acc = accuracy_score(y_enc, y_pred)
        kappa = cohen_kappa_score(y_enc, y_pred)
        logger.info(f"[+] Acurácia Global de Referência: {acc:.4f}")
        logger.info(f"[+] Cohen's Kappa Index: {kappa:.4f}")
        
        report = classification_report(y_enc, y_pred, target_names=class_names, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(self.cfg.reports_dir / "classification_report.csv")
        
        cm = confusion_matrix(y_enc, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predição Modelo')
        plt.ylabel('Ground Truth')
        plt.title('Matriz de Confusão Híbrida')
        plt.tight_layout()
        plt.savefig(self.cfg.reports_dir / "confusion_matrix.png")
        plt.close()
        
        # Consolidação do treinamento para Produção (Fit final com todos os dados)
        logger.info("Ajustando Pesos Finais no Dataset Completo...")
        pipeline.fit(X, y_enc)
        joblib.dump(pipeline, self.cfg.models_dir / "hybrid_stacking_model.joblib")
        
        # Registro Técnico
        config_dict = {
            'img_size': self.cfg.img_size,
            'imagenet_mean': self.cfg.imagenet_mean,
            'imagenet_std': self.cfg.imagenet_std,
            'classes': list(class_names)
        }
        with open(self.cfg.models_dir / "pipeline_config.json", 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        logger.info("Pipeline Final Salvo na camada de Persistência!")


# ==============================================================================
# Módulo de Inferência de Produção
# ==============================================================================

class Predictor:
    """Classe encapsulada para carga do pipeline e consumo de inferências live."""
    
    def __init__(self, models_dir: str = "thermal_vision_workspace/models"):
        self.models_dir = Path(models_dir)
        
        self.pipeline = joblib.load(self.models_dir / "hybrid_stacking_model.joblib")
        self.class_names = joblib.load(self.models_dir / "class_names.joblib")
        
        with open(self.models_dir / "pipeline_config.json", 'r') as f:
            config_dict = json.load(f)
            
        cfg = Config(
            img_size=tuple(config_dict['img_size']),
            imagenet_mean=config_dict['imagenet_mean'],
            imagenet_std=config_dict['imagenet_std']
        )
        self.extractor = FeatureExtractor(cfg)
        patch_feature_extractor(self.extractor, variant="dinov2_vitb14")

    def predict(self, img_path: str) -> Tuple[str, float]:
        """Realiza classificação zero-shot com os vetores do novo Input."""
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Imagem ausente no caminho especificado: {img_path}")
            
        feats = self.extractor.extract_hybrid_features(img_path)
        X_new = feats.reshape(1, -1)
        
        pred_idx = self.pipeline.predict(X_new)[0]
        probs = self.pipeline.predict_proba(X_new)[0]
        
        pred_class = self.class_names[pred_idx]
        pred_prob = float(probs[pred_idx])
        
        return pred_class, pred_prob


# ==============================================================================
# Ponto de Entrada (Bootstrapper)
# ==============================================================================

def main():
    """Executor principal do ciclo de vida do Machine Learning."""
    try:
        cfg = Config()
        
        dm = DataManager(cfg)
        dm.extract_dataset()
        metadata = dm.clean_and_organize()
        
        if metadata.empty:
            logger.error("Interrupção: Nenhuma imagem válida retornada.")
            return
        
        extractor = FeatureExtractor(cfg)
        patch_feature_extractor(extractor, variant="dinov2_vitb14")
        # Pronto — extract_hybrid_features agora usa DINOv2
        
        if cfg.X_npy.exists() and cfg.y_npy.exists():
            logger.info("Detectado cache de features prévio, carregando-os (X.npy, y.npy).")
            X = np.load(cfg.X_npy)
            y = np.load(cfg.y_npy)
        else:
            X, y = extractor.process_dataset(metadata)

        optimizer = PipelineOptimizer(cfg, X, y)
        pipeline, y_enc = optimizer.optimize()
        
        evaluator = ModelEvaluator(cfg)
        evaluator.evaluate_and_save(pipeline, X, y_enc, optimizer.le)
        
        logger.info("Sucesso! O Pipeline Híbrido foi concluído e os dados modelados.")
        
    except Exception as e:
        logger.exception(f"Erro fatal na orquestração: {str(e)}")


if __name__ == "__main__":
    main()