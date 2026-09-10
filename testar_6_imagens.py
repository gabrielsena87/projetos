"""Classifica e plota seis imagens de classes diferentes do dataset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from thermal_vision_pipeline import Predictor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def select_images(dataset_dir: Path, amount: int = 6) -> list[tuple[Path, str]]:
    class_images: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            class_images[class_dir.name] = images

    if len(class_images) < 5:
        raise SystemExit(
            f"São necessárias pelo menos 5 classes com imagens; encontradas: {len(class_images)}."
        )

    selected = [(images[0], class_name) for class_name, images in class_images.items()]
    selected = selected[:amount]

    while len(selected) < amount:
        class_name, images = next(iter(class_images.items()))
        image_index = min(len(selected) - len(class_images) + 1, len(images) - 1)
        selected.append((images[image_index], class_name))

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classifica seis imagens e salva um painel comparativo."
    )
    parser.add_argument(
        "--dataset-dir",
        default="thermal_equipment_dataset",
        help="Pasta raiz do dataset organizada por classe.",
    )
    parser.add_argument(
        "--models-dir",
        default="thermal_vision_workspace/models",
        help="Pasta com os arquivos do modelo salvo.",
    )
    parser.add_argument(
        "--save",
        default="thermal_vision_workspace/reports/prediction_6_classes.png",
        help="Caminho do painel PNG gerado.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Abre o painel após salvá-lo.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset não encontrado: {dataset_dir}")

    selected_images = select_images(dataset_dir)
    print("Foram selecionadas 6 imagens:", flush=True)
    for image, true_class in selected_images:
        print(f"- {true_class}: {image}", flush=True)

    print("Carregando o modelo e o extrator DINOv2...", flush=True)
    predictor = Predictor(args.models_dir)
    predictions = []

    for image, true_class in selected_images:
        print(f"Classificando: {image.name}", flush=True)
        predicted_class, probability = predictor.predict(str(image))
        predictions.append((image, true_class, predicted_class, probability))
        print(
            f"  Real: {true_class} | Prevista: {predicted_class} | "
            f"Probabilidade: {probability:.2%}",
            flush=True,
        )

    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    for axis, (image, true_class, predicted_class, probability) in zip(
        axes.flat, predictions
    ):
        with Image.open(image) as source_image:
            axis.imshow(source_image.convert("RGB"))
        color = "darkgreen" if true_class == predicted_class else "darkred"
        axis.set_title(
            f"Real: {true_class}\n"
            f"Prevista: {predicted_class} ({probability:.2%})",
            color=color,
            fontsize=10,
        )
        axis.axis("off")

    figure.suptitle("Classificação de 6 imagens", fontsize=16)
    figure.tight_layout()
    output_path = Path(args.save)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Painel salvo em: {output_path}", flush=True)

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
