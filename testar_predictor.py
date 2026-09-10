"""Testa o Predictor com uma imagem do dataset ou um caminho informado."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from thermal_vision_pipeline import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classifica uma imagem usando o modelo salvo do pipeline."
    )
    parser.add_argument(
        "imagem",
        nargs="?",
        help="Caminho da imagem a classificar.",
    )
    parser.add_argument(
        "--models-dir",
        default="thermal_vision_workspace/models",
        help="Pasta com os arquivos do modelo salvo.",
    )
    parser.add_argument(
        "--save",
        default="thermal_vision_workspace/reports/prediction_{stem}.png",
        help="Caminho do gráfico; por padrão usa thermal_vision_workspace/reports/prediction_<imagem>.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Abre a janela do gráfico após a classificação.",
    )
    args = parser.parse_args()

    image_path = args.imagem or input("Digite o caminho da imagem: ").strip()
    if not image_path:
        raise SystemExit("Nenhuma imagem foi informada.")

    image = Path(image_path)
    if not image.is_file():
        raise SystemExit(f"Imagem não encontrada: {image}")

    print("Carregando o modelo e o extrator DINOv2...", flush=True)
    predictor = Predictor(args.models_dir)
    print("Executando a classificação...", flush=True)
    predicted_class, probability = predictor.predict(str(image))

    print(f"Imagem: {image}")
    print(f"Classe prevista: {predicted_class}")
    print(f"Probabilidade: {probability:.2%}")

    output_path = Path(args.save.format(stem=image.stem))
    with Image.open(image) as source_image:
        plt.figure(figsize=(10, 7))
        plt.imshow(source_image.convert("RGB"))
        plt.title(
            f"Classe prevista: {predicted_class}\n"
            f"Probabilidade: {probability:.2%}"
        )
        plt.axis("off")
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em: {output_path}", flush=True)

        if args.show:
            plt.show()


if __name__ == "__main__":
    main()