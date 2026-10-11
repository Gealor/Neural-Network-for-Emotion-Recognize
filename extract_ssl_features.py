"""Извлечение замороженных SSL-эмбеддингов (WavLM/HuBERT) — Phase C.

Читает paths_{split}.npy/y_{split}.npy/groups_{split}.npy из
config.SSL_OUTPUT_DIR (см. prepare_data/ssl_manifest.py и prepare_ssl_data.py),
прогоняет каждый файл через замороженный энкодер (eval(), requires_grad=False)
и сохраняет mean-pooled по времени эмбеддинг последнего hidden-слоя одной
строкой на файл в processed_data_ssl/embeddings/{model}_{split}.npy.
y_{split}.npy/groups_{split}.npy применяются к эмбеддингам как есть —
порядок строк не меняется.
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

import config
from prepare_data.pipelines.audio.io import load_audio

MODELS = {
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
    "wavlm-large": "microsoft/wavlm-large",
    "hubert-large-ll60k": "facebook/hubert-large-ll60k",
}

SR = 16000
TRIM_TOP_DB = 35
SPLITS = ("train", "val", "test")


def load_manifest(type_dataset: str) -> List[Path]:
    paths = np.load(config.SSL_OUTPUT_DIR / f"paths_{type_dataset}.npy")
    return [config.ROOT_PATH / p for p in paths]


def load_model(model_key: str, device: torch.device):
    repo_id = MODELS[model_key]
    feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)
    model = AutoModel.from_pretrained(repo_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    return feature_extractor, model


def extract_embeddings(feature_extractor, model, files: List[Path], device: torch.device) -> np.ndarray:
    embeddings = []
    for i, file in enumerate(files, 1):
        audio, _ = load_audio(file, sr=SR, trim_top_db=TRIM_TOP_DB)
        inputs = feature_extractor(audio, sampling_rate=SR, return_tensors="pt")
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            hidden = model(input_values).last_hidden_state  # (1, T, D)

        embeddings.append(hidden.mean(dim=1).squeeze(0).cpu().numpy())

        if i % 200 == 0:
            print(f"  {i}/{len(files)}")

    return np.stack(embeddings).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="wavlm-base-plus")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число файлов на сплит (смоук-тест)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Модель: {args.model} ({MODELS[args.model]}), устройство: {device}")

    feature_extractor, model = load_model(args.model, device)

    out_dir = config.SSL_OUTPUT_DIR / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    for type_dataset in args.splits:
        files = load_manifest(type_dataset)
        if args.limit is not None:
            files = files[: args.limit]
        if not files:
            print(f"\n{type_dataset}: нет файлов, пропускаем")
            continue

        print(f"\n--- {type_dataset}: {len(files)} файлов ---")
        embeddings = extract_embeddings(feature_extractor, model, files, device)

        out_path = out_dir / f"{args.model}_{type_dataset}.npy"
        np.save(out_path, embeddings)
        print(f"Сохранено: {out_path} shape={embeddings.shape}")


if __name__ == "__main__":
    main()
