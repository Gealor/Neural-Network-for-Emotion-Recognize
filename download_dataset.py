import kagglehub
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Download latest version
path = kagglehub.dataset_download(
    handle="uwrfkaggler/ravdess-emotional-speech-audio"
)

print("Path to dataset files:", path)