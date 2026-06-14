import kagglehub
from pathlib import Path

# Download latest version
ravdess_path = kagglehub.dataset_download(
    handle="uwrfkaggler/ravdess-emotional-speech-audio",
)
print("Path to RAVDESS dataset files:", ravdess_path)

crema_d_path = kagglehub.dataset_download(
    handle="ejlok1/cremad",
)
print("Path to CREMA-D dataset files:", crema_d_path)

tess_path = kagglehub.dataset_download(
    handle="ejlok1/toronto-emotional-speech-set-tess",
)
print("Path to TESS dataset files:", tess_path)