# Reserved. The in-container Ollama stores its models in the `ollama-models`
# Docker named volume (see docker/docker-compose.gpu.yml), NOT in this folder.
# The list of models to auto-pull lives in models.toml (read by the ollama-init
# sidecar). This directory is currently unused; *.gguf/*.bin/.cache/ are
# gitignored just in case.
