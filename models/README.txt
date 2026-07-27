# Flat GGUF downloads from pull_models_before_build.py land here, e.g.
#   models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
#
# The script then imports each into the in-container Ollama (gpu-amd /
# gpu-nvidia profile) via /api/blobs + /api/create. Ollama keeps its own copy
# in the ollama-models volume; the file here is your inspectable/reusable source.
#
# Configure the download list in models.toml; set MODELS_DIR in .env to relocate.
# *.gguf and *.bin are gitignored (see .gitignore).
