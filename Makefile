# ============================================================
# Makefile for PJM Load Forecasting Project
# - Supports:
#     * make predictions  (used by Docker autograder)
#     * make clean        (remove intermediates, keep raw + notebooks)
#     * make rawdata      (delete & re-download raw data - OPTIONAL stub)
#     * make all          (lightweight default: does NOT run heavy pipeline)
# ============================================================

RAW_DIR       := data/raw
MERGED_DIR    := data/merged
PROCESSED_DIR := data/processed
MODELS_DIR    := models

.PHONY: all analysis predictions clean rawdata process train

# ------------------------------------------------------------
# Default target
#   We keep this LIGHT so that `make` inside Docker doesn't
#   accidentally rerun your whole analysis.
# ------------------------------------------------------------
all: analysis

analysis:
	@echo "Analysis target not fully automated."
	@echo "See code.ipynb for full workflow."
	@echo "For the challenge, the grader will run: make predictions"

# ------------------------------------------------------------
# make predictions
#   - This is what the instructor / autograder will run via:
#       docker run -it --rm yourdockerhubname/yourimagename make predictions
#   - It calls the make_predictions() function defined in:
#       src/make_predictions.py
# ------------------------------------------------------------
predictions:
	python -m src.make_predictions

# ------------------------------------------------------------
# make clean
#   - deletes intermediate products but leaves:
#       * raw data in data/raw
#       * notebooks, code, and models
#   - This matches the project requirement that raw data and
#     notebooks are NOT deleted by clean.
# ------------------------------------------------------------
clean:
	rm -rf $(PROCESSED_DIR)/*
	rm -rf $(MERGED_DIR)/*
	rm -rf figures
	find . -name "*.log" -delete

# ------------------------------------------------------------
# make rawdata
#   - deletes and (optionally) re-downloads raw data
#   - NOTE: we ONLY delete here, not in `clean`
#   - If you have a real download script, plug it in below.
# ------------------------------------------------------------
rawdata:
	rm -rf $(RAW_DIR)/*
	@echo "Raw data deleted from $(RAW_DIR)."
	@echo "TODO: call your raw data download script here, e.g.:"
	@echo "python src/download_raw_data.py"

# ------------------------------------------------------------
# process / train
#   - Stubs for reproducibility; you can wire them to your
#     own scripts if desired, but they are NOT used by the
#     autograder. Left as echoes to avoid accidental long runs.
# ------------------------------------------------------------
process:
	@echo "process target not wired to scripts."
	@echo "If desired, hook this to your data cleaning/merging pipeline."

train:
	@echo "train target not wired to scripts."
	@echo "You already trained models and saved them in $(MODELS_DIR)."
