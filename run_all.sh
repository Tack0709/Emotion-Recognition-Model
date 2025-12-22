#!/bin/bash
set -e # エラーが発生したら停止

echo "=========================================="
echo "       IEMOCAP Emotion Recognition"
echo "      5-Fold Cross Validation Script"
echo "=========================================="

# --- 1. データ前処理 ---
echo ""
echo "[Phase 1] Data Preprocessing..."
python create_label.py
python create_metadata.py
python create_bert_features.py
python create_wav2vec_features.py

# --- 2. 交差検証 (Fold 1-5) ---
echo ""
echo "[Phase 2] Starting 5-Fold Cross Validation..."

SEED=100
EPOCHS=100
EVAL_METRIC=loss
FOLDS=(1 2 3 4 5)
MODES=("default" "nma")
MODALITIES=("multimodal" "text" "audio")

for mode in "${MODES[@]}"; do
    mode_args=()
    [ "$mode" = "nma" ] && mode_args+=(--nma)

    for modality in "${MODALITIES[@]}"; do
        echo ""
        echo "===== Mode: ${mode^^} | Modality: ${modality} ====="
        mkdir -p "saved_models/seed${SEED}/${mode}/${modality}"

        echo ""
        echo "[Phase 2] Starting 5-Fold Cross Validation (${mode^^}/${modality})..."
        for i in "${FOLDS[@]}"; do
            echo ""
            echo "--- Running Fold $i (Test Session $i) ---"
            python run.py --test_session "$i" --eval_metric "$EVAL_METRIC" --epochs "$EPOCHS" --seed "$SEED" --modality "$modality" "${mode_args[@]}"
        done

        echo ""
        echo "[Phase 3] Calculating Average Scores (${mode^^}/${modality})..."
        python calculate_average.py --eval_metric "$EVAL_METRIC" --seed "$SEED" --modality "$modality" "${mode_args[@]}"

        echo ""
        echo "[Phase 4] Plotting Training Logs (${mode^^}/${modality})..."
        python plot_log.py --eval_metric "$EVAL_METRIC" --seed "$SEED" --modality "$modality" "${mode_args[@]}"
    done
done

echo ""
echo "=========================================="
echo "           All Tasks Completed!"
echo "=========================================="