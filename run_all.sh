#!/bin/bash
set -e # エラーが発生したら停止

echo "=========================================="
echo "       IEMOCAP Emotion Recognition"
echo "      5-Fold Cross Validation Script"
echo "=========================================="

# --- 1. データ前処理 ---
# 必要に応じてコメントアウトを解除してください
echo ""
echo "[Phase 1] Data Preprocessing..."
python create_label.py
python create_metadata.py
python create_bert_features.py
python create_wav2vec_features.py

# --- 2. 交差検証 (Fold 1-5) ---
echo ""
echo "[Phase 2] Starting 5-Fold Cross Validation..."

# ================= 設定項目 =================
SEED=30
EPOCHS=100
EVAL_METRIC=loss

# ★追加: バッチサイズと学習率
BATCH_SIZE=256
LR=8.84e-4

FOLDS=(1 2 3 4 5)
# ============================================

# モード配列 (ma2nma を含む)
MODES=("default" "nma")
MODES=("nma2ma") # テスト実行用（ma2nmaのみ実行したい場合）

# モダリティ設定
# MODALITIES=("multimodal" "text" "audio")
MODALITIES=("multimodal")

for mode in "${MODES[@]}"; do
    mode_args=()
    
    # モードに応じたフラグ設定
    if [ "$mode" = "nma" ]; then
        mode_args+=(--nma)
    elif [ "$mode" = "ma2nma" ]; then
        mode_args+=(--train_ma_test_nma)
    elif [ "$mode" = "nma2ma" ]; then
        mode_args+=(--train_nma_test_ma)
    fi

    for modality in "${MODALITIES[@]}"; do
        echo ""
        echo "===== Mode: ${mode^^} | Modality: ${modality} ====="
        
        # ディレクトリ作成
        mkdir -p "saved_models/seed${SEED}/${mode}/${modality}"

        echo ""
        echo "[Phase 2] Starting 5-Fold Cross Validation (${mode^^}/${modality})..."
        for i in "${FOLDS[@]}"; do
            echo ""
            echo "--- Running Fold $i (Test Session $i) ---"
            # ★修正: --batch_size と --lr を追加
            python run.py \
                --test_session "$i" \
                --eval_metric "$EVAL_METRIC" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --lr "$LR" \
                --seed "$SEED" \
                --modality "$modality" \
                "${mode_args[@]}"
        done

        echo ""
        echo "[Phase 3] Calculating Average Scores (${mode^^}/${modality})..."
        python calculate_average.py \
            --eval_metric "$EVAL_METRIC" \
            --seed "$SEED" \
            --modality "$modality" \
            "${mode_args[@]}"

        echo ""
        echo "[Phase 4] Plotting Training Logs (${mode^^}/${modality})..."
        python plot_log.py \
            --eval_metric "$EVAL_METRIC" \
            --seed "$SEED" \
            --modality "$modality" \
            "${mode_args[@]}"

        echo ""
        echo "[Phase 5] Analyzing Errors..."
        # 保存先パスの指定
        TARGET_DIR="saved_models/seed${SEED}/${mode}/${modality}"
        
        if [ -d "$TARGET_DIR" ]; then
            python analyze_errors.py "$TARGET_DIR" --output_dir "$TARGET_DIR"
        else
            echo "Warning: Directory not found: $TARGET_DIR"
        fi
    done # modalityループの終わり
done

echo ""
echo "=========================================="
echo "           All Tasks Completed!"
echo "=========================================="