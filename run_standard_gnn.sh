#!/bin/bash
set -e # エラーが発生したら停止

echo "=========================================="
echo "      Standard GNN (Ablation Study)"
echo "      5-Fold Cross Validation Script"
echo "=========================================="

# --- 1. データ前処理 ---
# ※すでに完了している場合はコメントアウトしてください
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

# Standard GNN は基本的に multimodal で比較実験する想定
# 必要であれば "text" "audio" を追加してください
MODALITIES=("multimodal")

# 必要であれば "nma" も追加可能: MODES=("default" "nma")
MODES=("default" "nma")

for mode in "${MODES[@]}"; do
    mode_args=()
    [ "$mode" = "nma" ] && mode_args+=(--nma)

    for modality in "${MODALITIES[@]}"; do
        echo ""
        echo "===== Mode: ${mode^^} | Modality: ${modality} | Arch: Standard GNN ====="
        
        # standard_gnn用フォルダの作成 (run.py内で作成されますが念のため)
        mkdir -p "saved_models/seed${SEED}/${mode}/standard_gnn"

        echo ""
        echo "[Phase 2] Starting 5-Fold Cross Validation (Standard GNN)..."
        for i in "${FOLDS[@]}"; do
            echo ""
            echo "--- Running Fold $i (Test Session $i) ---"
            # ★ポイント: --standard_gnn フラグを追加しています
            python run.py \
                --standard_gnn \
                --test_session "$i" \
                --eval_metric "$EVAL_METRIC" \
                --epochs "$EPOCHS" \
                --seed "$SEED" \
                --modality "$modality" \
                "${mode_args[@]}"
        done

        # --- 3. 平均スコアの計算 ---
        # calculate_average.py が standard_gnn のパスに対応している必要がありますが
        # もし対応していない場合でも、ログファイルさえ正しく読めれば動作します
        echo ""
        echo "[Phase 3] Calculating Average Scores..."
        python calculate_average.py \
            --standard_gnn \
            --eval_metric "$EVAL_METRIC" \
            --seed "$SEED" \
            --modality "$modality" \
            "${mode_args[@]}"

        # --- 4. ログのプロット ---
        echo ""
        echo "[Phase 4] Plotting Training Logs..."
        python plot_log.py \
            --standard_gnn \
            --eval_metric "$EVAL_METRIC" \
            --seed "$SEED" \
            --modality "$modality" \
            "${mode_args[@]}"
    done
done

echo ""
echo "=========================================="
echo "          All Tasks Completed!"
echo "=========================================="