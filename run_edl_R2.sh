#!/bin/bash
set -e # エラーが発生したら即停止

# ==========================================================
# 実験設定 (Global Configuration)
# ==========================================================
SEED=21

# ★変更: 論文の 20,000 steps に合わせるため Epoch数を増加
# (Batch=256, Trainデータ約8000件の場合: 8000/256 ≈ 31steps/epoch => 20000/31 ≈ 645 epochs)
EPOCHS=650

# ★変更: 論文設定 (Batch Size 256)
BATCH_SIZE=256

# ★変更: 論文設定 (Adafactor LR 8.84e-4)
LR=8.84e-4

# 交差検証のフォールド (1~5)
FOLDS=(1 2 3 4 5)

# モダリティ設定 (EDL(R2)はmultimodalのみ)
MODALITY="multimodal"

# モード設定: ("default" "nma") 両方やる場合は配列にする
# ここでは例として両方回す設定にしていますが、片方でよければ ("default") だけにしてください
MODES=("nma")

# ==========================================================
# 実行制御フラグ (ここを true/false で切り替え)
# ==========================================================
RUN_EDL_R2=true       # 提案手法: EDL(R2)
# Standard GNN / Simple NN はこのスクリプトではOFFにしておく
RUN_STANDARD_GNN=false 
RUN_SIMPLE_NN=false

# データ前処理を実行するか (初回のみtrue推奨)
RUN_PREPROCESSING=false

echo "=========================================================="
echo "       IEMOCAP Experiment Runner (EDL-R2 Paper Settings)"
echo "       Seed: $SEED | Epochs: $EPOCHS | Batch: $BATCH_SIZE"
echo "=========================================================="

# --- 0. データ前処理 ---
if [ "$RUN_PREPROCESSING" = true ]; then
    echo ""
    echo "[Phase 0] Data Preprocessing..."
    if [ -f "create_label.py" ]; then
        python create_label.py
        python create_metadata.py
        python create_bert_features.py
        python create_wav2vec_features.py
    else
        echo "Warning: Preprocessing scripts not found. Skipping."
    fi
fi

# ==========================================================
# 1. EDL(R2) Experiment
# ==========================================================
if [ "$RUN_EDL_R2" = true ]; then
    ARCH_NAME="EDL(R2)"
    # EDLは不確実性考慮のためNLL(loss)評価が一般的
    EVAL_METRIC="loss" 

    for mode in "${MODES[@]}"; do
        mode_args=()
        [ "$mode" = "nma" ] && mode_args+=(--nma)

        echo ""
        echo "##########################################################"
        echo "   Running Experiment: $ARCH_NAME (Mode: $mode)"
        echo "##########################################################"

        # 5-Fold Cross Validation
        for i in "${FOLDS[@]}"; do
            echo "--- Fold $i ($ARCH_NAME) ---"
            
            # ログ保存ディレクトリの作成 (run.pyでも行われるが念のため)
            mkdir -p "saved_models/seed${SEED}/${mode}/edl_r2"

            python run.py \
                --edl_r2 \
                --modality "$MODALITY" \
                --test_session "$i" \
                --seed "$SEED" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --lr "$LR" \
                --eval_metric "$EVAL_METRIC" \
                "${mode_args[@]}"
        done

        # 平均スコア計算
        echo "--- Calculating Average ($ARCH_NAME) ---"
        python calculate_average.py \
            --edl_r2 \
            --modality "$MODALITY" \
            --seed "$SEED" \
            --eval_metric "$EVAL_METRIC" \
            "${mode_args[@]}"

        # ログプロット
        echo "--- Plotting Logs ($ARCH_NAME) ---"
        python plot_log.py \
            --edl_r2 \
            --modality "$MODALITY" \
            --seed "$SEED" \
            --eval_metric "$EVAL_METRIC" \
            "${mode_args[@]}"
    done
fi

echo ""
echo "=========================================================="
echo "          All Experiments Completed Successfully!"
echo "=========================================================="