#!/bin/bash
set -e # エラーが発生したら即停止

# ==========================================================
# 実験設定 (Global Configuration)
# ==========================================================
SEED=100
EPOCHS=100           # run.pyのデフォルトに合わせています (必要なら100に変更)
BATCH_SIZE=16
LR=1e-4

# 交差検証のフォールド (1~5)
FOLDS=(1 2 3 4 5)

# モダリティ設定 (基本は multimodal)
MODALITY="multimodal"

# モード設定: ("default" "nma") 両方やる場合は配列にする
# ここでは例として両方回す設定にしていますが、片方でよければ ("default") だけにしてください
MODES=("default" "nma")

# ==========================================================
# 実行制御フラグ (ここを true/false で切り替え)
# ==========================================================
RUN_EDL_R2=true       # EDL(R2)を実行するか
RUN_STANDARD_GNN=false # Standard GNN (Ablation) を実行するか
RUN_SIMPLE_NN=false    # Simple NN (Ablation) を実行するか

# データ前処理を実行するか (初回のみtrue推奨)
RUN_PREPROCESSING=true

echo "=========================================================="
echo "       IEMOCAP Experiment Runner"
echo "       Seed: $SEED | Epochs: $EPOCHS | Modality: $MODALITY"
echo "=========================================================="

# --- 0. データ前処理 ---
if [ "$RUN_PREPROCESSING" = true ]; then
    echo ""
    echo "[Phase 0] Data Preprocessing..."
    python create_label.py
    python create_metadata.py
    python create_bert_features.py
    python create_wav2vec_features.py
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

# ==========================================================
# 2. Standard GNN (Ablation Study)
# ==========================================================
if [ "$RUN_STANDARD_GNN" = true ]; then
    ARCH_NAME="Standard GNN"
    # 比較のため loss か f1 かを選択 (ここではlossに統一)
    EVAL_METRIC="loss"

    for mode in "${MODES[@]}"; do
        mode_args=()
        [ "$mode" = "nma" ] && mode_args+=(--nma)

        echo ""
        echo "##########################################################"
        echo "   Running Experiment: $ARCH_NAME (Mode: $mode)"
        echo "##########################################################"

        for i in "${FOLDS[@]}"; do
            echo "--- Fold $i ($ARCH_NAME) ---"
            python run.py \
                --standard_gnn \
                --modality "$MODALITY" \
                --test_session "$i" \
                --seed "$SEED" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --lr "$LR" \
                --eval_metric "$EVAL_METRIC" \
                "${mode_args[@]}"
        done

        echo "--- Calculating Average ($ARCH_NAME) ---"
        python calculate_average.py \
            --standard_gnn \
            --modality "$MODALITY" \
            --seed "$SEED" \
            --eval_metric "$EVAL_METRIC" \
            "${mode_args[@]}"

        echo "--- Plotting Logs ($ARCH_NAME) ---"
        python plot_log.py \
            --standard_gnn \
            --modality "$MODALITY" \
            --seed "$SEED" \
            --eval_metric "$EVAL_METRIC" \
            "${mode_args[@]}"
    done
fi

# ==========================================================
# 3. Simple NN (Ablation Study)
# ==========================================================
if [ "$RUN_SIMPLE_NN" = true ]; then
    ARCH_NAME="Simple NN"
    EVAL_METRIC="loss"

    for mode in "${MODES[@]}"; do
        mode_args=()
        [ "$mode" = "nma" ] && mode_args+=(--nma)

        echo ""
        echo "##########################################################"
        echo "   Running Experiment: $ARCH_NAME (Mode: $mode)"
        echo "##########################################################"

        for i in "${FOLDS[@]}"; do
            echo "--- Fold $i ($ARCH_NAME) ---"
            python run.py \
                --simple_nn \
                --modality "$MODALITY" \
                --test_session "$i" \
                --seed "$SEED" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --lr "$LR" \
                --eval_metric "$EVAL_METRIC" \
                "${mode_args[@]}"
        done

        echo "--- Calculating Average ($ARCH_NAME) ---"
        python calculate_average.py \
            --simple_nn \
            --modality "$MODALITY" \
            --seed "$SEED" \
            --eval_metric "$EVAL_METRIC" \
            "${mode_args[@]}"

        echo "--- Plotting Logs ($ARCH_NAME) ---"
        python plot_log.py \
            --simple_nn \
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