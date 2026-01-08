#!/bin/bash
set -e # エラーが発生したら停止

# ==========================================================
# 実験設定 (Configuration)
# ==========================================================
SEED=5
EPOCHS=100           # run.pyのデフォルトに合わせています
BATCH_SIZE=256
LR=1e-4

# 交差検証のフォールド (1~5)
FOLDS=(1 2 3 4 5)

# --- 実行するアーキテクチャを選択 (true/false) ---
# ※ どれか1つを true にしてください
RUN_EDL_R2=true        # 提案手法: EDL(R2)
RUN_STANDARD_GNN=false # 比較用: Standard GNN
RUN_SIMPLE_NN=false    # 比較用: Simple NN
RUN_BASELINE=false     # 既存手法: DAG-ERC (元のまま)

# --- モード設定 ---
# "default": 学習=MA, テスト=MA
# "nma":     学習=MA, テスト=MA & NMA
MODES=("default" "nma") 

# --- モダリティ設定 ---
# EDL(R2) は multimodal のみ対応しています
MODALITIES=("multimodal")
# MODALITIES=("multimodal" "text" "audio") # ベースラインなら他も可

# データ前処理を実行するか (初回のみ true)
RUN_PREPROCESSING=false

echo "=========================================================="
echo "       IEMOCAP Emotion Recognition Experiment"
echo "       5-Fold Cross Validation"
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

# --- アーキテクチャに応じたフラグ設定 ---
EXTRA_ARGS=()
ARCH_NAME="DAG-ERC"
EVAL_METRIC="loss" # デフォルト

if [ "$RUN_EDL_R2" = true ]; then
    ARCH_NAME="EDL(R2)"
    EXTRA_ARGS+=(--edl_r2)
    EVAL_METRIC="loss" # EDLはNLLで評価
elif [ "$RUN_STANDARD_GNN" = true ]; then
    ARCH_NAME="Standard GNN"
    EXTRA_ARGS+=(--standard_gnn)
    EVAL_METRIC="loss"
elif [ "$RUN_SIMPLE_NN" = true ]; then
    ARCH_NAME="Simple NN"
    EXTRA_ARGS+=(--simple_nn)
    EVAL_METRIC="loss"
fi

echo "Target Architecture: $ARCH_NAME"
echo "Evaluation Metric  : $EVAL_METRIC"

# ==========================================================
# 実験ループ開始
# ==========================================================

for mode in "${MODES[@]}"; do
    # モードに応じた引数 (--nma) の設定
    mode_args=("${EXTRA_ARGS[@]}")
    [ "$mode" = "nma" ] && mode_args+=(--nma)

    for modality in "${MODALITIES[@]}"; do
        # EDLなどの制約チェック
        if [ "$RUN_EDL_R2" = true ] && [ "$modality" != "multimodal" ]; then
            echo "Skipping $modality for EDL(R2) (Supports multimodal only)"
            continue
        fi

        echo ""
        echo "##########################################################"
        echo "   Running: $ARCH_NAME | Mode: ${mode^^} | Modality: $modality"
        echo "##########################################################"

        # --- 1. 交差検証 (Fold 1-5) ---
        for i in "${FOLDS[@]}"; do
            echo ""
            echo "--- Fold $i ($ARCH_NAME / $modality) ---"
            
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

        # --- 2. 平均スコアの計算 ---
        echo ""
        echo "[Summary] Calculating Average Scores..."
        python calculate_average.py \
            --eval_metric "$EVAL_METRIC" \
            --seed "$SEED" \
            --modality "$modality" \
            "${mode_args[@]}"

        # --- 3. ログのプロット ---
        echo ""
        echo "[Plot] Plotting Training Logs..."
        python plot_log.py \
            --eval_metric "$EVAL_METRIC" \
            --seed "$SEED" \
            --modality "$modality" \
            "${mode_args[@]}"
    done
done