#!/bin/bash
set -e

echo "========================================================"
echo "    Global MA -> NMA (Pattern 5) Experiment Script"
echo "    - Text/Audio: Base Model Only"
echo "    - Multimodal: Base + Standard GNN + EDL R2"
echo "========================================================"

# --- 設定項目 ---
SEED=30
EPOCHS=50            # 必要に応じて変更 (例: 100)
BATCH_SIZE=32        # メモリに応じて調整
LR=1e-4
EVAL_METRIC="loss"

# 実験するモダリティ
MODALITIES=("text" "audio" "multimodal")

# 実験するモデル設定 ("表示名:フラグ")
EXPERIMENTS=(
    "Base::"                        # フラグなし (DAG-ERC)
    "StandardGNN:--standard_gnn"    # Standard GNN
    "EDL_R2:--edl_r2"               # EDL (R2)
)

# 交差検証のフォールド (テスト時は (5) のみでOK)
FOLDS=(1 2 3 4 5)

# 保存先ベースディレクトリ
BASE_SAVE_DIR="saved_models/seed${SEED}/global_ma2nma"

# 前処理 (必要な場合のみ true)
RUN_PREPROCESSING=false
if [ "$RUN_PREPROCESSING" = true ]; then
    echo "[Phase 0] Preprocessing..."
    python create_label.py; python create_metadata.py
    python create_bert_features.py; python create_wav2vec_features.py
fi

# --- 実験ループ ---
for modality in "${MODALITIES[@]}"; do
    echo ""
    echo "#################################################"
    echo "##### Target Modality: ${modality^^} #####"
    echo "#################################################"

    for exp in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r exp_name exp_flag <<< "$exp"

        # 【制御ロジック】
        # Multimodal 以外の場合、StandardGNN と EDL はスキップする
        if [ "$modality" != "multimodal" ]; then
            if [[ "$exp_flag" == *"--standard_gnn"* ]] || [[ "$exp_flag" == *"--edl_r2"* ]]; then
                # echo "   Skipping ${exp_name} for ${modality} (Multimodal only)"
                continue
            fi
        fi

        echo ""
        echo ">> Experiment: ${exp_name} (${modality})"
        echo "-------------------------------------------------"

        # コマンドライン引数の構築
        cmd_args=(--global_ma_train_nma_test)
        if [ -n "$exp_flag" ]; then
            cmd_args+=($exp_flag)
        fi

        # [Phase 1] Training & Testing
        for fold in "${FOLDS[@]}"; do
            echo "   Running Fold ${fold}..."
            # 実行 (ログを抑制したい場合は末尾に > /dev/null 2>&1 を追加)
            python run.py \
                --test_session "$fold" \
                --eval_metric "$EVAL_METRIC" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --lr "$LR" \
                --seed "$SEED" \
                --modality "$modality" \
                "${cmd_args[@]}"
        done

        # [Phase 2] Average & Plot
        echo "   Calculating Average & Plotting..."
        python calculate_average.py --eval_metric "$EVAL_METRIC" --seed "$SEED" --modality "$modality" --global_ma_train_nma_test $([ -n "$exp_flag" ] && echo "$exp_flag")
        python plot_log.py --eval_metric "$EVAL_METRIC" --seed "$SEED" --modality "$modality" --global_ma_train_nma_test $([ -n "$exp_flag" ] && echo "$exp_flag")

        # [Phase 3] Error Analysis
        # run.py (修正前オリジナル) の保存ロジックに合わせてパスを指定
        if [[ "$exp_flag" == *"--edl_r2"* ]]; then
            TARGET_DIR="${BASE_SAVE_DIR}/edl_r2"
        elif [[ "$exp_flag" == *"--standard_gnn"* ]]; then
            TARGET_DIR="${BASE_SAVE_DIR}/standard_gnn"
        else
            # Baseモデルの場合、run.pyはモダリティ名をフォルダ名にする
            # text -> .../text
            # audio -> .../audio
            # multimodal -> .../multimodal (DAG-ERCの場合)
            TARGET_DIR="${BASE_SAVE_DIR}/${modality}"
        fi

        if [ -d "$TARGET_DIR" ]; then
            echo "   Analyzing Errors in: ${TARGET_DIR}"
            python analyze_errors.py "$TARGET_DIR" --output_dir "$TARGET_DIR"
        else
            echo "   [Warning] Directory not found (Analysis skipped): ${TARGET_DIR}"
        fi

    done
done

echo ""
echo "=== All Experiments Completed ==="