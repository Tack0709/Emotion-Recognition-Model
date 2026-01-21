#!/bin/bash
set -e # エラーが発生したら停止

echo "=========================================="
echo "       IEMOCAP Emotion Recognition"
echo "      5-Fold Cross Validation Script"
echo " (Text/Audio/MM + Std GNN + EDL R2 + Analysis)"
echo "=========================================="

# --- 1. データ前処理 ---
# 初回のみコメントアウトを解除して実行してください
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
EVAL_METRIC="loss"

# バッチサイズと学習率
BATCH_SIZE=256
LR=8.84e-4

FOLDS=(1 2 3 4 5)
# ============================================

# モード設定: ("default" "nma" "ma2nma")
#MODES=("default" "nma")
MODES=("nma2ma") 

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

    echo ""
    echo "##########################################################"
    echo "   Starting Experiments for Mode: ${mode^^}"
    echo "##########################################################"

    # ==========================================================
    # 実験ループ: 5種類の実験設定
    # 形式: "表示名:モダリティ:フラグ"
    # ==========================================================
    EXPERIMENTS=(
        # "DAG-ERC:text:"                       # 1. Text Only
        "DAG-ERC:audio:"                      # 2. Audio Only
        # "DAG-ERC:multimodal:"                 # 3. Multimodal (Proposed)
        "Standard_GNN:multimodal:--standard_gnn" # 4. Standard GNN (Ablation)
        "EDL_R2:multimodal:--edl_r2"          # 5. EDL R2 (Uncertainty Learning)
    )

    for exp_config in "${EXPERIMENTS[@]}"; do
        # 設定を分解
        IFS=':' read -r arch_name modality flag <<< "$exp_config"
        
        # フラグ配列の作成
        current_args=("${mode_args[@]}")
        if [ -n "$flag" ]; then
            current_args+=($flag)
        fi

        echo ""
        echo "===== [${arch_name}] Modality: ${modality} ====="

        # --------------------------------------------------------
        # [Phase 2] 学習 & テスト (5-Fold)
        # --------------------------------------------------------
        echo "[Phase 2] Running 5-Fold CV..."
        for i in "${FOLDS[@]}"; do
            echo "--- Fold $i (Test Session $i) ---"
            
            python run.py \
                --test_session "$i" \
                --eval_metric "$EVAL_METRIC" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --lr "$LR" \
                --seed "$SEED" \
                --modality "$modality" \
                "${current_args[@]}"
        done

        # --------------------------------------------------------
        # [Phase 3] 平均スコア計算
        # --------------------------------------------------------
        echo ""
        echo "[Phase 3] Calculating Average Scores..."
        python calculate_average.py \
            --eval_metric "$EVAL_METRIC" \
            --seed "$SEED" \
            --modality "$modality" \
            "${current_args[@]}"

        # --------------------------------------------------------
        # [Phase 4] ログのプロット
        # --------------------------------------------------------
        echo ""
        echo "[Phase 4] Plotting Logs..."
        python plot_log.py \
            --eval_metric "$EVAL_METRIC" \
            --seed "$SEED" \
            --modality "$modality" \
            "${current_args[@]}"

        # --------------------------------------------------------
        # [Phase 5] エラー分析 (NEW)
        # --------------------------------------------------------
        echo ""
        echo "[Phase 5] Analyzing Errors..."

        # ターゲットディレクトリの自動判定 (run.pyの保存ロジックに対応)
        BASE_SAVE_DIR="saved_models/seed${SEED}/${mode}"
        
        if [[ "$flag" == *"--edl_r2"* ]]; then
             TARGET_DIR="${BASE_SAVE_DIR}/edl_r2"
        elif [[ "$flag" == *"--standard_gnn"* ]]; then
             TARGET_DIR="${BASE_SAVE_DIR}/standard_gnn"
        elif [[ "$flag" == *"--simple_nn"* ]]; then
             TARGET_DIR="${BASE_SAVE_DIR}/simple_nn"
        else
             # DAG-ERCの場合
             TARGET_DIR="${BASE_SAVE_DIR}/${modality}"
        fi

        # 分析スクリプトの実行
        # (ディレクトリを指定すると、その中の test_results_fold*.npy を全て読み込んで解析します)
        if [ -d "$TARGET_DIR" ]; then
            python analyze_errors.py "$TARGET_DIR" --output_dir "$TARGET_DIR"
            echo "Analysis saved to: $TARGET_DIR"
        else
            echo "Warning: Directory not found, skipping analysis: $TARGET_DIR"
        fi

    done
done

echo ""
echo "=========================================="
echo "          All Experiments Completed!"
echo "=========================================="