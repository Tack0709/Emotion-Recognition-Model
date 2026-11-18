#!/usr/bin/env bash
# experiment-settings用: FullモードのDAG-ERCを、選択式前処理つきで全fold実行する。
# 実行例: bash run_full.sh
# true/falseの項目と実験設定はこのファイルで変更する。

set -euo pipefail

# =============================================================================
# データ・保存先
# =============================================================================
DATA_DIR="output_data"
LOG_ROOT="saved_models"
DATASET_NAME="IEMOCAP"
IEMOCAP_ROOT="/home/datasets/mizuno/IEMOCAP_full_release/IEMOCAP_full_release" # 環境に合わせて変更

# =============================================================================
# Full実験設定
# =============================================================================
SEED=1
FOLDS=(1 2 3 4 5)
DEV_RATIO=0.2
MODALITIES=("multimodal") # DAG-ERCを実行するモダリティ: multimodal / text / audio
RAW_EMOTIONS=false         # trueなら元感情タグを個別クラスとして扱い all_category に保存
EVAL_METRIC="loss"         # loss / f1 / kl: Devデータで最良epochを選択する指標

# 学習設定
EPOCHS=100
BATCH_SIZE=16
LEARNING_RATE=1e-4
DROPOUT=0.2
MAX_GRAD_NORM=5.0

# DAG-ERCモデル設定
HIDDEN_DIM=300
MLP_LAYERS=2
GNN_LAYERS=4
ATTN_TYPE="rgcn"           # dotprod / linear / bilinear / rgcn
WINDOWP=1
WINDOWF=0
TEXT_DIM=768
AUDIO_DIM=768
FUSION_DIM_D=256
FUSION_DIM_O=512
SINGLE_MODAL_DIM=512

# 実行オプション
NO_CUDA=false
NO_REL_ATTN=false
TENSORBOARD=false
RUN_ANALYSIS=true            # 発話別エラー分析・対話フローレポートを作成

# =============================================================================
# 選択式前処理: 再利用する項目はfalse、作り直す項目だけtrueにする
# =============================================================================
RUN_LABEL_PREPROCESSING=false       # hard/soft labelとラベル集計
RUN_METADATA_PREPROCESSING=false   # text_dict/order/speaker_vocab
RUN_BERT_FEATURE_EXTRACTION=false  # BERTテキスト特徴量
RUN_WAV2VEC_FEATURE_EXTRACTION=false # wav2vec音声特徴量
BERT_MODEL="bert-base-uncased"
WAV2VEC_MODEL="facebook/wav2vec2-base-960h"

# =============================================================================
# 前処理
# =============================================================================
mkdir -p "$DATA_DIR"
if [[ "$RUN_LABEL_PREPROCESSING" == true || "$RUN_METADATA_PREPROCESSING" == true || "$RUN_WAV2VEC_FEATURE_EXTRACTION" == true ]]; then
    if [[ ! -d "$IEMOCAP_ROOT" ]]; then
        echo "IEMOCAP元データのディレクトリが見つかりません: $IEMOCAP_ROOT" >&2
        exit 1
    fi
fi

if [[ "$RUN_LABEL_PREPROCESSING" == true ]]; then
    echo "[Preprocessing] hard/soft labels"
    python create_label.py --iemo_root "$IEMOCAP_ROOT" --output_dir "$DATA_DIR" \
        | tee "$DATA_DIR/label_distribution_summary.txt"
fi
if [[ "$RUN_METADATA_PREPROCESSING" == true ]]; then
    echo "[Preprocessing] text, dialogue order, speaker vocabulary"
    python build_metadata.py --iemo_root "$IEMOCAP_ROOT" --output_dir "$DATA_DIR"
fi
if [[ "$RUN_BERT_FEATURE_EXTRACTION" == true ]]; then
    echo "[Preprocessing] BERT text features"
    python create_bert_features.py --output_dir "$DATA_DIR" --model_name "$BERT_MODEL"
fi
if [[ "$RUN_WAV2VEC_FEATURE_EXTRACTION" == true ]]; then
    echo "[Preprocessing] wav2vec audio features"
    python create_wav2vec_features.py --iemo_root "$IEMOCAP_ROOT" \
        --output_dir "$DATA_DIR" --model_name "$WAV2VEC_MODEL"
fi

# 学習に必要な前処理ファイルを事前確認する。
REQUIRED_FILES=(
    "IEMOCAP-hardlabel.npy"
    "IEMOCAP-softlabel-sum.npy"
    "order.pkl"
    "text_dict.pkl"
    "speaker_vocab.pkl"
    "bert-base-diag.npy"
    "w2v2-ft-diag.npy"
)
for filename in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$DATA_DIR/$filename" ]]; then
        echo "必要な前処理ファイルがありません: $DATA_DIR/$filename" >&2
        echo "run_full.sh冒頭のRUN_*_PREPROCESSING設定を確認してください。" >&2
        exit 1
    fi
done

# =============================================================================
# 学習・fold集計・分析
# =============================================================================
for modality in "${MODALITIES[@]}"; do
    RESULT_DIR="$LOG_ROOT/seed${SEED}/full/${modality}"
    if [[ "$RAW_EMOTIONS" == true ]]; then
        RESULT_DIR="$RESULT_DIR/all_category"
    fi

    for fold in "${FOLDS[@]}"; do
        echo "=========================================================="
        echo "Full experiment: modality=${modality}, test_session=${fold}"
        echo "seed=${SEED}, raw_emotions=${RAW_EMOTIONS}, epochs=${EPOCHS}, batch_size=${BATCH_SIZE}"
        echo "=========================================================="

        CMD=(python run.py
            --data_dir "$DATA_DIR"
            --dataset_name "$DATASET_NAME"
            --full
            --modality "$modality"
            --test_session "$fold"
            --dev_ratio "$DEV_RATIO"
            --seed "$SEED"
            --epochs "$EPOCHS"
            --batch_size "$BATCH_SIZE"
            --lr "$LEARNING_RATE"
            --dropout "$DROPOUT"
            --max_grad_norm "$MAX_GRAD_NORM"
            --eval_metric "$EVAL_METRIC"
            --hidden_dim "$HIDDEN_DIM"
            --mlp_layers "$MLP_LAYERS"
            --gnn_layers "$GNN_LAYERS"
            --attn_type "$ATTN_TYPE"
            --windowp "$WINDOWP"
            --windowf "$WINDOWF"
            --text_dim "$TEXT_DIM"
            --audio_dim "$AUDIO_DIM"
            --fusion_dim_D "$FUSION_DIM_D"
            --fusion_dim_O "$FUSION_DIM_O"
            --single_modal_dim "$SINGLE_MODAL_DIM")

        if [[ "$RAW_EMOTIONS" == true ]]; then CMD+=(--raw_emotions); fi
        if [[ "$NO_CUDA" == true ]]; then CMD+=(--no_cuda); fi
        if [[ "$NO_REL_ATTN" == true ]]; then CMD+=(--no_rel_attn); fi
        if [[ "$TENSORBOARD" == true ]]; then CMD+=(--tensorboard); fi
        "${CMD[@]}"
    done

    AVG_CMD=(python calculate_average.py
        --log_dir "$LOG_ROOT" --eval_metric "$EVAL_METRIC"
        --seed "$SEED" --modality "$modality" --full)
    PLOT_CMD=(python plot_log.py
        --log_dir "$LOG_ROOT" --eval_metric "$EVAL_METRIC"
        --seed "$SEED" --modality "$modality" --full)
    if [[ "$RAW_EMOTIONS" == true ]]; then
        AVG_CMD+=(--raw_emotions)
        PLOT_CMD+=(--raw_emotions)
    fi

    echo "[Summary] Calculating fold average: $RESULT_DIR"
    "${AVG_CMD[@]}"
    echo "[Summary] Plotting training logs: $RESULT_DIR"
    "${PLOT_CMD[@]}"

    if [[ "$RUN_ANALYSIS" == true ]]; then
        echo "[Analysis] Error and dialogue-flow reports: $RESULT_DIR"
        python analyze_errors.py "$RESULT_DIR" --output_dir "$RESULT_DIR"
    fi
done

echo "Full experiment completed. Results: $LOG_ROOT/seed${SEED}/full"
