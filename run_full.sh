#!/usr/bin/env bash
# FullモードのIEMOCAP実験を、全foldで実行して集計するスクリプト。
# 実行例: bash run_full.sh
# 前提: DATA_DIRにラベル、順序、テキスト、BERT、wav2vecの各ファイルが作成済みであること。

set -euo pipefail

# =============================================================================
# 実験設定
# =============================================================================

# データと結果の保存先
DATA_DIR="output_data"          # 前処理済みデータのディレクトリ
LOG_ROOT="saved_models"         # 学習ログ・結果の保存先ルート
DATASET_NAME="IEMOCAP"          # データセット名

# Fullモード設定
MODE="full"                     # 固定値。MA/NMAに分けず、XXXを含む全発話を使う
MODALITY="multimodal"           # multimodal / text / audio
TEST_SESSIONS=(1 2 3 4 5)        # 5-fold。各Sessionを順番にテスト用として実行
DEV_RATIO=0.2                    # テスト対象以外の対話のうち、検証用にする割合

# 学習設定
SEED=1                         # 乱数seed
EPOCHS=100                       # 学習epoch数
BATCH_SIZE=16                    # batch size（対話数）
LEARNING_RATE=1e-4               # AdamWのlearning rate
DROPOUT=0.2                      # dropout率
MAX_GRAD_NORM=5.0                # gradient clippingの上限

# epoch選択指標
EVAL_METRIC="loss"                # loss / f1 / kl。klは検証KLが最小のepochを選ぶ
# KL、JS、BC、R2は評価指標として計算する。JS、BC、R2は損失関数やepoch選択には使わない。

# DAG-ERCモデル設定（既定値も明示し、実験条件をこのファイルで確認できるようにする）
HIDDEN_DIM=300                   # GNNのhidden dimension
MLP_LAYERS=2                     # 出力MLPの層数
GNN_LAYERS=4                     # GNN層数
ATTN_TYPE="rgcn"                # dotprod / linear / bilinear / rgcn
WINDOWP=1                        # 過去方向の同一話者ノード接続数
WINDOWF=0                        # 未来方向の接続数
TEXT_DIM=768                     # テキスト特徴量の次元
AUDIO_DIM=768                    # 音声特徴量の次元
FUSION_DIM_D=256                 # multimodal fusionのD次元
FUSION_DIM_O=512                 # multimodal fusionのO次元
SINGLE_MODAL_DIM=512             # 単一モダリティ使用時の中間次元
NODAL_ATT_TYPE=""               # 空欄 / global / past

# 実行オプション。必要な場合だけtrueにする。
NO_CUDA=false                    # trueならCPUで実行
NO_REL_ATTN=false                # trueならrelational attentionを無効化
TENSORBOARD=false                # trueならTensorBoard出力を有効化
RUN_ANALYSIS=true                # trueならエラー分析・対話別レポートも作成

# =============================================================================
# 実行
# =============================================================================

if [[ ! -d "$DATA_DIR" ]]; then
    echo "データディレクトリが見つかりません: $DATA_DIR" >&2
    exit 1
fi

RESULT_DIR="${LOG_ROOT}/seed${SEED}/${MODE}/${MODALITY}"

for fold in "${TEST_SESSIONS[@]}"; do
    echo "=========================================================="
    echo "Full experiment: modality=${MODALITY}, test_session=${fold}"
    echo "seed=${SEED}, epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, eval_metric=${EVAL_METRIC}"
    echo "=========================================================="

    CMD=(python run.py
        --data_dir "$DATA_DIR"
        --dataset_name "$DATASET_NAME"
        --full
        --modality "$MODALITY"
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

    if [[ -n "$NODAL_ATT_TYPE" ]]; then
        CMD+=(--nodal_att_type "$NODAL_ATT_TYPE")
    fi
    if [[ "$NO_CUDA" == true ]]; then
        CMD+=(--no_cuda)
    fi
    if [[ "$NO_REL_ATTN" == true ]]; then
        CMD+=(--no_rel_attn)
    fi
    if [[ "$TENSORBOARD" == true ]]; then
        CMD+=(--tensorboard)
    fi

    "${CMD[@]}"
done

# 5 fold分の平均と標準偏差を保存する。
python calculate_average.py \
    --log_dir "$LOG_ROOT" \
    --eval_metric "$EVAL_METRIC" \
    --seed "$SEED" \
    --modality "$MODALITY" \
    --full

# 学習・検証・テスト指標のepoch推移を図にする。
python plot_log.py \
    --log_dir "$LOG_ROOT" \
    --eval_metric "$EVAL_METRIC" \
    --seed "$SEED" \
    --modality "$MODALITY" \
    --full

# 発話別エラー分析と、セッション・対話順のdialogue_flow_report.txtを作る。
if [[ "$RUN_ANALYSIS" == true ]]; then
    python analyze_errors.py "$RESULT_DIR" --output_dir "$RESULT_DIR"
fi

echo "Full experiment completed. Results: $RESULT_DIR"
