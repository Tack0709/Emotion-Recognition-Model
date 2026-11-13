#!/usr/bin/env bash
# FullモードのIEMOCAP実験と選択式の前処理を、全foldで実行・集計するスクリプト。
# 実行例: bash run_full.sh
# 前処理フラグがfalseの項目は、既存のDATA_DIR内ファイルを再利用する。

set -euo pipefail

# =============================================================================
# 実験設定
# =============================================================================

# データと結果の保存先
DATA_DIR="output_data"          # 前処理済みデータのディレクトリ
LOG_ROOT="saved_models"         # 学習ログ・結果の保存先ルート
DATASET_NAME="IEMOCAP"          # データセット名
IEMOCAP_ROOT="/home/datasets/mizuno/IEMOCAP_full_release/IEMOCAP_full_release" # 元データのルート。環境に合わせて変更

# Fullモード設定
MODE="full"                     # 固定値。MA/NMAに分けず、XXXを含む全発話を使う
MODALITIES=("multimodal" "text" "audio") # 実行するモダリティ。不要な項目はコメントアウト
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

# 前処理の選択。既存ファイルを再利用する項目はfalse、新しく作る／作り直す項目はtrue。
RUN_LABEL_PREPROCESSING=True    # EmoEvaluationからhard/soft labelを作成し、件数・割合をsummaryに保存
RUN_METADATA_PREPROCESSING=True # 発話テキスト・対話順・話者vocabを作成
RUN_BERT_FEATURE_EXTRACTION=True # text_dict.pklからBERT特徴量を抽出（モデル取得にネット接続が必要）
RUN_WAV2VEC_FEATURE_EXTRACTION=True # 発話wavからwav2vec特徴量を抽出（非常に時間がかかる場合あり）
BERT_MODEL="bert-base-uncased"  # create_bert_features.pyで使用するモデル
WAV2VEC_MODEL="facebook/wav2vec2-base-960h" # 音声特徴抽出モデル

# =============================================================================
# 実行
# =============================================================================

mkdir -p "$DATA_DIR"

if [[ "$RUN_LABEL_PREPROCESSING" == true || "$RUN_METADATA_PREPROCESSING" == true || "$RUN_WAV2VEC_FEATURE_EXTRACTION" == true ]]; then
    if [[ ! -d "$IEMOCAP_ROOT" ]]; then
        echo "IEMOCAP元データのディレクトリが見つかりません: $IEMOCAP_ROOT" >&2
        exit 1
    fi
fi

# 選択された前処理だけを、特徴量作成に必要な順で実行する。
if [[ "$RUN_LABEL_PREPROCESSING" == true ]]; then
    echo "[Preprocessing] hard/soft labels"
    python create_label.py \
        --iemo_root "$IEMOCAP_ROOT" \
        --output_dir "$DATA_DIR" \
        | tee "$DATA_DIR/label_distribution_summary.txt"
fi

if [[ "$RUN_METADATA_PREPROCESSING" == true ]]; then
    echo "[Preprocessing] text, dialogue order, speaker vocabulary"
    python build_metadata.py \
        --iemo_root "$IEMOCAP_ROOT" \
        --output_dir "$DATA_DIR"
fi

if [[ "$RUN_BERT_FEATURE_EXTRACTION" == true ]]; then
    echo "[Preprocessing] BERT text features"
    python create_bert_features.py \
        --output_dir "$DATA_DIR" \
        --model_name "$BERT_MODEL"
fi

if [[ "$RUN_WAV2VEC_FEATURE_EXTRACTION" == true ]]; then
    echo "[Preprocessing] wav2vec audio features"
    python create_wav2vec_features.py \
        --iemo_root "$IEMOCAP_ROOT" \
        --output_dir "$DATA_DIR" \
        --model_name "$WAV2VEC_MODEL"
fi

# 学習前に必要ファイルを確認し、未生成なら不足項目を表示して終了する。
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

# 実行するアーキテクチャ。必要な比較だけ残し、不要な行はコメントアウトする。
# 既定ではベースラインと3種類のアブレーションをすべて実行する。
EXPERIMENTS=(
    "DAG-ERC|"
    "SimpleNN|--simple_nn"
    "StandardGNN|--standard_gnn"
    "EDL_R2|--edl_r2"
)

for modality in "${MODALITIES[@]}"; do
    for experiment in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r experiment_name architecture_flag <<< "$experiment"

        # text/audioではベースラインDAG-ERCのみ実行し、アブレーションはmultimodalに限定する。
        if [[ "$modality" != "multimodal" && -n "$architecture_flag" ]]; then
            echo "Skipping ${experiment_name} (${modality}): ablations run only for multimodal input."
            continue
        fi

        ARCH_ARGS=()
        if [[ -n "$architecture_flag" ]]; then
            ARCH_ARGS+=("$architecture_flag")
        fi

        # run.pyの保存先規則に合わせてモデル別の結果フォルダを選ぶ。
        case "$architecture_flag" in
            --edl_r2) RESULT_DIR="${LOG_ROOT}/seed${SEED}/${MODE}/edl_r2" ;;
            --simple_nn) RESULT_DIR="${LOG_ROOT}/seed${SEED}/${MODE}/simple_nn" ;;
            --standard_gnn) RESULT_DIR="${LOG_ROOT}/seed${SEED}/${MODE}/standard_gnn" ;;
            *) RESULT_DIR="${LOG_ROOT}/seed${SEED}/${MODE}/${modality}" ;;
        esac

        echo "=========================================================="
        echo "Full experiment: ${experiment_name} (${modality})"
        echo "seed=${SEED}, epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, eval_metric=${EVAL_METRIC}"
        echo "=========================================================="

        for fold in "${TEST_SESSIONS[@]}"; do
            echo "--- ${experiment_name} (${modality}): test_session=${fold} ---"

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

            "${CMD[@]}" "${ARCH_ARGS[@]}"
        done

        # モダリティ・アーキテクチャごとにfold平均と学習曲線を保存する。
        python calculate_average.py \
            --log_dir "$LOG_ROOT" \
            --eval_metric "$EVAL_METRIC" \
            --seed "$SEED" \
            --modality "$modality" \
            --full "${ARCH_ARGS[@]}"

        python plot_log.py \
            --log_dir "$LOG_ROOT" \
            --eval_metric "$EVAL_METRIC" \
            --seed "$SEED" \
            --modality "$modality" \
            --full "${ARCH_ARGS[@]}"

        if [[ "$RUN_ANALYSIS" == true ]]; then
            python analyze_errors.py "$RESULT_DIR" --output_dir "$RESULT_DIR"
        fi

        echo "Completed ${experiment_name} (${modality}). Results: $RESULT_DIR"
    done
done
