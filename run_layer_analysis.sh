#!/usr/bin/env bash
# Full IEMOCAP DAG-ERCの層間特徴分析を、既存Noneチェックポイントでfoldごとに実行する。
# DAG-ERC本体は再学習しない。Step 2のLinear Probeだけを新規学習する。
# 実行例: ./run_layer_analysis.sh

set -euo pipefail

# =============================================================================
# 実験・入出力設定
# =============================================================================

DATA_DIR="output_data"
CHECKPOINT_DIR="saved_models/seed1/full/multimodal/best_model"
ANALYSIS_ROOT="layer_analysis"
IEMOCAP_ROOT="/home/datasets/mizuno/IEMOCAP_full_release/IEMOCAP_full_release"
SEED=1
FOLDS=(1)
ANALYSIS_SPLITS=("train" "dev")
EVAL_SPLIT="dev"                    # probe / ablation / ambiguity の評価対象: dev / test
DEVICE="auto"                         # auto / cpu / cuda
OVERWRITE_FEATURES=false              # trueの場合、既存features/*.npzを再生成

# 既存データを使う場合はfalse。再作成する場合だけtrueにする。
RUN_LABEL_PREPROCESSING=false
RUN_METADATA_PREPROCESSING=false
RUN_BERT_FEATURE_EXTRACTION=false
RUN_WAV2VEC_FEATURE_EXTRACTION=false
BERT_MODEL="bert-base-uncased"
WAV2VEC_MODEL="facebook/wav2vec2-base-960h"

# Stepごとの実行選択。後続Stepは、前段階の出力が存在すること。
RUN_FEATURE_EXTRACTION=true            # Step 0: checkpointからtrain/dev特徴を抽出（Testは明示指定時のみ）
RUN_CKA=true                           # Step 1: dev上のH0-H4 Linear CKA
RUN_LAYER_PROBES=false                 # Step 2: Trainで学習、Devで評価
RUN_LAYER_ABLATION=false               # Step 3: Train平均で置換、Devで評価
RUN_AMBIGUITY_ANALYSIS=false           # Step 4: Train由来entropy境界をDevに適用
RUN_FOLD_SUMMARY=false                 # fold別結果をseed単位で集計

# ProbeはDAG-ERC本体を固定したまま、Probe層だけを学習する。
PROBE_EPOCHS=100
PROBE_BATCH_SIZE=256
PROBE_LEARNING_RATE=1e-3

# =============================================================================
# 任意の前処理
# =============================================================================

mkdir -p "$DATA_DIR"
if [[ "$RUN_LABEL_PREPROCESSING" == true || "$RUN_METADATA_PREPROCESSING" == true || "$RUN_WAV2VEC_FEATURE_EXTRACTION" == true ]]; then
    if [[ ! -d "$IEMOCAP_ROOT" ]]; then
        echo "IEMOCAP元データのディレクトリが見つかりません: $IEMOCAP_ROOT" >&2
        exit 1
    fi
fi

if [[ "$RUN_LABEL_PREPROCESSING" == true ]]; then
    python create_label.py --iemo_root "$IEMOCAP_ROOT" --output_dir "$DATA_DIR" \
        | tee "$DATA_DIR/label_distribution_summary.txt"
fi
if [[ "$RUN_METADATA_PREPROCESSING" == true ]]; then
    python build_metadata.py --iemo_root "$IEMOCAP_ROOT" --output_dir "$DATA_DIR"
fi
if [[ "$RUN_BERT_FEATURE_EXTRACTION" == true ]]; then
    python create_bert_features.py --output_dir "$DATA_DIR" --model_name "$BERT_MODEL"
fi
if [[ "$RUN_WAV2VEC_FEATURE_EXTRACTION" == true ]]; then
    python create_wav2vec_features.py --iemo_root "$IEMOCAP_ROOT" \
        --output_dir "$DATA_DIR" --model_name "$WAV2VEC_MODEL"
fi

# =============================================================================
# 入力データ確認
# =============================================================================

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
        echo "必要なファイルがありません: $DATA_DIR/$filename" >&2
        exit 1
    fi
done

if [[ "$RUN_CKA" == true || "$RUN_LAYER_PROBES" == true || "$RUN_LAYER_ABLATION" == true || "$RUN_AMBIGUITY_ANALYSIS" == true ]]; then
    if [[ "$RUN_FEATURE_EXTRACTION" != true ]]; then
        echo "特徴抽出をスキップする場合、各foldのfeatures/*.npzが既に必要です。"
    fi
fi

# =============================================================================
# Step 0〜4: foldごとの分析
# =============================================================================

for fold in "${FOLDS[@]}"; do
    CHECKPOINT="$CHECKPOINT_DIR/best_model_fold${fold}.pt"
    FOLD_DIR="$ANALYSIS_ROOT/seed${SEED}/fold${fold}"
    FEATURE_DIR="$FOLD_DIR/features"

    if [[ "$RUN_FEATURE_EXTRACTION" == true || "$RUN_LAYER_ABLATION" == true ]]; then
        if [[ ! -f "$CHECKPOINT" ]]; then
            echo "Noneモデルのcheckpointがありません: $CHECKPOINT" >&2
            echo "CHECKPOINT_DIRを既存のfull/multimodal/Noneモデルに合わせてください。" >&2
            exit 1
        fi
    fi
    if [[ "$RUN_FEATURE_EXTRACTION" != true && ( "$RUN_CKA" == true || "$RUN_LAYER_PROBES" == true || "$RUN_LAYER_ABLATION" == true || "$RUN_AMBIGUITY_ANALYSIS" == true ) ]]; then
        for split in train dev; do
            if [[ ! -f "$FEATURE_DIR/${split}_features.npz" ]]; then
                echo "抽出済み特徴ファイルがありません: $FEATURE_DIR/${split}_features.npz" >&2
                exit 1
            fi
        done
        if [[ "$EVAL_SPLIT" == test && ( "$RUN_LAYER_PROBES" == true || "$RUN_LAYER_ABLATION" == true || "$RUN_AMBIGUITY_ANALYSIS" == true ) && ! -f "$FEATURE_DIR/test_features.npz" ]]; then
            echo "Test評価を指定していますが特徴ファイルがありません: $FEATURE_DIR/test_features.npz" >&2
            exit 1
        fi
    fi
    if [[ "$RUN_AMBIGUITY_ANALYSIS" == true && "$RUN_LAYER_PROBES" != true && ! -f "$FOLD_DIR/probe/predictions/${EVAL_SPLIT}_H0.npz" ]]; then
        echo "曖昧度別Probe分析には既存のProbe予測が必要です: $FOLD_DIR/probe/predictions/" >&2
        exit 1
    fi
    if [[ "$RUN_AMBIGUITY_ANALYSIS" == true && "$RUN_LAYER_ABLATION" != true && ! -f "$FOLD_DIR/ablation/utterance_deltas.csv" ]]; then
        echo "曖昧度別Ablation分析には既存のablation結果が必要です: $FOLD_DIR/ablation/utterance_deltas.csv" >&2
        exit 1
    fi

    echo "=========================================================="
    echo "Layer analysis: seed=${SEED}, fold=${fold}"
    if [[ "$RUN_FEATURE_EXTRACTION" == true || "$RUN_LAYER_ABLATION" == true ]]; then
        echo "Checkpoint: $CHECKPOINT"
    fi
    echo "=========================================================="

    if [[ "$RUN_FEATURE_EXTRACTION" == true ]]; then
        OVERWRITE_ARGS=()
        SPLIT_ARGS=("${ANALYSIS_SPLITS[@]}")
        if [[ "$EVAL_SPLIT" == test && ( "$RUN_LAYER_PROBES" == true || "$RUN_LAYER_ABLATION" == true || "$RUN_AMBIGUITY_ANALYSIS" == true ) ]]; then
            SPLIT_ARGS+=(test)
        fi
        if [[ "$OVERWRITE_FEATURES" == true ]]; then
            OVERWRITE_ARGS+=(--overwrite)
        fi
        python extract_layer_features.py \
            --checkpoint "$CHECKPOINT" \
            --data_dir "$DATA_DIR" \
            --output_dir "$ANALYSIS_ROOT" \
            --expected_seed "$SEED" \
            --expected_fold "$fold" \
            --splits "${SPLIT_ARGS[@]}" \
            "${OVERWRITE_ARGS[@]}"
    fi

    if [[ "$RUN_CKA" == true ]]; then
        python analyze_layer_cka.py \
            --input "$FEATURE_DIR/dev_features.npz" \
            --output_dir "$FOLD_DIR/cka"
    fi

    if [[ "$RUN_LAYER_PROBES" == true ]]; then
        python train_layer_probes.py \
            --fold_dir "$FOLD_DIR" \
            --seed "$SEED" \
            --epochs "$PROBE_EPOCHS" \
            --batch_size "$PROBE_BATCH_SIZE" \
            --learning_rate "$PROBE_LEARNING_RATE" \
            --eval_split "$EVAL_SPLIT" \
            --device "$DEVICE"
    fi

    if [[ "$RUN_LAYER_ABLATION" == true ]]; then
        python analyze_layer_ablation.py \
            --checkpoint "$CHECKPOINT" \
            --fold_dir "$FOLD_DIR" \
            --data_dir "$DATA_DIR" \
            --eval_split "$EVAL_SPLIT" \
            --device "$DEVICE"
    fi

    if [[ "$RUN_AMBIGUITY_ANALYSIS" == true ]]; then
        python analyze_layer_ambiguity.py --fold_dir "$FOLD_DIR" --eval_split "$EVAL_SPLIT"
    fi
done

if [[ "$RUN_FOLD_SUMMARY" == true ]]; then
    SUMMARY_ARGS=()
    FOLD_LIST=" ${FOLDS[*]} "
    if [[ "$FOLD_LIST" == *" 1 "* && "$FOLD_LIST" == *" 2 "* && \
          "$FOLD_LIST" == *" 3 "* && "$FOLD_LIST" == *" 4 "* && \
          "$FOLD_LIST" == *" 5 "* ]]; then
        SUMMARY_ARGS+=(--require_all_folds)
    fi
    python summarize_layer_analysis.py --root "$ANALYSIS_ROOT" --seed "$SEED" \
        --eval_split "$EVAL_SPLIT" "${SUMMARY_ARGS[@]}"
fi

echo "Layer analysis completed: $ANALYSIS_ROOT/seed${SEED}"
