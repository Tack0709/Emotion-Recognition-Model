#!/bin/bash
# =============================================================================
# run_rerun_batch16.sh  —  設定を揃えた再実験（batch=16 版）
#
# 【背景】
#   これまでの実験は seed/batch/lr がバラバラで、条件間の比較が成立していない。
#     seed30  : batch=256, epochs=100, lr=8.84e-4  ← 卒論の 1.633 / 1.527 はここ
#     seed100 : batch=16,  epochs=100, lr=1e-4
#     seed21  : batch=256, epochs=650, lr=8.84e-4
#   さらに batch=256 は「対話256本」を意味し、学習データ約96対話に対して
#   1エポック=1ステップ（100エポック=100ステップ）という極端な設定だった。
#   （ACL2024 の batch=256 は発話単位。単位の取り違えと思われる）
#
# 【この再実験の設定】
#   batch=16 → 1エポック約6ステップ、100エポックで約600ステップ
#   lr=1e-4、線形warmup(10%)→線形減衰、AdamW(wd=1e-2)
#   ※ scheduler は現状維持（案A）。SLT2022 と同じ形。
#     ACL2024 は Adafactor + Noam(逆平方根減衰)だが、総ステップ600では
#     線形減衰の方が収束が安定するため変更しない。
#
# 【既に分かっていること（既存ログからの診断結果）】
#   - ma2nma の NMA NLL = 1.633 は「到達可能な性能」ではない。
#     学習曲線上、epoch 30-50 で 1.45 程度まで下がった後に悪化しており、
#     MA の検証データで選んだエポックが NMA にとって最悪の点になっている。
#   - H(y^)=1.0454 ≈ H(y)=1.0301。予測分布の広がりは正解と同水準であり、
#     「過信」でも「潰れ」でもない。卒論4.8.1節の考察は誤り。
#   - nma モード(1.527)は Train F1 が 72→24 に落ちており、
#     分類性能を犠牲にして予測を鈍らせた結果。
#     卒論4.8.4節の「文脈の連続性」という説明は成立しない。
#   → この再実験の主目的は、上記が batch=16 でも再現するかの確認。
#
# 【実行方法】
#   ./run_rerun_batch16.sh
#   条件を増やすときは下の EXPERIMENTS 配列のコメントアウトを外す。
#
# 【注意】
#   - seed=100 で保存されるため、既存の seed30（batch=256）とは別ディレクトリになる。
#     上書きの心配はない。
#   - 過去に seed100 で batch=16/epochs=100/lr=1e-4 の default 実験があった記録が
#     あるため、saved_models/seed100/default/ に既存ファイルがないか先に確認すること。
#   - text/audio で --simple_nn や --standard_gnn を使うと保存先が
#     text/ audio/ になり DAG-ERC の結果と衝突する（run.py:143-149）。
#     そのため下の配列では multimodal のみに付けている。
# =============================================================================
set -e

# ================= 設定 =================
SEED=100
EPOCHS=100
BATCH_SIZE=16
LR=1e-4
DROPOUT=0.2
EVAL_METRIC=loss
FOLDS=(1 2 3 4 5)

RUN_PREPROCESSING=true   # 特徴量が未生成なら true
RUN_ANALYSIS=true         # analyze_errors.py まで実行するか
# ========================================

COMMON="--seed ${SEED} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --dropout ${DROPOUT} \
        --eval_metric ${EVAL_METRIC}"

# =============================================================================
# 実験条件
#   形式: "表示名|モード|モダリティ|追加フラグ"
#     モード: default / nma / ma2nma / nma2ma
#   コメントアウトを外すと実行される。
# =============================================================================
EXPERIMENTS=(
    # ---------- 必須の3条件（卒論の主要な数値の再現） ----------
    "DAG-ERC_MA学習MAテスト|default|multimodal|"
    "DAG-ERC_MA学習NMAテスト|ma2nma|multimodal|"
    "DAG-ERC_MA+NMA学習|nma|multimodal|"

    # ---------- 文脈の効果を分離（simple_nn = GNNをバイパス） ----------
    "SimpleNN_MA学習NMAテスト|ma2nma|multimodal|--simple_nn"
    "SimpleNN_MA+NMA学習|nma|multimodal|--simple_nn"
    "SimpleNN_MA学習MAテスト|default|multimodal|--simple_nn"

    # ---------- モダリティ別（卒論 表4.3 / 表4.4） ----------
    # "DAG-ERC_text|default|text|"
    # "DAG-ERC_audio|default|audio|"
    # "DAG-ERC_text_NMA|ma2nma|text|"
    # "DAG-ERC_audio_NMA|ma2nma|audio|"

    # ---------- 文脈モデル（卒論 表4.5 / 表4.6） ----------
    # 注: main の standard_gnn は adj=tril（未来を見ない）。
    #     卒論4.3.2節の記述（未来を見る）と食い違う。要確認。
    # "StandardGNN|default|multimodal|--standard_gnn"
    # "StandardGNN_NMA|ma2nma|multimodal|--standard_gnn"

    # ---------- 損失関数（卒論 表4.7 / 表4.8） ----------
    # 注: --edl_r2 は multimodal かつ DAG-ERC のときのみ使用可
    # "EDL_R2|default|multimodal|--edl_r2"
    # "EDL_R2_NMA|ma2nma|multimodal|--edl_r2"

    # ---------- MA+NMA学習 → MAテスト ----------
    # "DAG-ERC_NMA学習MAテスト|nma2ma|multimodal|"
)

# =============================================================================

echo "=========================================================="
echo "  再実験: batch=${BATCH_SIZE} / epochs=${EPOCHS} / lr=${LR}"
echo "  seed=${SEED}  scheduler=linear(warmup 10%)  optimizer=AdamW"
echo "  実行条件数: $(printf '%s\n' "${EXPERIMENTS[@]}" | grep -vc '^\s*#' || true)"
echo "=========================================================="

# --- 前処理 ---
if [ "$RUN_PREPROCESSING" = true ]; then
    echo ""
    echo "[Phase 0] 前処理"
    python create_label.py
    python create_metadata.py
    python create_bert_features.py
    python create_wav2vec_features.py
fi

# --- 実験ループ ---
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name mode modality flag <<< "$exp"

    # モードに応じたフラグ
    mode_args=()
    case "$mode" in
        nma)     mode_args+=(--nma) ;;
        ma2nma)  mode_args+=(--train_ma_test_nma) ;;
        nma2ma)  mode_args+=(--train_nma_test_ma) ;;
        default) ;;
        *) echo "不明なモード: $mode"; exit 1 ;;
    esac

    # アーキテクチャフラグ
    arch_args=()
    [ -n "$flag" ] && arch_args+=($flag)

    echo ""
    echo "##########################################################"
    echo "  ${name}"
    echo "  mode=${mode}  modality=${modality}  flag=${flag:-なし}"
    echo "##########################################################"

    # --- 学習（5 fold） ---
    for f in "${FOLDS[@]}"; do
        echo "--- fold ${f} ---"
        python run.py ${COMMON} \
            --modality "$modality" \
            --test_session "$f" \
            "${mode_args[@]}" "${arch_args[@]}"
    done

    # --- 平均スコア ---
    echo "--- 平均スコア ---"
    python calculate_average.py \
        --seed "$SEED" \
        --eval_metric "$EVAL_METRIC" \
        --modality "$modality" \
        "${mode_args[@]}" "${arch_args[@]}"

    # --- 学習曲線 ---
    echo "--- 学習曲線 ---"
    python plot_log.py \
        --seed "$SEED" \
        --eval_metric "$EVAL_METRIC" \
        --modality "$modality" \
        "${mode_args[@]}" "${arch_args[@]}"

    # --- エラー分析 ---
    if [ "$RUN_ANALYSIS" = true ]; then
        # 保存先の特定（run.py:143-149 の分岐に対応）
        BASE="saved_models/seed${SEED}/${mode}"
        if [[ "$flag" == *"--edl_r2"* ]]; then
            TARGET="${BASE}/edl_r2"
        elif [[ "$modality" == "multimodal" && "$flag" == *"--simple_nn"* ]]; then
            TARGET="${BASE}/simple_nn"
        elif [[ "$modality" == "multimodal" && "$flag" == *"--standard_gnn"* ]]; then
            TARGET="${BASE}/standard_gnn"
        else
            TARGET="${BASE}/${modality}"
        fi

        if [ -d "$TARGET" ]; then
            echo "--- エラー分析: ${TARGET} ---"
            python analyze_errors.py "$TARGET" --output_dir "$TARGET"
        else
            echo "[警告] ディレクトリが見つかりません: ${TARGET}"
        fi
    fi
done

# =============================================================================
# 診断
# =============================================================================
# =============================================================================
# 診断：有効な実験条件ごとに、平均結果ファイルへTest NLL診断を追記
# =============================================================================
BASE="saved_models/seed${SEED}"
declare -A DIAG_DONE=()

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name mode modality flag <<< "$exp"

    # モード名は run.py の保存先ディレクトリに合わせる
    case "$mode" in
        default|nma|ma2nma|nma2ma) mode_dir="$mode" ;;
        *)
            echo "[警告] 診断できないモードです: ${mode}"
            continue
            ;;
    esac

    # アーキテクチャ名を決定
    if [[ "$flag" == *"--edl_r2"* ]]; then
        arch_name="edl_r2"
    elif [[ "$flag" == *"--simple_nn"* ]]; then
        arch_name="simple_nn"
    elif [[ "$flag" == *"--standard_gnn"* ]]; then
        arch_name="standard_gnn"
    else
        arch_name="dag_erc"
    fi

    # run.py / calculate_average.py と同じ保存先を組み立てる
    if [ "$arch_name" = "edl_r2" ]; then
        target_dir="${BASE}/${mode_dir}/edl_r2"
    elif [ "$modality" = "multimodal" ] && [ "$arch_name" != "dag_erc" ]; then
        target_dir="${BASE}/${mode_dir}/${arch_name}"
    else
        target_dir="${BASE}/${mode_dir}/${modality}"
    fi

    summary="${target_dir}/average_result_${EVAL_METRIC}_${arch_name}_${modality}.txt"

    # 同じ保存先を複数の実験名が参照する場合、診断を重複追記しない
    if [[ -n "${DIAG_DONE[$summary]:-}" ]]; then
        continue
    fi
    DIAG_DONE["$summary"]=1

    if [ ! -f "$summary" ]; then
        echo "[警告] 平均結果ファイルがありません: ${summary}"
        continue
    fi

    # run.py が付けるログ名のモード別suffix
    case "$mode" in
        nma)     log_suffix="_nma" ;;
        nma2ma)  log_suffix="_nma2ma" ;;
        *)       log_suffix="" ;;
    esac

    {
        echo ""
        echo "=========================================================="
        echo "Test NLL 診断"
        echo "実験: ${name}"
        echo "mode=${mode}, modality=${modality}, architecture=${arch_name}"
        echo "=========================================================="

        for f in "${FOLDS[@]}"; do
            log="${target_dir}/logging_${EVAL_METRIC}_fold${f}_seed${SEED}${log_suffix}.log"

            if [ ! -f "$log" ]; then
                echo "fold${f}: ログが見つかりません"
                continue
            fi

            min=$(grep -oE "Test \[NLL [0-9.]+" "$log" \
                  | awk '{print NR" "$3}' \
                  | sort -k2 -g \
                  | head -1)

            best=$(grep "Best Epoch:" "$log" \
                   | tail -1 \
                   | sed 's/.*Best Epoch: *//')

            final=$(grep "Test NLL at Best Val:" "$log" \
                    | tail -1 \
                    | sed 's/.*Test NLL at Best Val: *//')

            if [ -n "$min" ]; then
                min_value="${min#* }"
                min_epoch="${min%% *}"
                echo "fold${f}: 最小Test NLL=${min_value} (epoch ${min_epoch}) | 選択epoch=${best} | 選択時Test NLL=${final}"
            else
                echo "fold${f}: Test NLL をログから読み取れませんでした"
            fi
        done
    } | tee -a "$summary"
done