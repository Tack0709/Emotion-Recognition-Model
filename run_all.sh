#!/bin/bash
set -e # エラーが発生したら停止

echo "=========================================="
echo "       IEMOCAP Emotion Recognition"
echo "      5-Fold Cross Validation Script"
echo "=========================================="

# --- 1. データ前処理 ---
echo ""
echo "[Phase 1] Data Preprocessing..."
python create_label.py
python create_metadata.py
python create_bert_features.py
python create_wav2vec_features.py

# --- 2. 交差検証 (Fold 1-5) ---
echo ""
echo "[Phase 2] Starting 5-Fold Cross Validation..."

for i in {1..5}
do
    echo ""
    echo "--- Running Fold $i (Test Session $i) ---"
    python run.py --test_session $i --eval_metric loss --epochs 100 --lr 5e-5 --seed 42
done

# --- 3. 結果の集計 ---
echo ""
echo "[Phase 3] Calculating Average Scores..."
python calculate_average.py --eval_metric loss

echo ""
echo "=========================================="
echo "           All Tasks Completed!"
echo "=========================================="