import os
import re
import glob
import pickle
import numpy as np
import torch
import librosa # 音声ファイル読み込み・リサンプリング用
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import sys

def create_wav2vec_features(
    iemo_root='IEMOCAP_full_release', 
    output_dir='output_data', 
    model_name='facebook/wav2vec2-base-960h'
):
    """
    IEMOCAPの発話ごとのWAVファイル
    (例: .../sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav)
    を読み込み、Wav2Vec 2.0 の特徴量を抽出し、w2v2-ft-diag.npy として保存します。
    """
    
    # --- 1. デバイスとモデル設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # Wav2Vec 2.0 が要求するサンプリングレート
    SAMPLING_RATE = 16000 
    
    output_path = os.path.join(output_dir, 'w2v2-ft-diag.npy')

    # --- 2. モデルとプロセッサのロード ---
    print(f"モデルをロード中: {model_name}")
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        model.eval() # 評価モード
    except Exception as e:
        print(f"エラー: モデル '{model_name}' のロードに失敗しました。: {e}")
        sys.exit(1)

    # --- 3. 処理対象の発話IDリストの取得 ---
    # (修正点: タイムスタンプ 解析を削除し、order.pkl を使用)
    order_path = os.path.join(output_dir, 'order.pkl')
    if not os.path.exists(order_path):
        print(f"エラー: {order_path} が見つかりません。")
        print("先に create_metadata.py を実行してください。")
        sys.exit(1)
        
    print(f"{order_path} から発話IDリストを読み込み中...")
    with open(order_path, 'rb') as f:
        order_dic = pickle.load(f) # {セッションID: [発話IDリスト]}

    # --- 4. 音声ファイル のロードと特徴抽出 ---
    w2v2_features_dic = {}

    print("Wav2Vec 2.0 特徴量抽出を開始します...")
    
    with torch.no_grad():
        # order.pkl の全セッション、全発話をループ
        for session_id, utt_ids in tqdm(order_dic.items(), desc="Processing Sessions"):
            
            # (修正点: セッションWAV のキャッシュは不要)
            
            for utt_id in utt_ids:
                # 1. 発話ごとのWAVファイルパスを構築
                # 例: .../Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav
                session_num = f"Session{utt_id[4]}" # 'Ses01...' -> 'Session1'
                wav_path = os.path.join(
                    iemo_root, 
                    session_num, 
                    "sentences", 
                    "wav", 
                    session_id, # 例: Ses01F_impro01
                    f"{utt_id}.wav"  # 例: Ses01F_impro01_F000.wav
                )

                if not os.path.exists(wav_path):
                    print(f"Warning: 音声ファイルが見つかりません: {wav_path}")
                    continue
                
                # 2. 音声ファイル をロード＆リサンプリング
                try:
                    audio_array, _ = librosa.load(wav_path, sr=SAMPLING_RATE)
                except Exception as e:
                    print(f"エラー: 音声ファイル {wav_path} のロードに失敗: {e}")
                    continue
                
                if len(audio_array) == 0:
                    continue # 音声が空の場合はスキップ

                # 3. 特徴抽出
                inputs = processor(
                    audio_array, 
                    sampling_rate=SAMPLING_RATE, 
                    return_tensors="pt", 
                    padding=True
                ).to(device)
                
                outputs = model(**inputs)
                
                # 特徴量 (最終隠れ層の時間軸方向の平均プーリング)
                feature = outputs.last_hidden_state.mean(dim=1) # (B, Seq, Dim) -> (B, Dim)
                
                w2v2_features_dic[utt_id] = feature.cpu().numpy().squeeze()

    # --- 5. ファイル保存 ---
    try:
        print(f"特徴量辞書を保存中: {output_path}")
        np.save(output_path, w2v2_features_dic)
    except Exception as e:
        print(f"エラー: ファイルの保存中に問題が発生しました: {e}")
        sys.exit(1)

    print(f"処理完了。{len(w2v2_features_dic)} 個の特徴量を {output_path} に保存しました。")
    if len(w2v2_features_dic) != 10039:
         print(f"Warning: 期待される発話数(10039) と一致しません。 (検出数: {len(w2v2_features_dic)})")

# --- スクリプト実行 ---
if __name__ == "__main__":
    create_wav2vec_features(
        iemo_root='IEMOCAP_full_release', 
        output_dir='output_data',
        model_name='facebook/wav2vec2-base-960h' # 768次元
    )