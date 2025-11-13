import os
import pickle
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import sys

def create_bert_features(
    input_dir='output_data', 
    output_dir='output_data', 
    model_name='bert-base-uncased'
):
    """
    text_dict.pkl から発話テキストを読み込み、
    BERTモデルで特徴量を抽出し、bert-base-diag.npy として保存します。
    """
    
    # --- 1. デバイス設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # --- 2. 入力ファイル読み込み ---
    text_dict_path = os.path.join(input_dir, 'text_dict.pkl')
    output_path = os.path.join(output_dir, 'bert-base-diag.npy')
    
    if not os.path.exists(text_dict_path):
        print(f"エラー: 入力ファイル {text_dict_path} が見つかりません。")
        print("先に create_metadata.py を実行してください。")
        sys.exit(1)

    print(f"読み込み中: {text_dict_path}")
    with open(text_dict_path, 'rb') as f:
        text_dict = pickle.load(f)

    # --- 3. モデルとトークナイザのロード ---
    print(f"モデルをロード中: {model_name}")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).to(device)
        model.eval() # 評価モードに設定
    except Exception as e:
        print(f"エラー: モデル '{model_name}' のロードに失敗しました。")
        print("インターネット接続を確認するか、モデル名が正しいか確認してください。")
        print(e)
        sys.exit(1)

    # --- 4. 特徴抽出 ---
    bert_features_dic = {}
    
    print("BERT特徴量抽出を開始します...")
    # tqdm を使って進捗を表示
    with torch.no_grad(): # 勾配計算を無効化してメモリ節約
        for utt_id, text in tqdm(text_dict.items(), desc="Processing Utterances"):
            # トークナイズ
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(device)
            
            # モデルで特徴抽出
            outputs = model(**inputs)
            
            # [CLS] トークンのベクトル (最終隠れ層) を取得
            # outputs.last_hidden_state の形状は (batch_size, sequence_length, hidden_size)
            # [CLS] トークンは最初 (インデックス 0) にある
            cls_vector = outputs.last_hidden_state[:, 0, :]
            
            # CPUに移動し、Numpy配列に変換
            bert_features_dic[utt_id] = cls_vector.cpu().numpy().squeeze()

    # --- 5. ファイル保存 ---
    try:
        print(f"特徴量辞書を保存中: {output_path}")
        np.save(output_path, bert_features_dic)
    except Exception as e:
        print(f"エラー: ファイルの保存中に問題が発生しました: {e}")
        sys.exit(1)

    print(f"処理完了。{len(bert_features_dic)} 個の特徴量を {output_path} に保存しました。")
    if len(bert_features_dic) != 10039:
         print(f"Warning: 期待される発話数(10039) と一致しません。 (検出数: {len(bert_features_dic)})")

# --- スクリプト実行 ---
if __name__ == "__main__":
    # output_data ディレクトリにある text_dict.pkl を読み込み、
    # output_data ディレクトリに bert-base-diag.npy を保存
    create_bert_features(
        input_dir='output_data', 
        output_dir='output_data',
        model_name='bert-base-uncased' # 'bert-base-uncased' (768次元) を使用
    )