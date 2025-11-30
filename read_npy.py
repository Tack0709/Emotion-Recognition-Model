import numpy as np
import os
import sys

def read_npy_to_text(npy_path, output_dir='npy_text_results'):
    """
    .npyファイルを読み込み、その中身（辞書）をテキストファイルに書き出します。
    結果は指定されたディレクトリ（デフォルト: npy_text_results）に保存されます。
    """
    
    if not os.path.exists(npy_path):
        print(f"エラー: 入力ファイルが見つかりません: {npy_path}")
        return

    # --- 修正: 出力ディレクトリの作成 ---
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"ディレクトリを作成しました: {output_dir}")
        except OSError as e:
            print(f"エラー: ディレクトリの作成に失敗しました: {e}")
            return

    # --- 修正: 出力ファイルパスの生成 ---
    # 入力: processed_data/data.npy -> 出力: npy_text_results/data.txt
    base_name = os.path.basename(npy_path)          # data.npy
    file_name = os.path.splitext(base_name)[0] + '.txt' # data.txt
    output_txt_path = os.path.join(output_dir, file_name) # npy_text_results/data.txt

    print(f"読み込み中: {npy_path}")
    
    try:
        # .item() を使って Python の辞書オブジェクトとして取り出す
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"エラー: 読み込みに失敗しました。辞書形式の.npyではない可能性があります。\n{e}")
        return

    print(f"書き出し中: {output_txt_path}")
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Source File: {os.path.abspath(npy_path)}\n")
        f.write(f"Total Items: {len(data)}\n")
        f.write("="*50 + "\n\n")

        # 辞書のキー（発話ID）でソートして出力
        for key in sorted(data.keys()):
            value = data[key]
            
            f.write(f"ID: {key}\n")
            
            # 値の型に応じて出力を変える
            if isinstance(value, np.ndarray):
                # Numpy配列の場合（特徴量やソフトラベル）
                if value.size > 10:
                    # 要素数が多い場合は省略表示 (Shapeと先頭・末尾のみ)
                    f.write(f"  Type:  Numpy Array\n")
                    f.write(f"  Shape: {value.shape}\n")
                    f.write(f"  Data:  {value[:5]} ... {value[-5:]}\n") 
                else:
                    # 要素数が少ない場合（ソフトラベルなど）は全表示
                    f.write(f"  Value: {value}\n")
            else:
                # 数値や文字列の場合（ハードラベルなど）
                f.write(f"  Value: {value}\n")
            
            f.write("-" * 30 + "\n")

    print(f"完了しました。確認はこちら: {output_txt_path}")

if __name__ == "__main__":
    # コマンドライン引数でファイルを指定する場合
    # 使用例: python read_npy.py processed_data/IEMOCAP-hardlabel.npy
    
    # 保存先ディレクトリ名
    TARGET_DIR = 'npy_text_results'

    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        read_npy_to_text(target_file, output_dir=TARGET_DIR)
    else:
        # 引数がない場合のデフォルト動作
        target_file = 'processed_data/IEMOCAP-hardlabel.npy' 
        print(f"引数が指定されていないため、デフォルトファイル ({target_file}) を読み込みます。")
        read_npy_to_text(target_file, output_dir=TARGET_DIR)