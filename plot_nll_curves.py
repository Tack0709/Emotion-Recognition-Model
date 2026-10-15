"""
NLL 推移の集計・可視化

既存のログ（logging_*.log）から毎エポックの Train/Val/Test NLL を読み取り、
5 fold を重ねてプロットする。追加学習は不要。

目的:
  「なぜ epoch 25 前後で Test NLL が底を打つのか」を観察する。
  - Train/Val/Test の3本がいつ乖離し始めるか
  - Val の最小点と Test の最小点がどれだけずれるか
  - mode（ma2nma / nma）でその挙動がどう変わるか

使い方:
  # 1条件
  python plot_nll_curves.py --dir saved_models/seed100/ma2nma/multimodal

  # 複数条件を並べる
  python plot_nll_curves.py --compare \
      saved_models/seed100/ma2nma/multimodal:ma2nma_DAG \
      saved_models/seed100/nma/multimodal:nma_DAG \
      saved_models/seed100/ma2nma/simple_nn:ma2nma_Simple \
      saved_models/seed100/nma/simple_nn:nma_Simple
"""

import os
import re
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNIFORM_NLL = float(np.log(5))  # 1.6094

# "Ep 12: Train [NLL 0.8234 F1 65.43 Acc 66.12] | Val [NLL ...] | Test [NLL ...]"
LINE_RE = re.compile(
    r"Ep\s+(\d+):\s+"
    r"Train\s+\[NLL\s+([\d.]+)\s+F1\s+([\d.]+)\s+Acc\s+([\d.]+)\]\s*\|\s*"
    r"Val\s+\[NLL\s+([\d.]+)\s+F1\s+([\d.]+)\s+Acc\s+([\d.]+)\]\s*\|\s*"
    r"Test\s+\[NLL\s+([\d.]+)\s+F1\s+([\d.]+)\s+Acc\s+([\d.]+)\]"
)


def parse_log(path):
    """ログ1本から毎エポックの値を読む。"""
    rec = {k: [] for k in
           ["epoch", "train_nll", "train_f1", "train_acc",
            "val_nll", "val_f1", "val_acc",
            "test_nll", "test_f1", "test_acc"]}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            g = m.groups()
            rec["epoch"].append(int(g[0]))
            rec["train_nll"].append(float(g[1]))
            rec["train_f1"].append(float(g[2]))
            rec["train_acc"].append(float(g[3]))
            rec["val_nll"].append(float(g[4]))
            rec["val_f1"].append(float(g[5]))
            rec["val_acc"].append(float(g[6]))
            rec["test_nll"].append(float(g[7]))
            rec["test_f1"].append(float(g[8]))
            rec["test_acc"].append(float(g[9]))
    return {k: np.array(v) for k, v in rec.items()} if rec["epoch"] else None


def load_dir(d):
    """1条件のログをすべて読む。"""
    files = sorted(glob.glob(os.path.join(d, "logging_*.log")))
    folds = []
    for f in files:
        r = parse_log(f)
        if r is not None:
            r["name"] = os.path.basename(f)
            folds.append(r)
    return folds


def summarise(folds):
    """fold ごとの要点と、fold 横断の統計を返す。"""
    rows = []
    for i, r in enumerate(folds, 1):
        v_min_i = int(np.argmin(r["val_nll"]))
        t_min_i = int(np.argmin(r["test_nll"]))
        rows.append({
            "fold": i,
            "val_min_ep": r["epoch"][v_min_i],
            "val_min": r["val_nll"][v_min_i],
            "test_at_val_min": r["test_nll"][v_min_i],
            "test_min_ep": r["epoch"][t_min_i],
            "test_min": r["test_nll"][t_min_i],
            "gap": r["test_nll"][v_min_i] - r["test_nll"][t_min_i],
            "ep_gap": int(r["epoch"][v_min_i] - r["epoch"][t_min_i]),
            "train_final": r["train_nll"][-1],
            "test_final": r["test_nll"][-1],
        })
    return rows


def print_summary(name, folds):
    rows = summarise(folds)
    print(f"\n{'=' * 78}")
    print(f"  {name}   （{len(folds)} folds）")
    print(f"{'=' * 78}")
    print(f"  {'fold':>4} {'Val最小ep':>10} {'Test最小ep':>11} {'epoch差':>8} "
          f"{'Test@Val最小':>13} {'Test最小':>10} {'損失':>8}")
    print("  " + "-" * 72)
    for r in rows:
        print(f"  {r['fold']:>4} {r['val_min_ep']:>10} {r['test_min_ep']:>11} "
              f"{r['ep_gap']:>+8} {r['test_at_val_min']:>13.4f} "
              f"{r['test_min']:>10.4f} {r['gap']:>+8.4f}")

    f = lambda k: np.array([r[k] for r in rows], dtype=float)
    print("  " + "-" * 72)
    print(f"  {'平均':>4} {f('val_min_ep').mean():>10.1f} "
          f"{f('test_min_ep').mean():>11.1f} {f('ep_gap').mean():>+8.1f} "
          f"{f('test_at_val_min').mean():>13.4f} {f('test_min').mean():>10.4f} "
          f"{f('gap').mean():>+8.4f}")
    print()
    print(f"  Train NLL 終端（平均）: {f('train_final').mean():.4f}")
    print(f"  Test  NLL 終端（平均）: {f('test_final').mean():.4f}")
    print(f"  一様分布の NLL        : {UNIFORM_NLL:.4f}")
    return rows


def plot_condition(ax, folds, title, show_legend=False):
    """1条件の Train/Val/Test NLL を fold 重ねでプロット。"""
    colors = {"train_nll": "tab:blue", "val_nll": "tab:orange",
              "test_nll": "tab:green"}
    labels = {"train_nll": "Train", "val_nll": "Val", "test_nll": "Test"}

    # 各 fold を薄く
    for r in folds:
        for k, c in colors.items():
            ax.plot(r["epoch"], r[k], color=c, alpha=0.18, lw=0.8)

    # fold 平均を濃く（長さが揃っている前提。違えば最短に合わせる）
    n = min(len(r["epoch"]) for r in folds)
    ep = folds[0]["epoch"][:n]
    for k, c in colors.items():
        mean = np.mean([r[k][:n] for r in folds], axis=0)
        ax.plot(ep, mean, color=c, lw=2.0, label=labels[k])

        if k == "test_nll":
            i = int(np.argmin(mean))
            ax.axvline(ep[i], color=c, ls=":", lw=1.2)
            ax.plot(ep[i], mean[i], "o", color=c, ms=7, zorder=5)
            ax.annotate(f"Test min\nep{ep[i]} / {mean[i]:.3f}",
                        xy=(ep[i], mean[i]), xytext=(8, -28),
                        textcoords="offset points", fontsize=8, color=c)
        if k == "val_nll":
            i = int(np.argmin(mean))
            ax.axvline(ep[i], color=c, ls=":", lw=1.2)
            ax.plot(ep[i], mean[i], "s", color=c, ms=7, zorder=5)
            ax.annotate(f"Val min\nep{ep[i]} (selected)",
                        xy=(ep[i], mean[i]), xytext=(8, 10),
                        textcoords="offset points", fontsize=8, color=c)

    ax.axhline(UNIFORM_NLL, color="gray", ls="--", lw=1.0)
    ax.text(ep[-1], UNIFORM_NLL, " uniform 1.609", fontsize=7,
            color="gray", va="bottom", ha="right")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(fontsize=8, loc="upper right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="1条件のディレクトリ")
    ap.add_argument("--compare", nargs="+",
                    help="複数条件。'パス:表示名' の形式")
    ap.add_argument("--out", default="nll_curves.png", help="出力画像のパス")
    args = ap.parse_args()

    if not args.dir and not args.compare:
        ap.error("--dir か --compare を指定してください")

    specs = []
    if args.dir:
        specs.append((args.dir, os.path.basename(args.dir.rstrip("/"))))
    else:
        for s in args.compare:
            if ":" in s:
                p, n = s.rsplit(":", 1)
            else:
                p, n = s, os.path.basename(s.rstrip("/"))
            specs.append((p, n))

    loaded = []
    for path, name in specs:
        folds = load_dir(path)
        if not folds:
            print(f"[警告] ログが見つかりません: {path}")
            continue
        print_summary(name, folds)
        loaded.append((name, folds))

    if not loaded:
        return

    # --- プロット ---
    ncol = min(2, len(loaded))
    nrow = (len(loaded) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4.5 * nrow),
                             squeeze=False)
    for i, (name, folds) in enumerate(loaded):
        ax = axes[i // ncol][i % ncol]
        plot_condition(ax, folds, name, show_legend=(i == 0))
    for j in range(len(loaded), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"\n図を保存しました: {args.out}")

    # --- 条件間の比較表 ---
    if len(loaded) >= 2:
        print(f"\n{'=' * 78}")
        print("  条件間の比較（5 fold 平均）")
        print(f"{'=' * 78}")
        print(f"  {'条件':<22} {'Val最小ep':>10} {'Test最小ep':>11} "
              f"{'Test最小':>10} {'Test@選択':>11} {'損失':>8}")
        print("  " + "-" * 72)
        for name, folds in loaded:
            rows = summarise(folds)
            f = lambda k: np.mean([r[k] for r in rows])
            print(f"  {name:<22} {f('val_min_ep'):>10.1f} "
                  f"{f('test_min_ep'):>11.1f} {f('test_min'):>10.4f} "
                  f"{f('test_at_val_min'):>11.4f} {f('gap'):>+8.4f}")


if __name__ == "__main__":
    main()
