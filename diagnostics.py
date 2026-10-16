"""
診断スクリプト: 保存済みの test_results_fold{N}.npy から指標を計算する

trainer.py の変更は不要。run.py が best epoch 時点で保存している
pred_probs / true_softs から、以下をすべて後処理で算出する。

  - NLL           : -Σ y_k log ŷ_k（ログの値と一致するはず。整合性チェックになる）
  - H(y)          : 正解分布のエントロピー = NLL の理論的下限
  - KL(y‖ŷ)       : NLL - H(y) = 実際の予測誤差
  - H(ŷ)          : 予測分布のエントロピー → 過信か潰れかの判別
  - ECE / MCE     : ソフトラベル版の較正誤差
  - 一様分布NLL   : log(5) = 1.6094（基準線）

使い方:
  # 1条件分（5 fold をまとめて）
  python diagnostics.py --dir saved_models/seed100/ma2nma/multimodal

  # 複数条件を並べて比較
  python diagnostics.py --compare \
      saved_models/seed100/ma2nma/multimodal:提案_MA学習 \
      saved_models/seed100/ma2nma/simple_nn:文脈なし_MA学習 \
      saved_models/seed100/nma/multimodal:提案_MA+NMA学習 \
      saved_models/seed100/nma/simple_nn:文脈なし_MA+NMA学習
"""

import os
import glob
import argparse

import numpy as np

N_CLASS = 5
UNIFORM_NLL = float(np.log(N_CLASS))  # 1.6094
CLASS_NAMES = ["neutral", "happy", "sad", "angry", "other"]
EPS = 1e-10


def entropy(p, axis=-1):
    """確率分布のエントロピー。p は正規化済みを仮定。"""
    return -np.where(p > 0, p * np.log(p + EPS), 0.0).sum(axis=axis)


def soft_ece(probs, softs, n_bins=10):
    """
    ソフトラベル版の ECE / MCE。

    予測の最大確率を信頼度 conf とし、そのクラスが正解分布で持つ確率を acc とする。
    ハードラベル版の「当たったか外れたか(0/1)」を、ソフトラベルでの確率に置き換えた形。
    ACL2024 の Table 3（MLE* の ECE 0.151 / EDL*(R2) の 0.025）と同じ趣旨の量。
    """
    conf = probs.max(axis=1)
    idx = probs.argmax(axis=1)
    acc = softs[np.arange(len(idx)), idx]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    n = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        gap = abs(acc[m].mean() - conf[m].mean())
        ece += (m.sum() / n) * gap
        mce = max(mce, gap)
    return ece, mce


def load_fold(path):
    """npy を読み、予測分布と正規化済み正解分布を返す。"""
    d = np.load(path, allow_pickle=True).item()
    probs = np.asarray(d["pred_probs"], dtype=float)
    softs = np.asarray(d["true_softs"], dtype=float)

    # 予測は念のため再正規化（EDL の場合 alpha/sum_alpha なので既に和1）
    probs = probs / (probs.sum(axis=1, keepdims=True) + EPS)
    # 正解は投票数で入っている場合があるので必ず正規化
    softs = softs / (softs.sum(axis=1, keepdims=True) + EPS)

    return probs, softs, d.get("epoch"), d.get("fold")


def divergences(probs, softs):
    """発話ごとの各種ダイバージェンス／距離を返す。

    KL は非有界で、正解クラス y_k>0 に対し ŷ_k≈0 のとき log(y/ŷ) が
    発散的に大きくなるため、少数の大外しに平均が引きずられる。
    TVD（有界・線形）と BD（中間的）を併記して構造を確認する。
    """
    nll = -(softs * np.log(probs + EPS)).sum(axis=1)
    h_true = entropy(softs)
    kl = nll - h_true

    # 総変動距離: (1/2)Σ|y-ŷ|  ∈ [0,1]
    tvd = 0.5 * np.abs(softs - probs).sum(axis=1)

    # Bhattacharyya 係数と距離、Hellinger 距離
    bc = np.sqrt(np.clip(softs, 0, None) * np.clip(probs, 0, None)).sum(axis=1)
    bc = np.clip(bc, EPS, 1.0)
    bd = -np.log(bc)                      # ∈ [0, ∞)
    hel = np.sqrt(np.clip(1.0 - bc, 0, None))  # ∈ [0,1]

    return {"nll": nll, "kl": kl, "h_true": h_true,
            "h_pred": entropy(probs), "tvd": tvd, "bd": bd, "hel": hel}


def analyse_fold(probs, softs):
    """1 fold 分の指標を返す。"""
    d = divergences(probs, softs)
    nll, kl = d["nll"], d["kl"]
    h_true, h_pred = d["h_true"], d["h_pred"]
    ece, mce = soft_ece(probs, softs)

    # KL が外れ値に引きずられているかを見る
    kl_sorted = np.sort(kl)[::-1]
    top10 = kl_sorted[:max(1, len(kl) // 10)].sum() / (kl.sum() + EPS)

    return {
        "n": len(probs),
        "nll": nll.mean(),
        "h_true": h_true.mean(),
        "kl": kl.mean(),
        "kl_median": float(np.median(kl)),
        "kl_top10_share": float(top10),
        "tvd": d["tvd"].mean(),
        "bd": d["bd"].mean(),
        "hel": d["hel"].mean(),
        "h_pred": h_pred.mean(),
        "h_pred_std": h_pred.std(),
        "ece": ece,
        "mce": mce,
        # 分布の形を見るための補助量
        "maxprob": probs.max(axis=1).mean(),
        "frac_overconfident": float((h_pred < 0.5).mean()),
        "frac_flat": float((h_pred > 1.4).mean()),
    }


def analyse_dir(d, verbose=True):
    """1条件（5 fold）をまとめて集計する。"""
    files = sorted(glob.glob(os.path.join(d, "test_results_fold*.npy")))
    if not files:
        print(f"  [!] npy が見つかりません: {d}")
        return None

    rows = []
    all_probs, all_softs = [], []
    for f in files:
        probs, softs, ep, fold = load_fold(f)
        r = analyse_fold(probs, softs)
        r["fold"] = fold if fold is not None else os.path.basename(f)
        r["epoch"] = ep
        rows.append(r)
        all_probs.append(probs)
        all_softs.append(softs)

    if verbose:
        print(f"\n  fold ごとの結果 ({len(files)} folds)")
        print(f"  {'fold':>5} {'n':>6} {'NLL':>8} {'KL':>8} {'TVD':>7} "
              f"{'BD':>7} {'H(y^)':>8} {'ECE':>7} {'epoch':>6}")
        print("  " + "-" * 68)
        for r in rows:
            print(f"  {str(r['fold']):>5} {r['n']:>6} {r['nll']:>8.4f} "
                  f"{r['kl']:>8.4f} {r['tvd']:>7.4f} {r['bd']:>7.4f} "
                  f"{r['h_pred']:>8.4f} {r['ece']:>7.4f} {str(r['epoch']):>6}")

    keys = ["nll", "h_true", "kl", "kl_median", "kl_top10_share",
            "tvd", "bd", "hel", "h_pred", "ece", "mce",
            "maxprob", "frac_overconfident", "frac_flat"]
    agg = {k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows]))
           for k in keys}
    agg["n_total"] = sum(r["n"] for r in rows)
    agg["_probs"] = np.concatenate(all_probs)
    agg["_softs"] = np.concatenate(all_softs)
    return agg


# =============================================================================
# 追加診断: A（曖昧さの認識）/ B（クラス間の配分）/ C（予測の固着）
# =============================================================================

def _rankdata(x):
    """同順位を平均化した順位（scipy 非依存）。"""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals))
    np.add.at(sums, inv, r)
    return (sums / cnt)[inv]


def diag_A_ambiguity_awareness(probs, softs):
    """
    仮説A: モデルは「曖昧さの程度」を認識できているか。

    正解分布のエントロピー H(y) と予測分布のエントロピー H(y^) の相関を見る。
      相関が高い → 曖昧な発話ほど広い分布を出せている（望ましい）
      相関が低い → 入力によらず同じ広がりを出しているだけ

    NMA の H(y) は同点パターンで決まるため離散値になる
    （2クラス同点=0.693 / 3クラス同点=1.099 など）。
    H(y) の水準ごとの H(y^) 平均も出す。
    """
    h_true = entropy(softs)
    h_pred = entropy(probs)

    pearson = (float(np.corrcoef(h_true, h_pred)[0, 1])
               if h_true.std() > 1e-9 and h_pred.std() > 1e-9 else float("nan"))
    rt, rp = _rankdata(h_true), _rankdata(h_pred)
    spearman = (float(np.corrcoef(rt, rp)[0, 1])
                if rt.std() > 1e-9 and rp.std() > 1e-9 else float("nan"))

    print("\n  [A] 曖昧さの認識")
    print(f"    H(y) と H(y^) の相関 : Pearson {pearson:+.4f} / "
          f"Spearman {spearman:+.4f}")
    print(f"    H(y^) の標準偏差      : {h_pred.std():.4f}"
          f"   （小さいほど入力によらず一定）")
    print(f"    H(y^) の範囲          : {h_pred.min():.4f} 〜 {h_pred.max():.4f}")

    levels = np.unique(np.round(h_true, 4))
    if len(levels) <= 15:
        nll = -(softs * np.log(probs + EPS)).sum(axis=1)
        print(f"\n    {'H(y)':>8} {'件数':>6} {'H(y^)平均':>11} "
              f"{'H(y^)std':>10} {'KL平均':>9}")
        print("    " + "-" * 48)
        for lv in levels:
            m = np.isclose(np.round(h_true, 4), lv)
            if m.sum() < 3:
                continue
            kl = (nll[m] - h_true[m]).mean()
            print(f"    {lv:>8.4f} {m.sum():>6} {h_pred[m].mean():>11.4f} "
                  f"{h_pred[m].std():>10.4f} {kl:>9.4f}")

    print()
    if not np.isnan(spearman):
        if abs(spearman) < 0.15:
            print("    → 相関がほぼゼロ。曖昧さの程度を認識できておらず、")
            print("       入力によらず同程度の広がりを出していると考えられる。")
        elif spearman > 0.3:
            print("    → 正の相関あり。曖昧な発話ほど広い分布を出せている。")
        else:
            print("    → 弱い相関。部分的にしか曖昧さを反映できていない。")

    return {"pearson": pearson, "spearman": spearman,
            "h_pred_std": float(h_pred.std())}


def diag_B_mass_matrix(probs, softs):
    """
    仮説B: どのクラスからどのクラスへ確率質量が流れているか。

    行 = 正解クラス、列 = 予測クラス、値 = Σ_utt y_i * p_j を行方向に正規化。
    これは誤分類件数ではなく、正解分布の確率質量が予測分布へどう配分されたか
    を表す期待値。argmax を使わないため tie-breaking の影響を受けない。
    
    さらに M_ideal = Σ_utt y_i * y_j を基準として差を出す。これは予測分布が
    正解分布 y と一致した場合の配分で、曖昧な正解に由来するクラス間の重なりも
    含む。差の正値は基準より多い配分、負値は基準より少ない配分を意味し、
    誤りの割合そのものではない。
    """
    K = probs.shape[1]
    M_pred = softs.T @ probs
    M_ideal = softs.T @ softs
    row = M_pred.sum(axis=1, keepdims=True)
    Mn = np.divide(M_pred, row, out=np.zeros_like(M_pred), where=row > 0)
    ideal_n = np.divide(M_ideal, row, out=np.zeros_like(M_ideal), where=row > 0)
    excess = Mn - ideal_n

    print("\n  [B] 確率質量の配分行列（行=正解, 列=予測, 行方向に正規化）")
    print(f"    {'正解\\予測':<13}" + "".join(f"{n:>10}" for n in CLASS_NAMES)
          + f"{'正解質量':>10}")
    print("    " + "-" * 75)
    for i in range(K):
        cells = "".join(
            (f"{Mn[i][j]:>9.3f}*" if i == j else f"{Mn[i][j]:>10.3f}")
            for j in range(K))
        print(f"    {CLASS_NAMES[i]:<13}{cells}{row[i][0]:>10.1f}")
    print(f"\n    対角（正しい配分）の平均: {np.diag(Mn).mean():.4f}")

    off = sorted(((Mn[i][j], i, j) for i in range(K) for j in range(K)
                  if i != j), reverse=True)
    print("    主な流出（正解 → 予測）:")
    for v, i, j in off[:5]:
        print(f"      {CLASS_NAMES[i]:>12} → {CLASS_NAMES[j]:<12} {v:.3f}")

    mt, mp = softs.mean(axis=0), probs.mean(axis=0)
    print("\n    周辺分布の比較（全発話平均）")
    print(f"    {'クラス':<14}{'正解':>9}{'予測':>9}{'差':>10}")
    print("    " + "-" * 42)
    for k in range(K):
        print(f"    {CLASS_NAMES[k]:<14}{mt[k]:>9.4f}{mp[k]:>9.4f}"
              f"{mp[k] - mt[k]:>+10.4f}")

    return {"diag_mean": float(np.diag(Mn).mean())}


def diag_C_stickiness(probs):
    """
    仮説C: 連続する発話で予測が固着していないか。

    npy の保存順は対話内の発話順を保っているので、隣接発話の予測分布間の
    距離を測れば固着の度合いが分かる。fold 境界や対話の切れ目も隣接として
    数えるため、絶対値ではなく条件間の相対比較に使う。
    """
    n = len(probs)
    if n < 10:
        return None

    adj = np.abs(probs[1:] - probs[:-1]).sum(axis=1)

    rng = np.random.default_rng(0)
    m = min(20000, n * 10)
    i1, i2 = rng.integers(0, n, size=m), rng.integers(0, n, size=m)
    rand = np.abs(probs[i1] - probs[i2]).sum(axis=1)

    am = probs.argmax(axis=1)
    runs, cur = [], 1
    for i in range(1, n):
        if am[i] == am[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    runs = np.array(runs)

    ratio = adj.mean() / (rand.mean() + EPS)

    print("\n  [C] 予測の固着")
    print(f"    隣接発話間の L1 距離 : {adj.mean():.4f} (± {adj.std():.4f})")
    print(f"    ランダム2発話の L1   : {rand.mean():.4f} (± {rand.std():.4f})")
    print(f"    比（隣接/ランダム）   : {ratio:.4f}   （小さいほど固着が強い）")
    print(f"    同一 argmax の連続長 : 平均 {runs.mean():.2f} / "
          f"最大 {runs.max()} / 中央値 {np.median(runs):.0f}")
    print(f"    連続5以上の区間の割合: {(runs >= 5).sum() / len(runs) * 100:.1f}%")

    return {"ratio": float(ratio), "run_mean": float(runs.mean()),
            "run_max": int(runs.max()), "adj": float(adj.mean())}


def report(name, agg):
    """1条件の診断レポートを表示する。"""
    print(f"\n{'=' * 68}")
    print(f"  {name}")
    print(f"{'=' * 68}")
    print(f"  発話数: {agg['n_total']}")
    print()
    print(f"  NLL          {agg['nll'][0]:.4f} (± {agg['nll'][1]:.4f})")
    print(f"  H(y)         {agg['h_true'][0]:.4f} (± {agg['h_true'][1]:.4f})"
          f"   ← NLL の理論的下限")
    print(f"  KL(y||y^)    {agg['kl'][0]:.4f} (± {agg['kl'][1]:.4f})"
          f"   ← 実際の予測誤差")
    print(f"  一様分布NLL  {UNIFORM_NLL:.4f}             ← 無学習モデルの水準")
    print()
    print(f"  H(y^)        {agg['h_pred'][0]:.4f} (± {agg['h_pred'][1]:.4f})"
          f"   ← 予測分布の平坦さ")
    print(f"  max prob     {agg['maxprob'][0]:.4f}")
    print(f"  ECE          {agg['ece'][0]:.4f} (± {agg['ece'][1]:.4f})")
    print(f"  MCE          {agg['mce'][0]:.4f}")
    print()
    print(f"  H(y^) < 0.5 の割合（過信）  {agg['frac_overconfident'][0] * 100:.1f}%")
    print(f"  H(y^) > 1.4 の割合（平坦）  {agg['frac_flat'][0] * 100:.1f}%")

    # --- 自動診断 ---
    print()
    print("  [診断]")
    nll, h_pred, h_true = agg["nll"][0], agg["h_pred"][0], agg["h_true"][0]

    if nll > UNIFORM_NLL:
        print(f"  * NLL が一様分布({UNIFORM_NLL:.4f})を上回っている。"
              f"無学習モデル以下の性能。")
    else:
        print(f"  * NLL は一様分布を {UNIFORM_NLL - nll:.4f} 下回っている。")

    print(f"  * 下限 H(y)={h_true:.4f} までの距離（KL）は {agg['kl'][0]:.4f}。")

    if h_pred < 0.5:
        print(f"  * H(y^)={h_pred:.4f} と小さい → 【過信】鋭い予測をして外している。"
              f"較正の問題。卒論4.8.1節の考察（文脈に引きずられる）と整合。")
    elif h_pred > 1.4:
        print(f"  * H(y^)={h_pred:.4f} と大きい → 【潰れ】ほぼ一様分布を出している。"
              f"学習が機能していない可能性。")
    else:
        print(f"  * H(y^)={h_pred:.4f} は中間。過信でも潰れでもない。"
              f"H(y)={h_true:.4f} との差 {h_pred - h_true:+.4f} に注目。")

    if h_pred < h_true - 0.2:
        print(f"  * 予測が正解分布より鋭い（H(y^) < H(y)）。"
              f"曖昧さを表現できていない。")
    elif h_pred > h_true + 0.2:
        print(f"  * 予測が正解分布より平坦（H(y^) > H(y)）。"
              f"必要以上に判断を保留している。")

    # --- 追加診断 A / B / C ---
    if "_probs" in agg and "_softs" in agg:
        diag_A_ambiguity_awareness(agg["_probs"], agg["_softs"])
        diag_B_mass_matrix(agg["_probs"], agg["_softs"])
        diag_C_stickiness(agg["_probs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="1条件のディレクトリ")
    ap.add_argument("--compare", nargs="+",
                    help="複数条件。'パス:表示名' の形式で並べる")
    ap.add_argument("--bins", type=int, default=10, help="ECE のビン数")
    args = ap.parse_args()

    if not args.dir and not args.compare:
        ap.error("--dir か --compare のどちらかを指定してください")

    print(f"\n一様分布の NLL (log {N_CLASS}) = {UNIFORM_NLL:.4f}")

    if args.dir:
        agg = analyse_dir(args.dir)
        if agg:
            report(os.path.basename(args.dir.rstrip("/")), agg)
        return

    # --- 複数条件の比較 ---
    results = []
    for spec in args.compare:
        if ":" in spec:
            path, name = spec.rsplit(":", 1)
        else:
            path, name = spec, os.path.basename(spec.rstrip("/"))
        print(f"\n読み込み中: {path}")
        agg = analyse_dir(path, verbose=False)
        if agg:
            results.append((name, agg))
            report(name, agg)

    if len(results) < 2:
        return

    print(f"\n{'=' * 84}")
    print("  条件間の比較")
    print(f"{'=' * 84}")
    print(f"  {'条件':<26} {'NLL':>8} {'KL':>8} {'TVD':>8} {'BD':>8} "
          f"{'H(y^)':>8} {'ECE':>8}")
    print("  " + "-" * 78)
    for name, a in results:
        print(f"  {name:<26} {a['nll'][0]:>8.4f} {a['kl'][0]:>8.4f} "
              f"{a['tvd'][0]:>8.4f} {a['bd'][0]:>8.4f} "
              f"{a['h_pred'][0]:>8.4f} {a['ece'][0]:>8.4f}")

    # --- KL と TVD で順位が一致するか ---
    print()
    print("  " + "=" * 74)
    print("  指標ごとの順位（1位が最良）")
    print("  " + "=" * 74)

    metrics = [("KL", "kl"), ("TVD", "tvd"), ("BD", "bd"),
               ("Hellinger", "hel"), ("NLL", "nll"), ("ECE", "ece")]
    ranks = {}
    for label, key in metrics:
        order = sorted(range(len(results)), key=lambda i: results[i][1][key][0])
        r = [0] * len(results)
        for pos, idx in enumerate(order):
            r[idx] = pos + 1
        ranks[label] = r

    print(f"  {'条件':<26}" + "".join(f"{lb:>11}" for lb, _ in metrics))
    print("  " + "-" * 74)
    for i, (name, _) in enumerate(results):
        print(f"  {name:<26}"
              + "".join(f"{ranks[lb][i]:>11}" for lb, _ in metrics))

    kl_r, tvd_r = ranks["KL"], ranks["TVD"]
    print()
    if kl_r == tvd_r:
        print("  → KL と TVD の順位は一致。どの指標で見ても同じ結論になる。")
    else:
        print("  → KL と TVD で順位が入れ替わっている。")
        for i, (name, _) in enumerate(results):
            if kl_r[i] != tvd_r[i]:
                print(f"     {name}: KL {kl_r[i]}位 / TVD {tvd_r[i]}位")
        print()
        print("     KL は正解クラスへのゼロ配分を強く罰するため、少数の大外しに")
        print("     引きずられる。TVD で順位が上がる条件は「平均的な分布の形は")
        print("     良いが、時々大きく外す」性質をもつ。逆に TVD で順位が下がる")
        print("     条件は「大外しを避けているが平均的には正解から離れている」")
        print("     安全策寄りと読める。")

    # KL の外れ値依存度
    print()
    print(f"  {'条件':<26}{'KL平均':>10}{'KL中央値':>11}{'上位10%寄与':>13}")
    print("  " + "-" * 60)
    for name, a in results:
        print(f"  {name:<26}{a['kl'][0]:>10.4f}{a['kl_median'][0]:>11.4f}"
              f"{a['kl_top10_share'][0] * 100:>12.1f}%")
    print()
    print(f"  （参考）一様分布: NLL {UNIFORM_NLL:.4f} / H(y^) {UNIFORM_NLL:.4f}")

    print()
    print("  参考: ACL2024 Table 4 (IEMOCAP, NMA)")
    print(f"  {'MLE (ハードラベル)':<28} {1.924:>8.4f}")
    print(f"  {'MLE* (ソフト+softmax)':<28} {1.137:>8.4f}   ← 比較すべき相手")
    print(f"  {'EDL*(R1)':<28} {0.951:>8.4f}")
    print(f"  {'EDL*(R2)':<28} {0.953:>8.4f}")
    print("  ※ ただし ACL の NMA テストは約8割が学習済み話者。"
          "こちらは話者完全分離なので単純比較は不可。")


if __name__ == "__main__":
    main()
