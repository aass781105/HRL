# plot_yseries_png.py
import json, ast, re
from pathlib import Path
from typing import List, Tuple, Union
import matplotlib.pyplot as plt

# ======= 在這裡設定 =======
INPUT_TXT  = "train_log/SD2/reward_dynamic_PPO.txt"    # 只含 y 值的 txt（或含 [x,y] 也可）
OUTPUT_PNG = "log_plot/reward_10x5.png"
TITLE      = None                            # 想加標題就填字串；否則 None
LINE_WIDTH = 2
# =========================

# 數字樣式（含科學記號）
NUM = r"-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?"
PAIR_RE = re.compile(r"\[\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*\]")
NUM_LIST_RE = re.compile(NUM)

def parse_content(text: str) -> Union[List[Tuple[float,float]], List[float]]:
    text = text.strip()

    # 試 1：JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in data):
                return [(float(p[0]), float(p[1])) for p in data]  # [ [x,y], ... ]
            if all(isinstance(v, (int, float)) for v in data):
                return [float(v) for v in data]                    # [ y, y, ... ]
    except Exception:
        pass

    # 試 2：Python literal
    try:
        data = ast.literal_eval(text)
        if isinstance(data, list):
            if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in data):
                return [(float(p[0]), float(p[1])) for p in data]
            if all(isinstance(v, (int, float)) for v in data):
                return [float(v) for v in data]
    except Exception:
        pass

    # 試 3：正則擷取
    pairs = [(float(a), float(b)) for a, b in PAIR_RE.findall(text)]
    if pairs:
        return pairs
    nums = [float(n) for n in NUM_LIST_RE.findall(text)]
    if nums:
        return nums

    raise ValueError("無法在檔案中解析出資料（支援 [ [x,y],... ] 或 [y,...]）。")

def load_xy(txt_path: Path) -> List[Tuple[float,float]]:
    text = txt_path.read_text(encoding="utf-8")
    parsed = parse_content(text)
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], tuple):
        return parsed  # 已是 (x,y)
    # 否則是 y 序列 → 用索引當 x
    ys = parsed  # type: ignore
    return list(enumerate(ys))

def plot_and_save_png(pairs: List[Tuple[float, float]], png_path: Path, title: str = None):
    xs, ys = zip(*pairs)
    plt.figure(figsize=(8, 4.5))
    # 純折線（不要點）
    plt.plot(xs, ys, linestyle='-', marker=None, linewidth=LINE_WIDTH)
    # plt.xlabel("validate every 10 Ep")
    plt.xlabel("Ep")
    # plt.ylabel("makespans")
    plt.ylabel("r")
    if title:
        plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=200)
    plt.close()

def main():
    in_path = Path(INPUT_TXT).expanduser().resolve()
    out_path = Path(OUTPUT_PNG).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"找不到檔案：{in_path}")

    pairs = load_xy(in_path)
    title = TITLE if TITLE is not None else f"{in_path.name} ({len(pairs)} points)"
    plot_and_save_png(pairs, out_path, title=title)
    print(f"✅ PNG 已輸出：{out_path}")

if __name__ == "__main__":
    main()
