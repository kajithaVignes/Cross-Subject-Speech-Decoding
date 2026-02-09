from pathlib import Path

def make_words_txt(lexicon_path, out_path):
    lexicon_path = Path(lexicon_path)
    out_path = Path(out_path)

    words = set()
    with lexicon_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            w = line.split()[0]
            if w == "<eps>":
                raise ValueError("lexicon.txt must not contain <eps> as a word")
            words.add(w.upper())

    words_sorted = sorted(words)

    lines = []
    lines.append("<eps> 0\n")
    for i, w in enumerate(words_sorted, start=1):
        lines.append(f"{w} {i}\n")

    n = len(words_sorted)
    lines.append(f"#0 {n+1}\n")
    lines.append(f"<s> {n+2}\n")
    lines.append(f"</s> {n+3}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")

    print(f"Wrote {out_path} with {n} words (+ <eps>, #0, <s>, </s>)")

if __name__ == "__main__":
    make_words_txt(lexicon_path="data/assets/lexicon.txt", out_path="data/assets/words.txt")
