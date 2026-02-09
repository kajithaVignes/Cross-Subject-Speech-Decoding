from pathlib import Path
import re

STRESS_RE = re.compile(r"^([A-Z]+)([0-2])$")

def strip_stress(phone):
    m = STRESS_RE.match(phone)
    if m:
        return m.group(1)
    return phone

def make_lexicon_nostress(in_lex, out_lex, sil_token_out = "sil"):
    in_lex = Path(in_lex)
    out_lex = Path(out_lex)
    assert in_lex.exists(), f"Missing {in_lex}"

    out_lines = []
    with in_lex.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            word = parts[0].upper()
            phones = parts[1:]

            norm_phones = []
            for p in phones:
                p2 = strip_stress(p.upper())
                if p2 in {"SIL", "<SIL>", "SILENCE"}:
                    p2 = sil_token_out
                norm_phones.append(p2)

            out_lines.append(word + " " + " ".join(norm_phones) + "\n")

    out_lex.parent.mkdir(parents=True, exist_ok=True)
    out_lex.write_text("".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_lex}")

if __name__ == "__main__":
    make_lexicon_nostress(in_lex="data/assets/lexicon.txt", out_lex="data/assets/lexicon_nostress.txt", sil_token_out="sil")
