"""One-shot, local only (needs latex + dvisvgm, e.g. MiKTeX; not on the cluster), run from repo root:
exp1/results/combined/table*.tex -> figures/tables/<name>.svg (PowerPoint / preview)
  - thesis preamble (MA-tex main.tex: article -> 345 pt text width) + longtable; tall page -> longtable on one page
  - glyphs as paths (no font dependency)
  - flags tables wider than the text width (LaTeX overfull warning; longtable has no resizebox)
"""
import glob
import os
import re
import subprocess
import tempfile

IN_DIR = "exp1/results/combined"
OUT_DIR = "exp1/thesis_plots/figures/tables"
DOC = r"""\documentclass{article}
\usepackage{amsmath,amssymb,booktabs,graphicx,longtable}
\pagestyle{empty}
\setlength{\textheight}{200cm}
\begin{document}
\input{%s}
\end{document}
"""


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120, check=True).stdout


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for tex in sorted(glob.glob(f"{IN_DIR}/table*.tex")):
        name = os.path.splitext(os.path.basename(tex))[0]
        out = os.path.abspath(f"{OUT_DIR}/{name}.svg")
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/doc.tex", "w") as f:
                f.write(DOC % os.path.abspath(tex).replace("\\", "/"))
            for _ in range(2):  # longtable column widths settle on the 2nd run
                log = run(["latex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"], tmp)
            run(["dvisvgm", "--no-fonts", "--bbox=min", "-o", out, "doc.dvi"], tmp)
        over = [float(w) for w in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)]
        print(f"Wrote {out}" + (f"  -- {max(over):.0f} pt wider than the text width" if over else ""))


if __name__ == "__main__":
    main()
