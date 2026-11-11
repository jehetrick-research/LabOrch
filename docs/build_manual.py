"""Build laborch_manual.docx / .tex / .pdf from laborch_manual.md.
Usage: python3 build_manual.py <md> <outdir>"""
import re, subprocess, sys, os, shutil
md, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
src = open(md).read()
# docx straight from the markdown
subprocess.run(["pandoc", md, "-f", "gfm", "-t", "docx", "-o", f"{out}/laborch_manual.docx"], check=True)
# LaTeX: drop the hand Contents list and title lines (LaTeX makes its own)
s = re.sub(r'## Contents\n.*?\n---\n', '', src, flags=re.S)
s = re.sub(r'^# laborch — Package Manual\n\nVersion .*?\n\n', '', s)
open(f"{out}/_tex.md", "w").write(s)
hdr = f"{out}/_hdr.tex"
open(hdr, "w").write(r"""\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\}}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
\usepackage{titling}
\renewcommand{\arraystretch}{1.1}
\setlength{\emergencystretch}{3em}
""")
tex = f"{out}/laborch_manual.tex"
subprocess.run(["pandoc", f"{out}/_tex.md", "-f", "markdown+gfm_auto_identifiers-auto_identifiers", "--columns=60",
    "-s", "-o", tex, "--toc", "--toc-depth=2", "--highlight-style=tango",
    "-V", "documentclass=article", "-V", "geometry:margin=2.2cm", "-V", "fontsize=10pt",
    "-V", "mainfont=DejaVu Serif", "-V", "sansfont=DejaVu Sans", "-V", "monofont=DejaVu Sans Mono",
    "-V", "monofontoptions=Scale=0.85", "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue",
    "-M", "title=laborch: Package Manual", "-M", "subtitle=Version 0.2.0 (Phase 1)",
    "-M", "author=LabOrch project", "-M", "date=November 2026", "-H", hdr], check=True)
t = open(tex).read()
t = t.replace("\\usepackage{lmodern}\n", "")
W = r"\s+"          # pandoc wraps lines; match any whitespace between words
R = [  # (regex, replacement)
 (r"spectra \\emph\{X\}ⱼ:", r"spectra $X_j$:"),
 (r"\\begin\{verbatim\}\nT\(fᵢ\) ≈ Σⱼ Xᵢⱼ αⱼ\s+with αⱼ ≥ 0\n\\end\{verbatim\}",
  r"\\[\n  T(f_i) \;\\approx\; \\sum_j X_{ij}\\,\\alpha_j, \\qquad \\alpha_j \\ge 0\n\\]"),
 (r"\\begin\{verbatim\}\nminimize\s+Σᵢ \(\(Tᵢ − Σⱼ Xᵢⱼ αⱼ\) / σᵢ\)²\s+subject to αⱼ ≥ 0\n\\end\{verbatim\}",
  r"\\[\n  \\min_{\\alpha}\; \\sum_i \\left( \\frac{T_i - \\sum_j X_{ij}\\,\\alpha_j}{\\sigma_i} \\right)^{2}\n  \\quad \\text{subject to} \\quad \\alpha_j \\ge 0\n\\]"),
 (r"\\begin\{verbatim\}\nx\(t\) = Σₖ aₖ sin\(2π pₖ f₀ t \+ φₖ\) · e\^\(−decay·t\)\n\\end\{verbatim\}",
  r"\\[\n  x(t) = e^{-\\lambda t} \\sum_k a_k \\sin\\!\\left(2\\pi\\, p_k f_0\\, t + \\phi_k\\right),\n  \\qquad \\lambda = \\texttt{decay}\n\\]"),
 (r"Target"+W+r"spectrum"+W+r"\\emph\{T\}"+W+r"as", r"Target spectrum $T$ as"),
 (r"The weights α are", r"The weights $\\alpha$ are"),
 (r"by"+W+r"1/σᵢ", r"by $1/\\sigma_i$"),
 (r"combine the frames as √\(Σ\\textbar Xₖ\\textbar²\) / n,", r"combine the frames as $\\bigl(\\sum_{\\text{frames}} |X_k|^2\\bigr)^{1/2} / n$,"),
 (r"the"+W+r"projection"+W+r"pⱼ"+W+r"="+W+r"Σᵢ"+W+r"Tᵢ"+W+r"Xᵢⱼ", r"the projection \\mbox{$p_j = \\sum_i T_i X_{ij}$}"),
 (r"‖T − model‖ / ‖T‖", r"$\\lVert T - \\text{model} \\rVert / \\lVert T \\rVert$"),
 (r"20"+W+r"log₁₀\(w"+W+r"/"+W+r"w\\_max\)", r"$20\\log_{10}(w / w_{\\max})$"),
 (r"\\emph\{k\}"+W+r"×"+W+r"10\.77"+W+r"Hz", r"$k \\times 10.77$~Hz"),
 (r"\\texttt\{\"(\w+)\"\\ →\\ \"([^\"]+)\"\}", r'\\texttt{"\1"}~$\\rightarrow$~\\texttt{"\2"}'),
 (r"\\texttt\{rtol\\ ×\\ max\}", r"\\texttt{rtol}~$\\times$~\\texttt{max}"),
 (r"σᵢ ∝ Tᵢ\}", r"sigma\\_i proportional to T\\_i}"),
 (r"χ²", r"$\\chi^2$"), (r"σ-weighted", r"$\\sigma$-weighted"), (r"\(σ = 1\)", r"($\\sigma = 1$)"),
 (r"σ(?=[ ),.])", r"$\\sigma$"), (r"≥", r"$\\ge$"), (r"≤", r"$\\le$"), (r"≈", r"$\\approx$"),
 (r" × ", r" $\\times$ "), (r"×", r"$\\times$"), (r" → ", r" $\\rightarrow$ "),
]
# apply outside code (Highlighting / verbatim bodies except the math blocks handled above)
for a, b in R[:4]:
    t, n = re.subn(a, b, t); assert n == 1, a
parts = re.split(r"(\\begin\{Highlighting\}.*?\\end\{Highlighting\}|\\begin\{verbatim\}.*?\\end\{verbatim\})", t, flags=re.S)
for i in range(0, len(parts), 2):
    for a, b in R[4:]:
        parts[i] = re.sub(a, b, parts[i])
t = "".join(parts)
for a, b in R[14:15]:
    t = re.sub(a, b, t)
# dagger marks: superscript in text, plain words in PDF bookmarks
t = re.sub(r"\}\n†\}\{(.*?) †\}\}", lambda m: "}~\\textsuperscript{\\textdagger}}{" + m.group(1) + " (needs pydub)}}", t)
t = t.replace("marked †", r"marked \textdagger{}")
# space between table rows
t = re.sub(r"\\endlastfoot\n.*?\\bottomrule",
           lambda m: re.sub(r"\\\\\n(?!\\bottomrule)", r"\\\\ \\addlinespace[5pt]\n", m.group(0)), t, flags=re.S)
t = ("% laborch package manual. Generated from docs/laborch_manual.md by build_manual.py\n"
     "% (pandoc + LaTeX math and table-spacing fixes). Build: xelatex laborch_manual.tex (twice).\n") + t
open(tex, "w").write(t)
left = [l for l in t.split("\n") if re.search(r"[^\x00-\x7F]", l) and not re.search(r"[├│└─§é–—“”’‘]", l)]
print("non-ASCII lines left:", *left[:20], sep="\n  ")
for _ in range(2):
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "laborch_manual.tex"], cwd=out, capture_output=True, text=True)
print("errors:", [l for l in r.stdout.split("\n") if l.startswith("!")])
for f in ("_tex.md", "_hdr.tex"): os.remove(f"{out}/{f}")
