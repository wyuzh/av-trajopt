"""Pipeline diagrams of the four methods (one iteration as a flow of steps, with equations).

    python method_pipelines.py

Writes to ../figures:
    pipeline_m1 ... pipeline_m4, timeline (.pdf/.svg/.png)   sans-serif, for the slides
    pipeline_paper_m14.pdf, _m2.pdf, _m3.pdf            serif, for the report (Fig. 2)

Requires pdflatex (TikZ) and PyMuPDF.
"""
import subprocess
import tempfile
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent.parent / "figures"

PREAMBLE = r"""\documentclass[tikz,border=3pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage[HTML]{xcolor}
%FONT%
\usetikzlibrary{positioning,arrows.meta,fit,backgrounds,calc,decorations.pathreplacing}
\definecolor{cIPM}{HTML}{2E86DE}\definecolor{cSQP}{HTML}{7B3FA0}
\definecolor{cPAN}{HTML}{0E9AA7}\definecolor{cRIC}{HTML}{1B3A6B}
\definecolor{cALM}{HTML}{17A673}\definecolor{cTXT}{HTML}{404040}
\definecolor{cOFF}{HTML}{A6A6A6}
\tikzset{
  stage/.style={draw=#1, fill=#1!7, rounded corners=3pt, line width=0.6pt, align=flush center,
                inner sep=3pt, minimum height=1.75cm, text=cTXT},
  off/.style={draw=cOFF, fill=black!3, rounded corners=3pt, line width=0.6pt, align=flush center,
              inner sep=3pt, minimum height=1.75cm, text=cOFF},
  flow/.style={-{Stealth[length=4.5pt]}, line width=0.8pt, draw=#1},
  io/.style={align=center, text=cTXT, inner sep=1pt},
  note/.style={font=\scriptsize, text=cTXT!80, align=center},
}
\newcommand{\hd}[2]{{\color{#1}\textbf{#2}}}
\hyphenpenalty=10000 \exhyphenpenalty=10000
\begin{document}
\color{cTXT}
"""
SANS = r"""\renewcommand{\familydefault}{\sfdefault}
\usepackage{sansmath}\sansmath"""
SERIF = r"""\usepackage{newtxtext,newtxmath}"""

# ---------------------------------------------------------------------------- M1 / M4
M1_BOXES = {
    "b1": (r"1\enspace Barrier problem",
           r"$\min\ f(z)-\mu_b\textstyle\sum_i\ln s_i$\\ s.t.\ $h=0,\ g+s=0$\\[2pt]"
           r"$\Rightarrow$ relaxed KKT: $Sy=\mu_b e$", "2.85cm"),
    "b2": (r"2\enspace Newton step",
           r"$\begin{bmatrix}H{+}\delta_w I & J_h^{\top}\\ J_h & 0\end{bmatrix}\!"
           r"\begin{bmatrix}\Delta z\\ \Delta\lambda\end{bmatrix}"
           r"={-}\begin{bmatrix} r\\ h\end{bmatrix}$\\[2pt]"
           r"{\scriptsize dense $LDL^{\top}$, $O(n^3)$; raise $\delta_w$ until\\ inertia $=(n,m_{\rm eq},0)$}",
           "4.3cm"),
    "b3": (r"3\enspace Step length",
           r"$s+\alpha\Delta s\ge(1-\tau)s$\\ $y+\alpha\Delta y\ge(1-\tau)y$\\[2pt]"
           r"{\scriptsize filter: accept if violation\\ or barrier cost drops}", "2.75cm"),
    "b4": (r"4\enspace Barrier update",
           r"$\mu_b\leftarrow\min(0.2\mu_b,\ \mu_b^{1.5})$\\[2pt]"
           r"{\scriptsize restoration phase if\\ no step is accepted}", "2.95cm"),
}
RICCATI_BOX = (r"2\enspace Newton step by Riccati recursion",
               r"lift: $\xi_k=(\Delta x_k,\Delta u_{k-1})$, "
               r"$\xi_{k+1}=\hat A_k\xi_k+\hat B_k\Delta u_k+\hat c_k$\\[2pt]"
               r"backward, $k=N{-}1,\dots,0$:\enspace "
               r"$G^k_{uu}=Q^k_{uu}+\hat B_k^{\top}P_{k+1}\hat B_k$\\"
               r"$K_k=-(G^k_{uu})^{-1}G^k_{ux},\quad P_k=G^k_{xx}+G^{k\top}_{ux}K_k$\\[2pt]"
               r"forward: $\Delta u_k=K_k\xi_k+\kappa_k$\\[2pt]"
               r"{\scriptsize $O(N)$ work;\enspace inertia correct $\iff G^k_{uu}\succ 0$ for all $k$ "
               r"($2{\times}2$ Cholesky)}", "7.6cm")


def row(boxes, color, styles=None, looppos=0.25, io=True, start=r"$z,\ s>0$\\$\lambda,\ y>0$",
        loop=r"repeat until violation and KKT residual $\le 10^{-6}$"):
    """A row of stage boxes b1..bk with a loop arrow from the last box back to the first."""
    styles = styles or {}
    keys = list(boxes)
    s = [r"\node[io] (in) {" + start + "};"] if io else []
    prev = "in" if io else None
    for k in keys:
        head, body, w = boxes[k]
        st = styles.get(k, f"stage={color}")
        col = "cOFF" if st == "off" else color
        place = f", right=4mm of {prev}" if prev else ""
        s.append(rf"\node[{st}, text width={w}{place}] ({k}) "
                 rf"{{\hd{{{col}}}{{{head}}}\\[3pt]{body}}};")
        if prev:
            s.append(rf"\draw[flow={color}] ({prev}) -- ({k});")
        prev = k
    if io:
        s.append(rf"\node[io, right=4mm of {prev}] (out) {{KKT\\point}};")
        s.append(rf"\draw[flow={color}] ({prev}) -- (out);")
    first, last = keys[0], keys[-1]
    s.append(r"\coordinate (low) at ([yshift=-0.38cm]current bounding box.south);")
    s.append(rf"\draw[flow={color}] ({last}.south) -- ({last}.south |- low) -| ({first}.south) "
             rf"node[pos={looppos}, below, note] {{{loop}}};")
    return "\n".join(s)


def pic_m1():
    return (r"\begin{tikzpicture}[font=\footnotesize]" + row(M1_BOXES, "cIPM", io=False)
            + r"\end{tikzpicture}")


def pic_m4():
    boxes = dict(M1_BOXES)
    boxes["b2"] = RICCATI_BOX
    styles = {"b1": "off", "b3": "off", "b4": "off", "b2": "stage=cRIC"}
    body = row(boxes, "cRIC", styles, io=False)
    body += (r"\node[note, above=1pt of b1] {same as M1};\node[note, above=1pt of b3] {same as M1};"
             r"\node[note, above=1pt of b4] {same as M1};"
             r"\node[note, above=1pt of b2, text=cRIC] {\textbf{only change:} the linear solve};")
    return r"\begin{tikzpicture}[font=\footnotesize]" + body + r"\end{tikzpicture}"


def pic_m14():
    """Paper: the M1 row, with the M4 replacement of step 2 hanging below it."""
    rbox = RICCATI_BOX
    body = row(M1_BOXES, "cIPM", looppos=0.1, io=False)
    body += (rf"\node[stage=cRIC, text width={rbox[2]}, below=9mm of b2] (r2) "
             r"{\hd{cRIC}{M4: step 2 by Riccati recursion}\\[3pt]" + rbox[1] + "};"
             r"\draw[-{Stealth[length=4.5pt]}, dashed, line width=0.8pt, draw=cRIC] "
             r"(b2.south |- low) -- (r2.north) node[midway, left, note, text=cRIC] "
             r"{same system, solved in $O(N)$};")
    return r"\begin{tikzpicture}[font=\footnotesize]" + body + r"\end{tikzpicture}"


# ---------------------------------------------------------------------------- M2
M2_BOXES = {
    "b1": (r"1\enspace Linearise",
           r"$\nabla f,\ J_h,\ J_g,\ \nabla^2_{zz}\mathcal L$\\[2pt]"
           r"{\scriptsize exact derivatives\\ (CasADi)}", "2.7cm"),
    "b2": (r"2\enspace Convexify",
           r"$B=\nabla^2_{zz}\mathcal L+\rho J_{\mathcal A}^{\top}J_{\mathcal A}\succ 0$\\[2pt]"
           r"{\scriptsize Finsler: same QP solution\\ ($\mathcal A$ = active constraints)}", "3.6cm"),
    "b3": (r"3\enspace QP subproblem",
           r"$\min_d\ \nabla f^{\top}d+\tfrac12 d^{\top}Bd$\\ s.t.\ $h+J_h d=0$\\ \phantom{s.t.\ }$g+J_g d\le 0$\\[2pt]"
           r"{\scriptsize gives $d,\ \lambda_{\rm QP},\ \mu_{\rm QP}$;\\ elastic $\ell_1$ QP if infeasible}",
           "3.45cm"),
    "b4": (r"4\enspace Filter line search",
           r"$z\leftarrow z+\alpha d$\\ $(\lambda,\mu)\leftarrow(\lambda_{\rm QP},\mu_{\rm QP})$\\[2pt]"
           r"{\scriptsize accepts steps a merit\\ function rejects (Maratos)}", "3.1cm"),
}


def pic_m2():
    return (r"\begin{tikzpicture}[font=\footnotesize]"
            + row(M2_BOXES, "cSQP", start=r"$z$\\$\lambda,\ \mu$") + r"\end{tikzpicture}")


# ---------------------------------------------------------------------------- M3
def pic_m3():
    c = "cPAN"
    return rf"""\begin{{tikzpicture}}[font=\footnotesize]
\node[io] (in) {{$z,\ \lambda,\ \mu$\\$\rho$}};
\node[stage={c}, text width=2.75cm, right=4mm of in] (o1)
  {{\hd{{{c}}}{{Outer: ALM subproblem}}\\[3pt]$\min_{{z\in C}}\ \psi(z)=\mathcal L_A(z;\lambda,\mu,\rho)$\\[2pt]
   {{\scriptsize $C=\{{l\le z\le u\}}$: 400 simple bounds,\\ handled by projection}}}};
\node[stage={c}, text width=3.0cm, right=9mm of o1] (i1)
  {{\hd{{{c}}}{{a\enspace Projected gradient}}\\[3pt]$\bar z=\Pi_C\big(z-\gamma\nabla\psi(z)\big)$\\ $r=z-\bar z$}};
\node[stage={c}, text width=2.15cm, right=4mm of i1] (i2)
  {{\hd{{{c}}}{{b\enspace L-BFGS}}\\[3pt]$d=-H_k\,r$\\[2pt]{{\scriptsize from pairs\\ $(\Delta z,\ \Delta r)$}}}};
\node[stage={c}, text width=3.3cm, right=4mm of i2] (i3)
  {{\hd{{{c}}}{{c\enspace Envelope line search}}\\[3pt]$z^+=z-(1-\tau)\,r+\tau\,d$\\
   $\varphi_\gamma(z^+)\le\varphi_\gamma(z)-\sigma\|r\|^2$\\[2pt]{{\scriptsize $\tau=1,\tfrac12,\dots,0$\\ ($\tau=0$ always passes)}}}};
\node[stage={c}, text width=3.0cm, right=9mm of i3] (o2)
  {{\hd{{{c}}}{{Outer: dual update}}\\[3pt]$\lambda\leftarrow\lambda+\rho\,h(z)$\\ $\mu\leftarrow\max\big(0,\ \mu+\rho\,g(z)\big)$\\[2pt]
   {{\scriptsize $\rho\leftarrow 5\rho$ if the violation\\ does not drop}}}};
\node[io, right=4mm of o2] (out) {{KKT\\point}};
\draw[flow={c}] (in) -- (o1); \draw[flow={c}] (o1) -- (i1); \draw[flow={c}] (i1) -- (i2);
\draw[flow={c}] (i2) -- (i3); \draw[flow={c}] (i3) -- (o2); \draw[flow={c}] (o2) -- (out);
\draw[flow={c}] (i3.south) -- ++(0,-0.3) -| (i1.south) node[pos=0.25, below, note]
  {{inner loop until $\|r\|_\infty/\gamma\le\varepsilon_k$}};
\begin{{scope}}[on background layer]
\node[draw={c}, dashed, rounded corners=5pt, fit=(i1)(i2)(i3), inner xsep=4pt, inner ysep=19pt,
      yshift=-4pt, label={{[note, text={c}, yshift=-1pt]above:\textbf{{PANOC inner solver: gradients and a projection only, no Hessian, no linear system}}}}] (box) {{}};
\end{{scope}}
\draw[flow={c}] (o2.south) -- ++(0,-1.05) -| (o1.south) node[pos=0.25, below, note]
  {{outer loop until violation and KKT residual $\le 10^{{-6}}$;\enspace $\varepsilon_k\leftarrow\varepsilon_k/10$}};
\end{{tikzpicture}}"""


# ---------------------------------------------------------------------------- timeline
def pic_timeline():
    """Family tree of the four methods along a time axis (slides only)."""
    def X(year):                       # cm along the axis
        return (year - 1935) / 92 * 18.5
    lanes = {"ipm": 2.4, "sqp": 0.75, "alm": -0.95, "ric": 4.1}
    s = [r"\begin{tikzpicture}[font=\normalsize]"]
    # time axis
    s.append(rf"\draw[-{{Stealth[length=5pt]}}, line width=0.8pt, cOFF] ({X(1935)},-2.2) -- ({X(2028)},-2.2);")
    for yr in (1940, 1960, 1980, 2000, 2020):
        s.append(rf"\draw[cOFF] ({X(yr)},-2.28) -- ({X(yr)},-2.12) node[below=3pt, text=cOFF, font=\normalsize] {{{yr}}};")
    # trunk: KKT conditions
    s.append(rf"\draw[line width=3pt, cTXT!55, line cap=round] ({X(1939)},0.75) -- ({X(1958)},0.75);")
    s.append(rf"\fill[cTXT] ({X(1939)},0.75) circle (2.6pt) node[above=2pt, align=center, font=\normalsize] "
             r"{\textbf{Karush 1939}}"
             rf" ({X(1951)},0.75) circle (2.6pt) node[below=2pt, align=center, font=\normalsize] "
             r"{\textbf{Kuhn, Tucker 1951}\\KKT conditions};")
    s.append(rf"\node[font=\normalsize\itshape, text=cTXT, align=center] at ({X(1949)},1.75) {{optimality = equations}};")

    def branch(y0, y1, x0, x1, col, w=2.2):
        return (rf"\draw[line width={w}pt, {col}, line cap=round] ({x0},{y0}) "
                rf"to[out=0, in=180] ({x1},{y1});")

    def lane(y, x0, x1, col, w=2.2):
        return rf"\draw[line width={w}pt, {col}, line cap=round] ({x0},{y}) -- ({x1},{y});"

    def mile(year, y, col, label, pos="above"):
        return (rf"\fill[{col}] ({X(year)},{y}) circle (2.6pt) node[{pos}=2pt, align=center, "
                rf"font=\normalsize, text=cTXT] {{{label}}};")

    # interior point lane
    s.append(branch(0.75, lanes["ipm"], X(1958), X(1966), "cIPM"))
    s.append(lane(lanes["ipm"], X(1966), X(2021), "cIPM"))
    s.append(mile(1968, lanes["ipm"], "cIPM", r"barrier method\\Fiacco, McCormick 1968"))
    s.append(mile(2006, lanes["ipm"], "cIPM", r"IPOPT\\W\"achter, Biegler 2006"))
    # Riccati branch out of the interior-point lane
    s.append(branch(lanes["ipm"], lanes["ric"], X(1990), X(1998), "cRIC"))
    s.append(lane(lanes["ric"], X(1998), X(2021), "cRIC"))
    s.append(mile(1998, lanes["ric"], "cRIC", r"Riccati interior point for MPC\\Rao, Wright, Rawlings 1998", "above"))
    s.append(mile(2023, lanes["ric"], "cRIC", r"FATROP\\2023", "above"))
    # SQP lane
    s.append(lane(lanes["sqp"], X(1958), X(2021), "cSQP"))
    s.append(mile(1977, lanes["sqp"], "cSQP", r"SQP\\Han 1977, Powell 1978", "below"))
    s.append(mile(2002, lanes["sqp"], "cSQP", r"filter methods\\Fletcher, Leyffer 2002", "below"))
    # penalty -> ALM -> PANOC lane
    s.append(branch(0.75, lanes["alm"], X(1958), X(1966), "cPAN"))
    s.append(lane(lanes["alm"], X(1966), X(2021), "cPAN"))
    s.append(mile(1969, lanes["alm"], "cPAN", r"augmented Lagrangian\\Hestenes 1969, Powell 1969", "below"))
    s.append(mile(1989, lanes["alm"], "cPAN", r"L-BFGS\\Liu, Nocedal 1989", "below"))
    s.append(mile(2017, lanes["alm"], "cPAN", r"PANOC\\Stella et al.\ 2017", "below"))
    s.append(mile(2021, lanes["alm"], "cPAN", r"OpEn 2020\\alpaqa 2022", "above"))
    # leaves: our four methods
    for key, code, name, col, kind in [("ric", "M4", "Riccati interior point", "cRIC", "modern"),
                                       ("ipm", "M1", "Interior point", "cIPM", "classical"),
                                       ("sqp", "M2", "SQP", "cSQP", "classical"),
                                       ("alm", "M3", "PANOC-ALM", "cPAN", "modern")]:
        y = lanes[key]
        s.append(rf"\draw[-{{Stealth[length=6pt]}}, line width=2.2pt, {col}] ({X(2021)},{y}) -- ({X(2026.5)},{y});")
        s.append(rf"\node[draw={col}, fill={col}, text=white, rounded corners=3pt, font=\small\bfseries, "
                 rf"minimum width=1.05cm, minimum height=0.62cm, right=1pt] (L{code}) at ({X(2026.5)},{y}) {{{code}}};")
        s.append(rf"\node[right=2pt of L{code}, align=left, font=\normalsize, text=cTXT] "
                 rf"{{\textbf{{{name}}}\\{{\color{{cOFF}}{kind}}}}};")
    s.append(r"\end{tikzpicture}")
    return "\n".join(s)


def pic_m2_paper():
    return (r"\begin{tikzpicture}[font=\footnotesize]" + row(M2_BOXES, "cSQP", io=False)
            + r"\end{tikzpicture}")


def pic_m3_paper(label="PANOC inner solver: gradients and a projection only"):
    """Paper: two rows, the dual update sits under the inner loop to keep the figure narrow."""
    c = "cPAN"
    return rf"""\begin{{tikzpicture}}[font=\footnotesize]
\node[stage={c}, text width=2.75cm] (o1)
  {{\hd{{{c}}}{{Outer: ALM subproblem}}\\[3pt]$\min_{{z\in C}}\ \psi(z)=\mathcal L_A(z;\lambda,\mu,\rho)$\\[2pt]
   {{\scriptsize $C=\{{l\le z\le u\}}$: 400 simple bounds,\\ handled by projection}}}};
\node[stage={c}, text width=3.0cm, right=9mm of o1] (i1)
  {{\hd{{{c}}}{{a\enspace Projected gradient}}\\[3pt]$\bar z=\Pi_C\big(z-\gamma\nabla\psi(z)\big)$\\ $r=z-\bar z$}};
\node[stage={c}, text width=2.15cm, right=4mm of i1] (i2)
  {{\hd{{{c}}}{{b\enspace L-BFGS}}\\[3pt]$d=-H_k\,r$\\[2pt]{{\scriptsize from pairs\\ $(\Delta z,\ \Delta r)$}}}};
\node[stage={c}, text width=3.3cm, right=4mm of i2] (i3)
  {{\hd{{{c}}}{{c\enspace Envelope line search}}\\[3pt]$z^+=z-(1-\tau)\,r+\tau\,d$\\
   $\varphi_\gamma(z^+)\le\varphi_\gamma(z)-\sigma\|r\|^2$\\[2pt]{{\scriptsize $\tau=1,\tfrac12,\dots,0$\\ ($\tau=0$ always passes)}}}};
\draw[flow={c}] (o1) -- (i1); \draw[flow={c}] (i1) -- (i2); \draw[flow={c}] (i2) -- (i3);
\draw[flow={c}] (i3.south) -- ++(0,-0.3) -| (i1.south) node[pos=0.25, below, note]
  {{inner loop until $\|r\|_\infty/\gamma\le\varepsilon_k$}};
\begin{{scope}}[on background layer]
\node[draw={c}, dashed, rounded corners=5pt, fit=(i1)(i2)(i3), inner xsep=4pt, inner ysep=19pt,
      yshift=-4pt, label={{[note, text={c}, yshift=-1pt]above:\textbf{{{label}}}}}] (box) {{}};
\end{{scope}}
\node[stage={c}, text width=5.6cm, minimum height=1.1cm, below=6mm of box.south east, anchor=north east] (o2)
  {{\hd{{{c}}}{{Outer: dual update}}\\[3pt]$\lambda\leftarrow\lambda+\rho\,h(z),\quad \mu\leftarrow\max\big(0,\ \mu+\rho\,g(z)\big)$\\[2pt]
   {{\scriptsize $\rho\leftarrow 5\rho$ if the violation does not drop;\enspace $\varepsilon_k\leftarrow\varepsilon_k/10$}}}};
\draw[flow={c}] (box.south -| o2.north) ++(0,0) -- (o2.north);
\draw[flow={c}] (o2.west) -| (o1.south) node[pos=0.25, above, note, align=center]
  {{outer loop until violation and\\ KKT residual $\le 10^{{-6}}$}};
\end{{tikzpicture}}"""


PAPER = {"pipeline_paper_m14": pic_m14, "pipeline_paper_m2": pic_m2_paper,
         "pipeline_paper_m3": pic_m3_paper}


SLIDE = {"pipeline_m1": pic_m1, "pipeline_m2": pic_m2_paper,
         "pipeline_m3": lambda: pic_m3_paper("PANOC inner solver: gradients and a projection only, "
                                             "no Hessian, no linear system"),
         "pipeline_m4": pic_m4, "timeline": pic_timeline}


def pic_symbols():
    """The planning problem in a picture: what each symbol of the NLP stands for (slides only).
    Left: the car at one time step (kinematic bicycle, rear axle).  Right: the plan over 5 s."""
    import math
    s = [r"\begin{tikzpicture}[font=\large]",
         r"\definecolor{cCAR}{HTML}{2E86DE}\definecolor{cOBS}{HTML}{FF7A3D}\definecolor{cRED}{HTML}{A02337}"]
    # ---------------- left panel: the car at step k
    th, d, L = math.radians(22), math.radians(26), 3.0
    c, sn = math.cos(th), math.sin(th)
    px, py = 0.0, 0.0
    s.append(r"\node[anchor=west, text=black!55, font=\Large\bfseries] at (-3.0,3.7) {the car at step $k$};")
    s.append(rf"\begin{{scope}}[rotate={math.degrees(th):.2f}]")
    s.append(r"\fill[cCAR, opacity=0.16, rounded corners=5pt] (-0.75,-0.9) rectangle (3.85,0.9);")
    s.append(r"\draw[cCAR, line width=0.9pt, rounded corners=5pt] (-0.75,-0.9) rectangle (3.85,0.9);")
    s.append(rf"\draw[cCAR!60!black, line width=0.9pt] (0,0) -- ({L},0);")
    s.append(r"\draw[black!80, line width=4pt] (-0.4,0) -- (0.4,0);")                                  # rear wheel
    s.append(rf"\draw[black!80, line width=4pt] ({L - 0.4 * math.cos(d):.3f},{-0.4 * math.sin(d):.3f}) -- "
             rf"({L + 0.4 * math.cos(d):.3f},{0.4 * math.sin(d):.3f});")                                 # front wheel
    s.append(rf"\draw[black!50, dashed] ({L},0) -- ({L + 1.9},0);")                                      # steering
    s.append(rf"\draw[black!50, dashed] ({L},0) -- ({L + 1.9 * math.cos(d):.3f},{1.9 * math.sin(d):.3f});")
    s.append(rf"\draw[cRED, line width=0.9pt] ({L + 1.5},0) arc (0:{math.degrees(d):.1f}:1.5) "
             r"node[pos=0.55, right=1pt, text=cRED] {$\delta$ steering};")
    s.append(rf"\draw[-{{Stealth[length=7pt]}}, cRED, line width=1.4pt] (0.3,1.35) -- (2.4,1.35) "
             r"node[above=1pt, text=cRED] {$v$ speed};")
    s.append(rf"\draw[black!60, {{Stealth[length=4pt]}}-{{Stealth[length=4pt]}}] (0,-1.3) -- ({L},-1.3) "
             r"node[midway, below=0pt, text=black!75, font=\large] {$L$ wheelbase};")
    s.append(r"\end{scope}")
    s.append(r"\draw[black!50, dashed] (0,0) -- (-2.0,0);")                                             # heading
    s.append(rf"\draw[black!50, dashed] (0,0) -- ({-2.0 * c:.3f},{-2.0 * sn:.3f});")
    s.append(rf"\draw[cRED, line width=0.9pt] (-1.7,0) arc (180:{180 + math.degrees(th):.1f}:1.7) "
             r"node[pos=0.5, left=1pt, text=cRED] {$\theta$ heading};")
    s.append(r"\fill[cRED] (0,0) circle (3pt);")
    s.append(r"\node[anchor=north, text=cRED, align=center] at (-0.1,-0.35) {};")
    s.append(r"\node[anchor=east, text=cRED] at (-0.85,0.75) {position $p_k=(p_x,p_y)$};")
    # ---------------- right panel: the plan over the horizon
    ox = 7.4
    yc, A, xo, wd = 1.0, 1.5, ox + 7.0, 2.4
    yo, ro = 0.7, 0.55
    R = yc + A - yo

    def traj(x):
        return yc + A * math.exp(-((x - xo) / wd) ** 2)

    s.append(rf"\node[anchor=west, text=black!55, font=\Large\bfseries] at ({ox},3.7) "
             r"{the plan: $N$ steps over 5\,s};")
    s.append(rf"\fill[black!5] ({ox},-1.3) rectangle ({ox + 13.9},3.25);")
    s.append(rf"\draw[black!35, line width=0.8pt] ({ox},-1.3) -- ({ox + 13.9},-1.3) ({ox},3.25) -- ({ox + 13.9},3.25);")
    s.append(rf"\draw[black!40, dashed] ({ox},{yc}) -- ({ox + 13.9},{yc});")
    s.append(rf"\node[anchor=north west, text=black!60, font=\large] at ({ox + 1.0},{yc - 0.08}) "
             r"{lane centre $y_{\rm ref}$};")
    s.append(rf"\fill[cOBS] ({xo},{yo}) circle ({ro});")
    s.append(rf"\draw[cOBS, dashed, line width=0.9pt] ({xo},{yo}) circle ({R});")
    s.append(rf"\fill[white] ({xo},{yo}) circle (1.6pt);")
    s.append(rf"\node[text=white, font=\bfseries] at ({xo},{yo - 0.3}) {{$o_j$}};")
    ang = 205
    ex, ey = xo + R * math.cos(math.radians(ang)), yo + R * math.sin(math.radians(ang))
    s.append(rf"\draw[cRED, line width=0.9pt] ({xo},{yo}) -- ({ex:.3f},{ey:.3f}) "
             r"node[pos=0.62, below=0pt, text=cRED] {$R_j$};")
    s.append(rf"\node[align=left, text=cRED, font=\large, anchor=west] at ({xo + R + 0.15},{yo - 0.25}) "
             r"{no collision:\\$\|p_k-o_j\|\ge R_j$ for all $k$};")
    xs = [ox + 0.45 + 0.45 * i for i in range(28)]
    for x in xs:
        s.append(rf"\fill[cCAR] ({x:.3f},{traj(x):.3f}) circle (2.1pt);")
    s.append(rf"\node[below=3pt, text=cCAR] at ({xs[0]:.3f},{traj(xs[0]):.3f}) {{$p_0$}};")
    s.append(rf"\node[above=3pt, text=cCAR] at ({xs[-1]:.3f},{traj(xs[-1]):.3f}) {{$p_N$}};")
    s.append(rf"\fill[cRED] ({xs[9]:.3f},{traj(xs[9]):.3f}) circle (3pt) node[above=3pt, text=cRED] {{$p_k$}};")
    s.append(rf"\draw[decorate, decoration={{brace, amplitude=3pt}}, black!60] "
             rf"({xs[2]:.3f},{yc + 0.2}) -- ({xs[3]:.3f},{yc + 0.2}) "
             r"node[midway, above=3pt, font=\large, text=black!75] {$\Delta t=0.1$\,s};")
    s.append(r"\end{tikzpicture}")
    return "\n".join(s)


def build(pics, font, png):
    with tempfile.TemporaryDirectory() as td:
        tex = Path(td) / "p.tex"
        tex.write_text(PREAMBLE.replace("%FONT%", font)
                       + "\n".join(f() for f in pics.values()) + "\n\\end{document}\n")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "p.tex"],
                           cwd=td, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:])
            raise SystemExit("pdflatex failed")
        doc = pymupdf.open(Path(td) / "p.pdf")
        for page, name in zip(doc, pics):
            w, h = page.rect.width / 72, page.rect.height / 72
            if png:
                page.get_pixmap(dpi=600, alpha=False).save(OUT / f"{name}.png")   # fallback
                (OUT / f"{name}.svg").write_text(page.get_svg_image(text_as_path=True))  # vector
            one = pymupdf.open()
            one.insert_pdf(doc, from_page=page.number, to_page=page.number)
            one.save(OUT / f"{name}.pdf")
            print(f"{name:22s} {w:.2f} x {h:.2f} in")


SLIDE["symbols"] = pic_symbols


if __name__ == "__main__":
    build(SLIDE, SANS, png=True)
    build(PAPER, SERIF, png=False)
