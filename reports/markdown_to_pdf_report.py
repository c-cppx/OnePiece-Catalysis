from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "docs" / "reports" / "catalysis_hub_master_report"
MD_PATH = REPORT_DIR / "catalysis_hub_master_report.md"
TEX_PATH = REPORT_DIR / "catalysis_hub_master_report.tex"
PDF_PATH = REPORT_DIR / "catalysis_hub_master_report.pdf"


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def inline_format(text: str) -> str:
    placeholders: list[str] = []

    def hold(pattern: str, source: str, wrapper):
        def repl(match):
            placeholders.append(wrapper(match.group(1)))
            return f"@@PLACEHOLDER{len(placeholders)-1}@@"
        return re.sub(pattern, repl, source)

    text = hold(r"`([^`]+)`", text, lambda s: r"\texttt{" + escape_latex(s) + "}")
    text = hold(r"\*\*([^*]+)\*\*", text, lambda s: r"\textbf{" + escape_latex(s) + "}")
    text = escape_latex(text)
    for idx, replacement in enumerate(placeholders):
        text = text.replace(escape_latex(f"@@PLACEHOLDER{idx}@@"), replacement)
    text = text.replace("“", "``").replace("”", "''")
    return text


def render_table(lines: list[str]) -> list[str]:
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append(cells)
    if len(rows) < 2:
        return [inline_format(" ".join(lines))]
    header = rows[0]
    body = [row for row in rows[2:] if any(cell.strip("- ") for cell in row)]
    ncols = len(header)
    colspec = " | ".join(["p{0.15\\textwidth}"] + ["p{0.13\\textwidth}"] * (ncols - 1))
    out = [
        r"\begin{center}",
        r"\small",
        rf"\begin{{tabular}}{{| {colspec} |}}",
        r"\hline",
        " & ".join(inline_format(cell) for cell in header) + r" \\",
        r"\hline",
    ]
    for row in body:
        padded = row + [""] * (ncols - len(row))
        out.append(" & ".join(inline_format(cell) for cell in padded[:ncols]) + r" \\")
        out.append(r"\hline")
    out.extend([r"\end{tabular}", r"\normalsize", r"\end{center}"])
    return out


def convert_markdown(md: str) -> str:
    lines = md.splitlines()
    out = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=2.2cm]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{array}",
        r"\usepackage{longtable}",
        r"\usepackage{booktabs}",
        r"\usepackage{fontspec}",
        r"\usepackage{hyperref}",
        r"\usepackage{parskip}",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
    ]

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("# "):
            out.append(r"\title{" + inline_format(stripped[2:]) + "}")
            out.append(r"\maketitle")
            i += 1
            continue
        if stripped.startswith("## "):
            out.append(r"\section{" + inline_format(stripped[3:]) + "}")
            i += 1
            continue
        if stripped.startswith("### "):
            out.append(r"\subsection{" + inline_format(stripped[4:]) + "}")
            i += 1
            continue
        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            alt, path = image_match.groups()
            out.extend(
                [
                    r"\begin{figure}[h]",
                    r"\centering",
                    rf"\includegraphics[width=0.92\linewidth]{{{path}}}",
                    rf"\caption*{{{inline_format(alt)}}}",
                    r"\end{figure}",
                ]
            )
            i += 1
            continue
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.extend(render_table(table_lines))
            continue
        if re.match(r"- ", stripped):
            out.append(r"\begin{itemize}")
            while i < len(lines) and re.match(r"- ", lines[i].strip()):
                out.append(r"\item " + inline_format(lines[i].strip()[2:]))
                i += 1
            out.append(r"\end{itemize}")
            continue
        if re.match(r"\d+\. ", stripped):
            out.append(r"\begin{enumerate}")
            while i < len(lines) and re.match(r"\d+\. ", lines[i].strip()):
                item = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                out.append(r"\item " + inline_format(item))
                i += 1
            out.append(r"\end{enumerate}")
            continue

        paragraph = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith("|")
                or nxt.startswith("![")
                or re.match(r"- ", nxt)
                or re.match(r"\d+\. ", nxt)
            ):
                break
            paragraph.append(nxt)
            i += 1
        out.append(inline_format(" ".join(paragraph)))

    out.append(r"\end{document}")
    return "\n".join(out) + "\n"


def main() -> None:
    md = MD_PATH.read_text(encoding="utf-8")
    tex = convert_markdown(md)
    TEX_PATH.write_text(tex, encoding="utf-8")
    for _ in range(2):
        subprocess.run(
            [
                "/opt/local/bin/xelatex",
                "-interaction=nonstopmode",
                "-output-directory",
                str(REPORT_DIR),
                str(TEX_PATH),
            ],
            check=True,
            cwd=ROOT,
        )
    print(PDF_PATH)


if __name__ == "__main__":
    main()
