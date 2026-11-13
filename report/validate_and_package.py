"""Validate LaTeX report syntax, verify all asset links, and package into report-upload.zip."""
import os
import re
import sys
import zipfile
from pathlib import Path

REPORT_DIR = Path(__file__).parent.resolve()
TEX_FILE = REPORT_DIR / "main.tex"
BIB_FILE = REPORT_DIR / "references.bib"
ZIP_FILE = REPORT_DIR / "report-upload.zip"

def validate_latex():
    print(f"--> Validating {TEX_FILE.name}...")
    content = TEX_FILE.read_text(encoding="utf-8")
    
    # 1. Check balanced braces
    open_braces = content.count("{")
    close_braces = content.count("}")
    if open_braces != close_braces:
        raise ValueError(f"Unbalanced braces in {TEX_FILE.name}: {open_braces} open vs {close_braces} close!")
    print(f"    [OK] Balanced curly braces ({open_braces} pairs).")

    # 2. Check bibliography keys
    bib_content = BIB_FILE.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_content))
    citations = set(re.findall(r"\\cite\{([^}]+)\}", content))
    flat_citations = set()
    for c in citations:
        for k in c.split(","):
            flat_citations.add(k.strip())
    
    missing_keys = flat_citations - bib_keys
    if missing_keys:
        print(f"    [WARNING] Citations missing in references.bib: {missing_keys}")
    else:
        print(f"    [OK] All {len(flat_citations)} citation keys verified in references.bib.")

    # 3. Check graphic inclusions
    graphics = re.findall(r"\\safeincludegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content)
    graphics += re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content)
    graphics = [g for g in graphics if '#' not in g]
    missing_figs = []
    found_figs = []
    for g in graphics:
        p1 = REPORT_DIR / g
        p2 = REPORT_DIR / f"{g}.pdf"
        p3 = REPORT_DIR / f"{g}.png"
        if p1.exists() or p2.exists() or p3.exists():
            found_figs.append(g)
        else:
            missing_figs.append(g)
    
    if missing_figs:
        print(f"    [WARNING] Missing graphic files: {missing_figs}")
    else:
        print(f"    [OK] All {len(found_figs)} referenced graphics found on disk.")

    # 4. Check label duplicates
    labels = re.findall(r"\\label\{([^}]+)\}", content)
    dup_labels = [l for l in labels if labels.count(l) > 1]
    if dup_labels:
        raise ValueError(f"Duplicate labels found: {set(dup_labels)}")
    print(f"    [OK] All {len(labels)} labels are unique.")


def package_report():
    print(f"--> Packaging report into {ZIP_FILE.name}...")
    with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(REPORT_DIR):
            for file in files:
                p = Path(root) / file
                if p.name == "report-upload.zip" or p.suffix in (".pyc", ".aux", ".log", ".out"):
                    continue
                rel = p.relative_to(REPORT_DIR)
                z.write(p, rel.as_posix())
    size_mb = ZIP_FILE.stat().st_size / (1024 * 1024)
    print(f"    [SUCCESS] Created {ZIP_FILE.name} ({size_mb:.2f} MB). Ready for Overleaf / LMS upload.")


if __name__ == "__main__":
    validate_latex()
    package_report()
