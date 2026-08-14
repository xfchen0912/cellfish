"""Load optional TTF fonts. ``fonttools`` is only required for signature checks."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def font_signature(ttf_path: Path) -> tuple:
    """Return ``(family, subfamily, weight, is_italic)`` for a TTF file."""
    from fontTools.ttLib import TTFont

    font = TTFont(str(ttf_path))
    family = ""
    subfamily = ""
    for rec in font["name"].names:
        if rec.nameID == 1 and not family:
            family = rec.toUnicode()
        if rec.nameID == 2 and not subfamily:
            subfamily = rec.toUnicode()
    weight = font["OS/2"].usWeightClass if "OS/2" in font else 400
    is_italic = bool(font["post"].italicAngle) if "post" in font else False
    return (family, subfamily, weight, is_italic)


def validate_and_load_fonts(font_list: list, font_dir: str = "~/tmp/fonts/") -> list:
    """Validate and load custom fonts from ``font_dir``."""
    font_path = Path(os.path.expanduser(font_dir))
    valid_fonts: set[str] = set()
    loaded_signatures: set[tuple] = set()
    if not font_path.exists():
        return []

    for f in fm.fontManager.ttflist:
        try:
            loaded_signatures.add(font_signature(Path(f.fname)))
        except Exception:
            pass

    for base_font in font_list:
        for font_file in font_path.glob(f"{base_font}*.ttf"):
            try:
                sig = font_signature(font_file)
                if sig[0].startswith(base_font) and sig not in loaded_signatures:
                    fm.fontManager.addfont(str(font_file))
                    loaded_signatures.add(sig)
                    valid_fonts.add(" ".join(filter(None, sig[:2])))
            except Exception as exc:
                print(f"Failed to load font {font_file}: {exc}")
    return sorted(valid_fonts)


def export_mplstyle(filename: str = "default.mplstyle") -> None:
    """Write current matplotlib rcParams that differ from defaults."""
    defaults = mpl.rcParamsDefault
    current = plt.rcParams
    ignore_keys = (
        "backend",
        "backend_fallback",
        "interactive",
        "toolbar",
        "axes.prop_cycle",
        "timezone",
        "figure.figsize",
        "figure.dpi",
    )
    with open(filename, "w") as f:
        f.write("# Custom Matplotlib Style - Exported\n")
        for key, value in current.items():
            if any(x in key for x in ignore_keys):
                continue
            if str(value) != str(defaults.get(key)):
                f.write(f"{key}: {value}\n")
