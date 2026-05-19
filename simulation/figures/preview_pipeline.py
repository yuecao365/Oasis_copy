import os
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

# Patch make_figures to also save a PNG preview of fig4
import importlib.util, sys
spec = importlib.util.spec_from_file_location("make_figures", "simulation/figures/make_figures.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Re-run just fig4 but save as PNG
original = mod.fig4_pipeline

def fig4_png():
    import os as _os
    # call the real function (saves SVG), then re-save as PNG
    original()
    # reload the SVG as a matplotlib figure isn't possible directly,
    # so we re-run the drawing logic
    pass

# Just call fig4_pipeline which saves SVG, then separately save PNG
# We'll duplicate the save in a patched version
OUT_DIR = "simulation/figures"

def save_png():
    mod.fig4_pipeline.__globals__["OUT_DIR"] = OUT_DIR
    # Monkey-patch plt.savefig temporarily
    real_savefig = plt.savefig
    def patched(path, **kw):
        real_savefig(path, **kw)
        png_path = path.replace(".svg", "_preview.png")
        real_savefig(png_path, dpi=150, bbox_inches="tight",
                     facecolor="white")
        print(f"Preview saved: {png_path}")
    plt.savefig = patched
    mod.fig4_pipeline()
    plt.savefig = real_savefig

save_png()
