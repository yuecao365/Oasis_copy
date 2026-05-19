import os, sys
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

real_save = plt.savefig

def patched_save(path, **kw):
    real_save(path, **kw)
    if path.endswith(".svg"):
        png = path.replace(".svg", "_preview.png")
        real_save(png, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  preview -> {png}")

plt.savefig = patched_save

import importlib.util
spec = importlib.util.spec_from_file_location("mf", "simulation/figures/make_figures.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.fig1_improvement()
mod.fig3_agent_roles()
mod.fig4_pipeline()
mod.fig5_combined()
