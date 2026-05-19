from matplotlib import font_manager
matches = [f.name for f in font_manager.fontManager.ttflist if "Calibri" in f.name]
print("Found:", matches[:5] if matches else "NONE")
