"""BUG-0022: drvarma's Windows wheels carry no compiled engine.

Downloads the win_amd64 wheel of the latest drvarma on PyPI (cp312) and looks
for the compiled extension (drvarma/_drvarma_engine*.pyd). Exits 1 while the
published Windows wheel has none.

    python3 bugs/BUG-0022-repro/repro.py
"""
import glob, subprocess, sys, tempfile, zipfile

with tempfile.TemporaryDirectory() as d:
    subprocess.run([sys.executable, "-m", "pip", "download", "-q", "--no-deps",
                    "--only-binary", ":all:", "--platform", "win_amd64",
                    "--python-version", "3.12", "drvarma", "-d", d], check=True)
    whl = glob.glob(f"{d}/drvarma-*-win_amd64.whl")[0]
    eng = [n for n in zipfile.ZipFile(whl).namelist()
           if n.startswith("drvarma/_drvarma_engine") and n.endswith(".pyd")]
print(whl.rsplit("/", 1)[-1], "->", eng or "NO compiled engine")
sys.exit(0 if eng else 1)
