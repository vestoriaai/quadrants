import pathlib, shutil, subprocess, sys, tempfile, zipfile
venv = pathlib.Path(sys.argv[1]); wheel = pathlib.Path(sys.argv[2])
sp = next((venv.glob("lib/python*/site-packages")))
with tempfile.TemporaryDirectory() as td:
    with zipfile.ZipFile(wheel) as z: z.extractall(td)
    src = pathlib.Path(td)
    for name in ("quadrants", "quadrants_pytest"): shutil.rmtree(sp / name, ignore_errors=True)
    for old in sp.glob("quadrants-*.dist-info"): shutil.rmtree(old)
    for entry in src.iterdir():
        dest = sp / entry.name
        if entry.is_dir(): shutil.copytree(entry, dest)
        else: shutil.copy2(entry, dest)
proc = subprocess.run([str(venv / "bin" / "python"), "-c", "import quadrants; print(quadrants.__version__)"], capture_output=True, text=True)
print(proc.stdout.strip()); assert proc.returncode == 0, proc.stderr
print("INSTALL_OK")
