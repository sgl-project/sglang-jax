sky exec sky-f73f-jiongxuan 'python3 - <<"PY"
from pathlib import Path
for path in sorted(Path("/home/gcpuser").glob("mimo_smoke_node*.log")):
    print(f"===== {path.name} =====")
    lines = path.read_text(errors="ignore").splitlines()
    for line in lines[-60:]:
        print(line)
PY'
