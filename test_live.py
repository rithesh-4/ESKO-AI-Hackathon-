import os, glob, time
import dotenv
dotenv.load_dotenv()
print("llm key present:", bool(os.getenv("LLM_API_KEY")))
print("base:", os.getenv("LLM_BASE_URL"))
print("model:", os.getenv("LLM_MODEL"))

from core.llm import extract_with_llm
from core.ingest import ingest
from core.gaps import detect_conflicts
from core.schema import region_hint, completeness

chunks = []
for p in sorted(glob.glob("Samples by Esko/*")):
    if os.path.isdir(p):
        continue
    chunks += ingest(os.path.basename(p), open(p, "rb").read())
print("chunks:", len(chunks))

t0 = time.time()
state, backend = extract_with_llm(chunks)
dt = time.time() - t0
print(f"BACKEND={backend} took={dt:.1f}s")

issues = detect_conflicts(chunks, state)
print(f"region={region_hint(state)} completeness={completeness(state)}% issues={len(issues)}")
for i in issues:
    print(f"  [{i.kind}] {i.field}: {i.message[:110]}")
print("fields with values:", sum(1 for v in state.values() if v.value))
for k, pf in state.items():
    if pf.value:
        print(f"  {k}={pf.status}/{pf.confidence} :: {pf.value[:90]!r}")
