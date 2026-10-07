#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
source gpu-env.sh
export PYTHONPATH="$root/third_party/jax-md-remd:$root/mace-jax${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${2:-0}
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=2
mode=${1:?Use polar, mlmm or mechanical}
case "$mode" in polar|mlmm|mechanical) ;; *) exit 2 ;; esac
base=Examples/HydrogenMaleate-REMD-CSVR-20261006
mkdir -p "$base"
exec 9>"$base/$mode.lock"
flock -n 9 || { printf 'This REMD mode is already running\n' >&2; exit 2; }
printf '%s\n' "$$" >"$base/$mode.pid"
initial=Examples/HydrogenMaleate-FIRES-25W/structure/initial-10ps-25w.xyz
charges=Examples/HydrogenMaleate-FIRES-25W/structure/fixed-charges-25w.json
driver=mace-jax/examples/HydrogenMaleate-MACE-POLAR/remd.py
extra=()
if [[ "$mode" == mechanical ]]; then extra+=(--mechanical-qm-charges "$charges"); fi
run() {
  local steps=$1
  local directory=$2
  shift 2
  venv/bin/python -u "$driver" --initial "$initial" \
    --bundle artifacts/cueq-benchmark/MACE-POLAR-1-M-jax-fused.msgpack \
    --output "$directory" --mode "$mode" --steps "$steps" --require-gpu \
    --temperatures "$@" "${extra[@]}"
}
# The pilot is the start of the production trajectory; resume from its exact state.
# Judge adjacent exchanges after discarding the first 0.2 ps (400 MD steps).
output="$base/$mode-6rep"
run 2000 "$output" 300 320 340 360 380 400
if venv/bin/python - "$output" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1])
rows=[json.loads(s) for s in (root/'progress.jsonl').read_text().splitlines()]
last=rows[-1]
start=next(r for r in rows if r['step']==400)
counts=np.array(last['swap_accepts'])-np.array(start['swap_accepts'])
n=len(counts)
acceptance=counts[np.arange(n-1),np.arange(1,n)]/((last['step']-400)/10)
decision=dict(adjacent_acceptance_after_0_2ps=acceptance.tolist(),
              threshold=0.10, keep_six=bool(np.all(acceptance>=.10)))
(root/'ladder-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
print(json.dumps(decision),flush=True)
sys.exit(0 if decision['keep_six'] else 1)
PY
then
  run 200000 "$output" 300 320 340 360 380 400
else
  output="$base/$mode-11rep"
  run 2000 "$output" 300 310 320 330 340 350 360 370 380 390 400
  # Stop for inspection if the denser ladder still has an unaccepted adjacent pair.
  venv/bin/python - "$output" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); rows=[json.loads(s) for s in (root/'progress.jsonl').read_text().splitlines()]
a=next(r for r in rows if r['step']==400); b=rows[-1]
c=np.array(b['swap_accepts'])-np.array(a['swap_accepts']); n=len(c)
p=c[np.arange(n-1),np.arange(1,n)]/((b['step']-400)/10)
d=dict(adjacent_acceptance_after_0_2ps=p.tolist(), threshold=.10, keep_eleven=bool(np.all(p>=.10)))
(root/'ladder-decision.json').write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(d),flush=True)
if not d['keep_eleven']: raise SystemExit('Eleven-temperature exchange is poor; stopping at 1 ps for inspection')
PY
  run 200000 "$output" 300 310 320 330 340 350 360 370 380 390 400
fi
printf '%s\n' "$output" >"$base/$mode-production-path.txt"
