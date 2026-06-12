#!/usr/bin/env bash
# Determinism probe for the fraud-detection trainer.
#
# Runs the training script three times back to back, writes each
# run's metrics to a separate JSON file, and compares all three
# files byte for byte. Three runs (rather than two) drive the
# probability of a spurious "pass" from two unseeded runs
# coincidentally landing on the same metrics down to roughly 1 %.
# Exits 0 when all three files are identical (reproducible training),
# non-zero otherwise, and prints a diff for diagnosis.
set -u

TRAIN_PY="/root/code/fraud-detection/src/models/train.py"
REPORTS="/root/code/fraud-detection/reports"
METRICS_1="${REPORTS}/metrics_run_1.json"
METRICS_2="${REPORTS}/metrics_run_2.json"
METRICS_3="${REPORTS}/metrics_run_3.json"

mkdir -p "${REPORTS}"
rm -f "${METRICS_1}" "${METRICS_2}" "${METRICS_3}"

echo "=== running train.py (repro-run-1)"
METRICS_OUT="${METRICS_1}" MLFLOW_RUN_NAME="repro-run-1" python3 "${TRAIN_PY}" || {
  echo "FAIL: first run errored out — see stderr above."
  exit 2
}

echo "=== running train.py (repro-run-2)"
METRICS_OUT="${METRICS_2}" MLFLOW_RUN_NAME="repro-run-2" python3 "${TRAIN_PY}" || {
  echo "FAIL: second run errored out — see stderr above."
  exit 2
}

echo "=== running train.py (repro-run-3)"
METRICS_OUT="${METRICS_3}" MLFLOW_RUN_NAME="repro-run-3" python3 "${TRAIN_PY}" || {
  echo "FAIL: third run errored out — see stderr above."
  exit 2
}

if diff -q "${METRICS_1}" "${METRICS_2}" >/dev/null \
  && diff -q "${METRICS_2}" "${METRICS_3}" >/dev/null; then
  echo "OK: all three runs produced byte-identical metrics."
  exit 0
fi

echo "FAIL: the three runs did not produce byte-identical metrics."
echo
echo "--- diff ${METRICS_1} ${METRICS_2} ---"
diff "${METRICS_1}" "${METRICS_2}" || true
echo "--- diff ${METRICS_2} ${METRICS_3} ---"
diff "${METRICS_2}" "${METRICS_3}" || true
echo "-------------------------------------"
exit 1