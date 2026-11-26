#!/usr/bin/env bash
# Submits one payload through the GitHub relay, waits for the result and writes RES_DIR/NAME.json.
# Usage: RELAY_ENV=/path/to/relay_env.sh bash go.sh NAME payload.json [result_timeout_s]
# RELAY_ENV is a file that defines GIT_PAT, GIT_REPO, GIT_BRANCH and SK (the folder with the relay scripts
# submit_job.sh, poll_submission.sh and poll_result.sh of the cluster skill). Never commit that file.
# If the wait ends before the job, fetch the result later: bash "$SK/poll_result.sh" NAME 30 3 > RES_DIR/NAME.json
set -u
source "${RELAY_ENV:-$HOME/relay_env.sh}"
RES="${RES_DIR:-./results}"; mkdir -p "$RES"
N="$1"; P="$2"
bash "$SK/submit_job.sh" "$P" "$N" > "$RES/$N.submit.log" 2>&1 || { echo "SUBMIT FAILED"; sed "s/$GIT_PAT/<hidden>/g" "$RES/$N.submit.log" | tail -n 3; exit 1; }
bash "$SK/poll_submission.sh" "$N" 300 3 > "$RES/$N.sub.json" 2>&1 || { echo "NO SUBMISSION"; exit 1; }
cat "$RES/$N.sub.json"
bash "$SK/poll_result.sh" "$N" "${3:-1200}" 3 > "$RES/$N.json" 2> "$RES/$N.err" || { echo "NO RESULT"; exit 1; }
echo done
