#!/bin/bash
#!/bin/bash
#SBATCH --job-name=backup
#SBATCH --partition=barnard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/backup_%j.out
# Pack backup tiers listed in backup_manifest.txt into one tar per tier, for download to the laptop.
# - usage: sbatch (or bash) scripts/backup_pack.sh [tier ...]   (default: core; tiers: core models inputs)
# - plain uncompressed tar: pure I/O, same bytes the scp download reads anyway
# - DT=1 runs it on a ZIH datamover node instead (dttar --blocking; often unavailable)
# - output: $WS/backup/<tier>_<date>.tar (+ .list of included paths)
set -euo pipefail
shopt -s nullglob

WS=/data/horse/ws/tibo990i-thesis_data
MANIFEST=$HOME/Thesis_Repo/scripts/backup_manifest.txt  # not via $0: sbatch runs a spooled copy
OUT=$WS/backup
DATE=$(date +%F)
mkdir -p "$OUT"

for tier in "${@:-core}"; do
    list=$OUT/${tier}_$DATE.list
    : > "$list"
    while read -r t glob; do
        [ "$t" = "$tier" ] || continue
        eval "matches=( $glob )"
        [ ${#matches[@]} -gt 0 ] || { echo "WARN: no match: $glob" >&2; continue; }
        printf '%s\n' "${matches[@]#/}" >> "$list"
    done < <(grep -v '^\s*#' "$MANIFEST")
    [ -s "$list" ] || { echo "ERROR: tier '$tier' is empty" >&2; exit 1; }

    echo "Packing $tier: $(wc -l < "$list") paths -> $OUT/${tier}_$DATE.tar"
    if [ "${DT:-0}" = 1 ]; then TAR=(dttar --blocking); else TAR=(tar); fi
    "${TAR[@]}" -cf "$OUT/${tier}_$DATE.tar" -C / -T "$list"
    ls -lh "$OUT/${tier}_$DATE.tar"
done
