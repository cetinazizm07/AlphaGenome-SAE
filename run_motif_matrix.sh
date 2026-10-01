#!/usr/bin/env bash
set -e
export PYTHONPATH=/mnt/ag/analysis-code-a1f31ce
export PYTHONUNBUFFERED=1

mkdir -p /mnt/ag/analysis_runs/motif_matrix

echo "================================================================================" >> /mnt/ag/analysis_runs/motif_matrix/master_campaign.log
echo "Starting Master Motif Matrix Campaign at $(date)" >> /mnt/ag/analysis_runs/motif_matrix/master_campaign.log
echo "================================================================================" >> /mnt/ag/analysis_runs/motif_matrix/master_campaign.log

/mnt/ag/venv/bin/python3 -u -m ag_sae.motif_intervene \
  --campaign --all-folds \
  >> /mnt/ag/analysis_runs/motif_matrix/master_campaign.log 2>&1

echo "================================================================================" >> /mnt/ag/analysis_runs/motif_matrix/master_campaign.log
echo "Master Motif Matrix Campaign finished at $(date)" >> /mnt/ag/analysis_runs/motif_matrix/master_campaign.log
echo "================================================================================" >> /mnt/ag/analysis_runs/motif_matrix/master_campaign.log
