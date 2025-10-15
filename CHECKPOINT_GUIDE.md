# Checkpointing

Collection script auto-saves progress every 50 clips to prevent data loss.

## Usage

### Start fresh
```bash
python collect_data.py --use-targets
```
Checkpoints save to `data/checkpoints/checkpoint_2023-24_<timestamp>.csv`

### Resume from interruption
```bash
ls data/checkpoints/  # find latest checkpoint
python collect_data.py --use-targets --resume-from data/checkpoints/checkpoint_2023-24_*.csv
```

Skips already-collected clips and continues from where you left off.

## How it works

- Saves every 50 clips (~10 min intervals)
- Saves locally and to S3
- On resume: loads checkpoint, extracts collected (game_id, event_num) pairs, skips duplicates
- Lost progress: at most 49 clips

## Files

```
data/
├── checkpoints/
│   └── checkpoint_2023-24_*.csv    # Auto-saved
└── metadata/
    └── nba_fouls_*clips_*.csv      # Final CSV (when complete)
```

S3 backup: `s3://nba-foul-dataset-oh/checkpoints/`
