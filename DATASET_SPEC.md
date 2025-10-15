# Dataset Specification

## Overview

- 1,213 clips collected (target: 1,200)
- 5 foul categories from NBA API
- 30 frames per clip
- Source: NBA API 2023-24 season

## Categories

| Category | API Code | Count | Description |
|----------|----------|-------|-------------|
| shooting_foul | 2 | 300 | Contact during shooting motion |
| personal_foul | 1 | 250 | Defensive fouls (blocking, reach-ins, etc.) |
| loose_ball | 3 | 250 | Contact during rebounds/scrambles |
| charging | 26 | 250 | Offensive player into set defender |
| offensive_foul | 4 | 150 | Other offensive fouls (screens, push-offs) |

## CSV Schema

```
game_id, event_num, frame_index, frame_timestamp_sec, season, period,
game_clock, description, action_type, foul_type, fouler_id, fouler_name,
fouler_team, fouled_player_id, fouled_player_name, fouled_team, s3_url,
is_foul_frame
```

Key fields:
- `action_type` (int): NBA API code (1, 2, 3, 4, 26)
- `foul_type` (string): Our category name
- `frame_index` (int): 0-29 within clip
- `s3_url`: Link to frame image
- `is_foul_frame` (bool): To be annotated

## S3 Structure

```
s3://nba-foul-dataset-oh/
├── frames/2023-24/{game_id}/{game_id}_{event_num}_frame_{000-029}.jpg
├── metadata/nba_fouls_multi-season_1213clips_*.csv
├── annotations/{game_id}_{event_num}_annotation.json
└── checkpoints/checkpoint_*.csv
```

## Annotation

Task: Mark which frame shows foul contact

1. View 30 frames
2. Select frame where contact occurs
3. Save to S3

Time: ~15 sec/clip, 5 hours total for 1,213 clips

## Collection

```bash
# Fresh collection
python collect_data.py --use-targets

# Resume from checkpoint
python collect_data.py --use-targets --resume-from data/checkpoints/checkpoint_*.csv
```

Checkpoints auto-save every 50 clips.

## Training Splits

- Train: 70% (~850 clips)
- Val: 15% (~180 clips)
- Test: 15% (~180 clips)

Stratify by `foul_type` to maintain category balance.
