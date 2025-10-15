# Project Structure

```
foul_data_aws/
├── data/
│   ├── metadata/          # CSV files from collection
│   ├── annotations/       # Local annotation backups
│   ├── checkpoints/       # Collection checkpoints
│   ├── temp_videos/       # Temp (auto-cleaned)
│   └── temp_frames/       # Temp (auto-cleaned)
│
├── annotation_tool/
│   ├── app.py            # Streamlit app
│   ├── utils/
│   │   ├── s3_loader.py      # Load frames from S3
│   │   └── annotation_io.py  # Save/load annotations
│   └── requirements.txt
│
├── collect_data.py       # NBA data collection
├── verify_dataset.py     # Dataset verification
└── requirements.txt
```

## Key Files

- `collect_data.py` - Collects foul clips from NBA API
- `annotation_tool/app.py` - Web interface for annotating clips
- `annotation_tool/utils/annotation_io.py` - Handles S3 storage
- `data/metadata/` - CSV files with frame metadata
- S3: `s3://nba-foul-dataset-oh/annotations/` - Shared annotations

## Running Locally

```bash
# Annotation tool
cd annotation_tool
pip install -r requirements.txt
streamlit run app.py

# Data collection
pip install -r requirements.txt
python collect_data.py --use-targets
```
