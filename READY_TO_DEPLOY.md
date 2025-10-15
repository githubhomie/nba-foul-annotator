# Deployment Instructions

## What Changed

- Annotations now save to S3 at `s3://nba-foul-dataset-oh/annotations/`
- Multiple users can annotate simultaneously
- Git repo initialized

## Deploy to Streamlit Cloud

### 1. Push to GitHub
```bash
# Create repo at github.com/new (make it public)
git remote add origin https://github.com/YOUR_USERNAME/nba-foul-annotator.git
git push -u origin main
```

### 2. Deploy App
1. Go to share.streamlit.io
2. Sign in, click "New app"
3. Select your repo
4. Set main file: `annotation_tool/app.py`
5. Add secrets (click Advanced settings):
```toml
AWS_ACCESS_KEY_ID = "your-access-key"
AWS_SECRET_ACCESS_KEY = "your-secret-key"
AWS_REGION = "us-east-2"
S3_BUCKET = "nba-foul-dataset-oh"
```
6. Click Deploy

### 3. Share with Team

Send them:
- App URL
- CSV file (upload to Google Drive or similar)
- Category assignments:
  - Person A: shooting_foul (300 clips)
  - Person B: charging (250 clips)
  - Person C: personal_foul (250 clips)
  - Person D: loose_ball (250 clips)
  - Person E: offensive_foul (150 clips)

Instructions: Upload CSV, enter name, select assigned category, uncheck "Show annotated clips", start annotating.

## How It Works

- Each person filters to their category
- Annotations save to S3
- Updates appear for everyone on next clip (~1-2 sec)
- If two people annotate same clip, last save wins

## Test Locally First

```bash
cd annotation_tool
pip install -r requirements.txt
streamlit run app.py
```
