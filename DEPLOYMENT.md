# Deployment Options

## Streamlit Cloud (Recommended)

**Free, easiest option**

1. Push code to GitHub (public repo)
2. Go to share.streamlit.io
3. Connect repo, set main file to `annotation_tool/app.py`
4. Add AWS credentials in secrets
5. Deploy

Updates automatically when you push to GitHub.

## AWS App Runner

**Better performance, costs ~$25/month**

```bash
# Create Dockerfile
cd annotation_tool
cat > Dockerfile << 'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
EOF

# Build and deploy via AWS console
```

## AWS EC2

**Cheapest, requires management, ~$10/month**

```bash
# SSH into t3.micro instance
ssh -i key.pem ubuntu@instance-ip

# Setup
sudo apt update && sudo apt install -y python3-pip git
git clone https://github.com/your-username/nba-foul-annotator.git
cd nba-foul-annotator/annotation_tool
pip3 install -r requirements.txt

# Run
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

Access at `http://instance-ip:8501`

## Multi-User Coordination

Split by category to avoid collisions:
- Person A: shooting_foul
- Person B: charging
- Person C: personal_foul
- Person D: loose_ball
- Person E: offensive_foul

Everyone unchecks "Show annotated clips" to hide completed work.
