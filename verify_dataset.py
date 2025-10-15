# verify_dataset.py
import os
import pandas as pd
import boto3
import requests
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

def verify_dataset(csv_path):
    """Verify dataset integrity and print statistics"""

    print("=" * 80)
    print("NBA FOUL DATASET VERIFICATION")
    print("=" * 80)

    # Load CSV (preserve game_id as string to keep leading zeros)
    df = pd.read_csv(csv_path, dtype={'game_id': str})
    print(f"\n✅ CSV loaded: {len(df)} rows\n")

    # Basic statistics
    print("DATASET STATISTICS:")
    print("-" * 80)
    print(f"Total frames: {len(df)}")
    print(f"Unique games: {df['game_id'].nunique()}")
    print(f"Unique clips (game+event): {df.groupby(['game_id', 'event_num']).ngroups}")
    print(f"Frames per clip: {df.groupby(['game_id', 'event_num']).size().value_counts().to_dict()}")

    # Check for missing data
    print(f"\nMissing label data (is_foul_frame): {df['is_foul_frame'].isna().sum()}")
    print(f"Missing S3 URLs: {df['s3_url'].isna().sum()}")

    # Verify URL structure
    print("\nURL STRUCTURE VERIFICATION:")
    print("-" * 80)

    # Sample 5 random rows and check URL pattern
    sample = df.sample(min(5, len(df)))
    all_urls_correct = True

    for _, row in sample.iterrows():
        expected_pattern = f"frames/{row['season']}/{row['game_id']}/{row['game_id']}_{row['event_num']}_frame_{row['frame_index']:03d}.jpg"
        actual_url = row['s3_url']

        if expected_pattern in actual_url:
            print(f"✅ {row['game_id']}_event{row['event_num']}_frame{row['frame_index']}: URL matches expected pattern")
        else:
            print(f"❌ {row['game_id']}_event{row['event_num']}_frame{row['frame_index']}: URL mismatch")
            print(f"   Expected: ...{expected_pattern}")
            print(f"   Got: {actual_url}")
            all_urls_correct = False

    # Test S3 accessibility
    print("\nS3 ACCESSIBILITY TEST:")
    print("-" * 80)

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )
    bucket = os.getenv('S3_BUCKET_NAME')

    # Test 3 random URLs
    test_sample = df.sample(min(3, len(df)))
    s3_accessible = 0

    for _, row in test_sample.iterrows():
        s3_key = f"frames/{row['season']}/{row['game_id']}/{row['game_id']}_{row['event_num']}_frame_{row['frame_index']:03d}.jpg"

        try:
            s3_client.head_object(Bucket=bucket, Key=s3_key)
            print(f"✅ {s3_key}: Exists in S3")
            s3_accessible += 1
        except:
            print(f"❌ {s3_key}: NOT FOUND in S3")

    # Test URL accessibility from web
    print("\nWEB ACCESSIBILITY TEST (public URLs):")
    print("-" * 80)

    web_accessible = 0
    for _, row in test_sample.iterrows():
        url = row['s3_url']
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {url.split('/')[-1]}: Publicly accessible")
                web_accessible += 1
            else:
                print(f"❌ {url.split('/')[-1]}: Status {response.status_code}")
        except Exception as e:
            print(f"❌ {url.split('/')[-1]}: {e}")

    # Data integrity checks
    print("\nDATA INTEGRITY CHECKS:")
    print("-" * 80)

    # Check frame indices are 0-29 for each clip
    frame_counts = df.groupby(['game_id', 'event_num'])['frame_index'].apply(list)

    bad_clips = []
    for (game_id, event_num), frames in frame_counts.items():
        expected_frames = list(range(30))
        if sorted(frames) != expected_frames:
            bad_clips.append((game_id, event_num, frames))

    if bad_clips:
        print(f"❌ Found {len(bad_clips)} clips with incorrect frame indices:")
        for game_id, event_num, frames in bad_clips[:3]:
            print(f"   Game {game_id}, Event {event_num}: {frames}")
    else:
        print(f"✅ All clips have correct frame indices (0-29)")

    # Check timestamps are sequential
    timestamp_issues = 0
    for (game_id, event_num), group in df.groupby(['game_id', 'event_num']):
        timestamps = group.sort_values('frame_index')['frame_timestamp_sec'].tolist()
        if timestamps != sorted(timestamps):
            timestamp_issues += 1

    if timestamp_issues:
        print(f"❌ Found {timestamp_issues} clips with non-sequential timestamps")
    else:
        print(f"✅ All clips have sequential timestamps")

    # Player data completeness
    fouler_missing = df['fouler_name'].isna().sum()
    fouled_missing = df['fouled_player_name'].isna().sum()

    print(f"\nPlayer data:")
    print(f"  - Fouler name missing: {fouler_missing} rows ({fouler_missing/len(df)*100:.1f}%)")
    print(f"  - Fouled player name missing: {fouled_missing} rows ({fouled_missing/len(df)*100:.1f}%)")

    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    checks = [
        ("CSV structure", len(df) > 0),
        ("URL patterns", all_urls_correct),
        ("S3 accessibility", s3_accessible == len(test_sample)),
        ("Web accessibility", web_accessible == len(test_sample)),
        ("Frame indices", len(bad_clips) == 0),
        ("Timestamp order", timestamp_issues == 0),
    ]

    passed = sum(1 for _, result in checks if result)
    print(f"\n{passed}/{len(checks)} checks passed\n")

    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")

    if passed == len(checks):
        print("\n🎉 Dataset is properly organized and all frames are accessible!")
    else:
        print("\n⚠️  Some issues found. Review the details above.")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Verify NBA foul dataset integrity')
    parser.add_argument('csv_path', help='Path to CSV metadata file')

    args = parser.parse_args()
    verify_dataset(args.csv_path)
