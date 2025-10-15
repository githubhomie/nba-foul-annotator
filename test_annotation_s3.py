#!/usr/bin/env python3
"""
Quick test script to verify annotation S3 integration works
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'annotation_tool'))

from utils.annotation_io import save_annotation, load_annotation, get_annotated_clips
from dotenv import load_dotenv

load_dotenv()

def test_s3_annotation():
    """Test S3 read/write for annotations"""

    print("🏀 Testing NBA Annotation S3 Integration\n")

    # Test data
    test_game_id = "TEST_GAME_999"
    test_event_num = 999
    test_frame = 15
    test_annotator = "test_user"

    # Test 1: Save annotation
    print("1. Testing save_annotation to S3...")
    try:
        result = save_annotation(
            test_game_id,
            test_event_num,
            test_frame,
            annotator=test_annotator,
            notes="Test annotation"
        )
        print(f"   ✅ Saved to: {result}\n")
    except Exception as e:
        print(f"   ❌ Failed to save: {e}\n")
        return False

    # Test 2: Load annotation
    print("2. Testing load_annotation from S3...")
    try:
        annotation = load_annotation(test_game_id, test_event_num)
        if annotation:
            print(f"   ✅ Loaded annotation:")
            print(f"      - Game ID: {annotation['game_id']}")
            print(f"      - Event: {annotation['event_num']}")
            print(f"      - Frame: {annotation['foul_frame']}")
            print(f"      - Annotator: {annotation['annotator']}\n")
        else:
            print("   ❌ No annotation found\n")
            return False
    except Exception as e:
        print(f"   ❌ Failed to load: {e}\n")
        return False

    # Test 3: List annotations
    print("3. Testing get_annotated_clips from S3...")
    try:
        clips = get_annotated_clips()
        print(f"   ✅ Found {len(clips)} annotated clips")

        # Show a few examples
        examples = list(clips)[:5]
        for game_id, event_num in examples:
            print(f"      - {game_id}_{event_num}")
        if len(clips) > 5:
            print(f"      ... and {len(clips) - 5} more\n")
        else:
            print()
    except Exception as e:
        print(f"   ❌ Failed to list: {e}\n")
        return False

    print("✅ All tests passed! S3 annotation system is working.\n")
    print("Next steps:")
    print("1. Push code to GitHub (see GITHUB_SETUP.md)")
    print("2. Deploy to Streamlit Cloud (see DEPLOYMENT.md)")
    print("3. Share URL with your team!\n")

    return True

if __name__ == "__main__":
    success = test_s3_annotation()
    sys.exit(0 if success else 1)
