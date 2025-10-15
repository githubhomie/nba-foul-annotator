# test_aws_connection.py
import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔍 Testing AWS Connection...\n")

# Check if credentials exist
if not os.getenv('AWS_ACCESS_KEY_ID'):
    print("❌ Error: AWS_ACCESS_KEY_ID not found in .env file")
    exit(1)

print("✅ Environment variables loaded")
print(f"   Region: {os.getenv('AWS_REGION')}")
print(f"   Bucket: {os.getenv('S3_BUCKET_NAME')}\n")

# Create S3 client
try:
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )
    print("✅ S3 client created\n")
except Exception as e:
    print(f"❌ Error creating S3 client: {e}")
    exit(1)

# Test bucket access
bucket = os.getenv('S3_BUCKET_NAME')
try:
    s3.head_bucket(Bucket=bucket)
    print(f"✅ Successfully accessed bucket: {bucket}\n")
except Exception as e:
    print(f"❌ Error accessing bucket: {e}")
    exit(1)

# Test upload
print("📤 Testing file upload...")
test_file = 'test_upload.txt'
try:
    # Create test file
    with open(test_file, 'w') as f:
        f.write('Hello from NBA foul collector!')

    # Upload to S3
    s3.upload_file(test_file, bucket, 'test/test_upload.txt')
    print("✅ File uploaded successfully\n")

    # Generate public URL
    url = f"https://{bucket}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/test/test_upload.txt"
    print(f"🌐 Public URL: {url}")
    print("   (Open this URL in browser to verify)\n")

    # Clean up
    s3.delete_object(Bucket=bucket, Key='test/test_upload.txt')
    os.remove(test_file)
    print("✅ Test file cleaned up")

except Exception as e:
    print(f"❌ Upload test failed: {e}")
    if os.path.exists(test_file):
        os.remove(test_file)
    exit(1)

print("\n" + "="*50)
print("🎉 ALL TESTS PASSED!")
print("="*50)
print("You're ready to collect NBA data!")
