# AI Music Generator - Implementation Plan
## Based on Actual Build Process

---

## Project Overview

This plan recreates the AI Music Generation SaaS application following the proven implementation path. The system allows users to generate original music using AI models from text descriptions, custom lyrics, or style prompts.

### Technology Stack

**Backend:**
- Python/FastAPI on Modal (serverless GPU)
- ACE-Step (music generation model)
- Qwen2-7B-Instruct (LLM for prompt/lyric generation)
- SDXL-Turbo (thumbnail generation)
- AWS S3 (storage)

**Frontend:**
- Next.js 15 with App Router
- React, TypeScript
- Tailwind CSS, Shadcn UI
- Drizzle ORM
- BetterAuth (authentication)

**Infrastructure:**
- Inngest (queue system)
- Neon PostgreSQL (database)
- Vercel (deployment)
- Polar.sh (payments)

---

## PHASE 1: Backend Setup with Modal

### STEP 1.1: Environment & Modal Setup

#### Micro-steps:
1. Create project directory: `ai-music-generator`
2. Create backend folder
3. Install Python 3.12 and create virtual environment
4. Install Modal CLI: `pip install modal`
5. Authenticate Modal: `modal setup`
6. Clone ACE-Step as submodule: `git clone --recurse-submodules https://github.com/ace-step/ACE-Step.git backend/ACE-Step`
7. Create requirements.txt
8. Initialize git repository

#### Tests:
```bash
# Verify Python version
python --version
# Expected: Python 3.12.x

# Verify Modal CLI
modal --version
# Expected: modal, version X.X.X

# Verify Modal authentication
modal token list
# Expected: Shows your active token

# Verify ACE-Step submodule
ls backend/ACE-Step
# Expected: Shows ACE-Step repository files

# Test directory structure
tree -L 2 backend/
# Expected:
# backend/
# ├── ACE-Step/
# ├── main.py
# ├── prompts.py
# └── requirements.txt
```

---

### STEP 1.2: AWS S3 Storage Setup

#### Micro-steps:
1. Create AWS account
2. Navigate to S3 console
3. Create bucket (e.g., `music-gen-bucket-yourname`)
4. Choose region closest to users
5. Keep default settings (Block public access: ON)
6. Create IAM user "music-gen-backend" with programmatic access
7. Attach custom policy for backend:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": ["s3:PutObject", "s3:GetObject"],
         "Resource": "arn:aws:s3:::your-bucket-name/*"
       },
       {
         "Effect": "Allow",
         "Action": "s3:ListBucket",
         "Resource": "arn:aws:s3:::your-bucket-name"
       }
     ]
   }
   ```
8. Generate and save access keys
9. Create IAM user "music-gen-frontend" with read-only policy
10. Generate and save frontend access keys

#### Tests:
```python
# Create backend/test_s3.py
import boto3
import os

# Configure
s3 = boto3.client(
    's3',
    aws_access_key_id='YOUR_KEY',
    aws_secret_access_key='YOUR_SECRET',
    region_name='us-east-1'
)
bucket = 'your-bucket-name'

# Test 1: Upload test file
print("Test 1: Uploading file...")
with open('/tmp/test.txt', 'w') as f:
    f.write('Test upload')
s3.upload_file('/tmp/test.txt', bucket, 'test/test.txt')
print("✓ Upload successful")

# Test 2: Download file
print("\nTest 2: Downloading file...")
s3.download_file(bucket, 'test/test.txt', '/tmp/download.txt')
with open('/tmp/download.txt', 'r') as f:
    assert f.read() == 'Test upload'
print("✓ Download successful")

# Test 3: List objects
print("\nTest 3: Listing objects...")
response = s3.list_objects_v2(Bucket=bucket, Prefix='test/')
assert response['KeyCount'] > 0
print(f"✓ Found {response['KeyCount']} objects")

# Cleanup
s3.delete_object(Bucket=bucket, Key='test/test.txt')
print("\n✓✓✓ ALL S3 TESTS PASSED! ✓✓✓")
```

```bash
python backend/test_s3.py
# Expected: All tests pass
```

---

### STEP 1.3: Create Modal Application with Image

#### Micro-steps:
1. Create `backend/main.py`
2. Import Modal SDK
3. Define Modal App
4. Create Docker image with dependencies
5. Clone and install ACE-Step in image
6. Create Modal volumes for model caching
7. Create Modal secret for AWS credentials
8. Define Pydantic models

#### Tests:
```python
# backend/main.py
import modal
import os
from pydantic import BaseModel
from typing import List

app = modal.App("music-generator-v2")

# Create image with all dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "boto3",
        "transformers==4.45.2",
        "diffusers==0.31.0",
        "torch==2.5.1",
        "pydantic==2.0.0",
        "accelerate==0.33.0",
        "fastapi==0.115.0"
    )
    .run_commands([
        "git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step",
        "cd /tmp/ACE-Step && pip install ."
    ])
    .env({"HF_HOME": "/.cache/huggingface"})
)

# Create volumes for model caching
model_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

# Create secret for AWS
music_gen_secrets = modal.Secret.from_name("music-gen-secret")

# Pydantic Models
class AudioGenerationBase(BaseModel):
    audio_duration: float = 180.0
    seed: int = -1
    guidance_scale: float = 15.0
    infer_step: int = 60
    instrumental: bool = False

class GenerateFromDescriptionRequest(AudioGenerationBase):
    full_described_song: str

class GenerateWithCustomLyricsRequest(AudioGenerationBase):
    prompt: str
    lyrics: str

class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    prompt: str
    described_lyrics: str

class GenerateMusicResponseS3(BaseModel):
    s3_key: str
    cover_image_s3_key: str
    categories: List[str]

# Test function
@app.function(image=image)
def test_environment():
    import sys
    import torch
    import transformers

    print(f"✓ Python: {sys.version}")
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ Transformers: {transformers.__version__}")

    # Test ACE-Step import
    try:
        from acestep.pipeline_ace_step import ACEStepPipeline
        print("✓ ACE-Step imported successfully")
        return True
    except ImportError as e:
        print(f"✗ ACE-Step import failed: {e}")
        return False

@app.local_entrypoint()
def main():
    result = test_environment.remote()
    if result:
        print("\n✓✓✓ ENVIRONMENT TEST PASSED! ✓✓✓")
```

```bash
# Create Modal secret
modal secret create music-gen-secret \
  AWS_ACCESS_KEY_ID=your_key \
  AWS_SECRET_ACCESS_KEY=your_secret \
  AWS_REGION=us-east-1 \
  S3_BUCKET_NAME=your-bucket-name

# Test environment
modal run backend/main.py
# Expected: All imports successful, ACE-Step loads

# Verify volumes created
modal volume list
# Expected: ace-step-models, qwen-hf-cache

# Verify secrets
modal secret list
# Expected: music-gen-secret
```

---

### STEP 1.4: Create Prompt Templates

#### Micro-steps:
1. Create `backend/prompts.py`
2. Define prompt generator template
3. Define lyrics generator template
4. Use clear formatting instructions
5. Test templates manually

#### Tests:
```python
# backend/prompts.py
PROMPT_GENERATOR_PROMPT = """Generate a concise music style prompt based on this description: '{user_prompt}'

Output ONLY a comma-separated list of music tags like: genre, tempo, mood, instruments

Example: electronic, 120BPM, energetic, synthesizer, bass"""

LYRICS_GENERATOR_PROMPT = """Generate song lyrics based on this description: '{description}'

Format the lyrics with proper structure tags:
[verse]
Lyrics here...

[chorus]
Chorus lyrics...

[verse]
More lyrics...

[bridge]
Bridge lyrics...

Keep it creative and match the description."""

# Manual test
if __name__ == "__main__":
    test_description = "upbeat electronic dance music"

    prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt=test_description)
    print("Prompt Template:")
    print(prompt)
    print("\n" + "="*50 + "\n")

    lyrics = LYRICS_GENERATOR_PROMPT.format(description=test_description)
    print("Lyrics Template:")
    print(lyrics)

    print("\n✓ Templates created")
```

```bash
python backend/prompts.py
# Expected: Templates print correctly with placeholders
```

---

## PHASE 2: AI Model Integration

### STEP 2.1: Load All Three Models

#### Micro-steps:
1. Create MusicGenServer class
2. Add @modal.cls decorator with GPU config
3. Implement @modal.enter() for model loading
4. Load ACE-Step pipeline
5. Load Qwen2-7B model and tokenizer
6. Load SDXL-Turbo pipeline
7. Test model loading
8. Verify GPU allocation

#### Tests:
```python
# Add to backend/main.py
@app.cls(
    image=image,
    gpu="L40S",  # Or "A10G" or "T4" depending on availability
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=15,  # Keep warm for 15 seconds after last request
    timeout=600  # 10 minute timeout
)
class MusicGenServer:
    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from diffusers import AutoPipelineForText2Image
        import torch

        print("Loading models...")

        # 1. Music Generation Model
        print("Loading ACE-Step...")
        self.music_model = ACEStepPipeline(
            checkpoint_dir="/models",
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )
        print("✓ ACE-Step loaded")

        # 2. Large Language Model
        print("Loading Qwen2-7B...")
        model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir="/.cache/huggingface"
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface"
        )
        print("✓ Qwen2-7B loaded")

        # 3. Image Generation Model
        print("Loading SDXL-Turbo...")
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="/.cache/huggingface"
        )
        self.image_pipe.to("cuda")
        print("✓ SDXL-Turbo loaded")

        print("✓✓✓ All models loaded successfully!")

@app.local_entrypoint()
def test_model_loading():
    server = MusicGenServer()
    print("Models initialized!")
```

```bash
# Test model loading
modal run backend/main.py::test_model_loading
# Expected: All three models load without errors
# Note: First run will download models (may take 10-15 minutes)
# Subsequent runs will use cached models (faster)

# Monitor Modal dashboard
# Check GPU utilization during model loading
```

---

### STEP 2.2: Implement LLM Helper Methods

#### Micro-steps:
1. Create prompt_qwen() method
2. Implement generate_prompt()
3. Implement generate_lyrics()
4. Implement generate_categories()
5. Test each method individually
6. Verify output formatting

#### Tests:
```python
# Add to MusicGenServer class in backend/main.py
def prompt_qwen(self, question: str):
    """Helper to query Qwen2-7B"""
    messages = [{"role": "user", "content": question}]
    text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

    generated_ids = self.llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def generate_prompt(self, description: str):
    """Generate music style tags from description"""
    from prompts import PROMPT_GENERATOR_PROMPT
    full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt=description)
    return self.prompt_qwen(full_prompt)

def generate_lyrics(self, description: str):
    """Generate structured lyrics from description"""
    from prompts import LYRICS_GENERATOR_PROMPT
    full_prompt = LYRICS_GENERATOR_PROMPT.format(description=description)
    return self.prompt_qwen(full_prompt)

def generate_categories(self, description: str) -> List[str]:
    """Extract 3-5 genre categories"""
    prompt = f"Based on this music description, list 3-5 genres as comma-separated: '{description}'"
    response = self.prompt_qwen(prompt)
    categories = [cat.strip() for cat in response.split(",") if cat.strip()]
    return categories[:5]

@modal.method()
def test_llm_methods(self):
    """Test all LLM helper methods"""
    import time

    description = "upbeat electronic dance music"

    # Test 1: Prompt generation
    print("\n--- Test 1: Prompt Generation ---")
    start = time.time()
    prompt = self.generate_prompt(description)
    print(f"Generated prompt: {prompt}")
    print(f"Time: {time.time() - start:.2f}s")
    assert len(prompt) > 0
    assert "," in prompt
    print("✓ Prompt generation works")

    # Test 2: Lyrics generation
    print("\n--- Test 2: Lyrics Generation ---")
    start = time.time()
    lyrics = self.generate_lyrics("song about summer and freedom")
    print(f"Generated lyrics (first 200 chars):\n{lyrics[:200]}...")
    print(f"Time: {time.time() - start:.2f}s")
    assert len(lyrics) > 50
    assert "[verse]" in lyrics.lower() or "[chorus]" in lyrics.lower()
    print("✓ Lyrics generation works")

    # Test 3: Category generation
    print("\n--- Test 3: Category Generation ---")
    start = time.time()
    categories = self.generate_categories(description)
    print(f"Generated categories: {categories}")
    print(f"Time: {time.time() - start:.2f}s")
    assert isinstance(categories, list)
    assert 2 <= len(categories) <= 5
    print("✓ Category generation works")

    print("\n✓✓✓ All LLM tests passed!")
    return True

# Add to local_entrypoint
@app.local_entrypoint()
def test_llm():
    server = MusicGenServer()
    server.test_llm_methods.remote()
```

```bash
modal run backend/main.py::test_llm
# Expected: All three LLM functions work correctly
# Verify prompt is comma-separated
# Verify lyrics have structure tags
# Verify 2-5 categories returned
```

---

### STEP 2.3: Implement Full Pipeline Method

#### Micro-steps:
1. Create generate_and_upload_to_s3() method
2. Generate audio with music model
3. Upload audio to S3
4. Generate thumbnail with image model
5. Upload thumbnail to S3
6. Extract categories
7. Return response with S3 keys
8. Add cleanup of temp files
9. Test end-to-end

#### Tests:
```python
# Add to MusicGenServer class
def generate_and_upload_to_s3(
    self,
    prompt: str,
    lyrics: str,
    instrumental: bool,
    audio_duration: float,
    infer_step: int,
    guidance_scale: float,
    seed: int,
    description_for_categorization: str
) -> GenerateMusicResponseS3:
    import uuid
    import boto3

    # Setup
    s3_client = boto3.client("s3")
    bucket_name = os.environ["S3_BUCKET_NAME"]
    output_dir = "/tmp/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare lyrics
    final_lyrics = "[instrumental]" if instrumental else lyrics
    print(f"Generated lyrics:\n{final_lyrics}")
    print(f"Prompt: {prompt}")

    # 1. Generate audio
    print("\nStep 1: Generating audio...")
    audio_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

    self.music_model(
        prompt=prompt,
        lyrics=final_lyrics,
        audio_duration=audio_duration,
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        save_path=audio_path,
        manual_seeds=str(seed)
    )
    print(f"✓ Audio generated: {os.path.getsize(audio_path) / (1024*1024):.2f} MB")

    # 2. Upload audio to S3
    print("Step 2: Uploading audio to S3...")
    audio_s3_key = f"{uuid.uuid4()}.wav"
    s3_client.upload_file(audio_path, bucket_name, audio_s3_key)
    print(f"✓ Uploaded: s3://{bucket_name}/{audio_s3_key}")
    os.remove(audio_path)

    # 3. Generate thumbnail
    print("Step 3: Generating thumbnail...")
    thumbnail_prompt = f"{prompt}, album cover art"
    image = self.image_pipe(
        prompt=thumbnail_prompt,
        num_inference_steps=2,  # SDXL-Turbo uses 2 steps
        guidance_scale=0.0       # SDXL-Turbo doesn't use guidance
    ).images[0]

    image_path = os.path.join(output_dir, f"{uuid.uuid4()}.png")
    image.save(image_path)
    print(f"✓ Thumbnail generated: {os.path.getsize(image_path) / 1024:.2f} KB")

    # 4. Upload thumbnail to S3
    print("Step 4: Uploading thumbnail to S3...")
    image_s3_key = f"{uuid.uuid4()}.png"
    s3_client.upload_file(image_path, bucket_name, image_s3_key)
    print(f"✓ Uploaded: s3://{bucket_name}/{image_s3_key}")
    os.remove(image_path)

    # 5. Generate categories
    print("Step 5: Generating categories...")
    categories = self.generate_categories(description_for_categorization)
    print(f"✓ Categories: {categories}")

    return GenerateMusicResponseS3(
        s3_key=audio_s3_key,
        cover_image_s3_key=image_s3_key,
        categories=categories
    )

@modal.method()
def test_full_pipeline(self):
    """Test complete pipeline"""
    print("=== Testing Full Pipeline ===\n")

    # Use short duration for testing
    result = self.generate_and_upload_to_s3(
        prompt="electronic, upbeat, 120BPM",
        lyrics="[verse]\nTest lyrics for demo\n[chorus]\nThis is a test",
        instrumental=False,
        audio_duration=30.0,  # Short for testing
        infer_step=60,
        guidance_scale=15.0,
        seed=42,
        description_for_categorization="upbeat electronic dance music"
    )

    print(f"\n=== Results ===")
    print(f"Audio S3 key: {result.s3_key}")
    print(f"Thumbnail S3 key: {result.cover_image_s3_key}")
    print(f"Categories: {result.categories}")

    # Verify files exist in S3
    import boto3
    s3 = boto3.client("s3")
    bucket = os.environ["S3_BUCKET_NAME"]

    audio_obj = s3.head_object(Bucket=bucket, Key=result.s3_key)
    thumb_obj = s3.head_object(Bucket=bucket, Key=result.cover_image_s3_key)

    print(f"\nVerification:")
    print(f"✓ Audio exists in S3: {audio_obj['ContentLength'] / (1024*1024):.2f} MB")
    print(f"✓ Thumbnail exists in S3: {thumb_obj['ContentLength'] / 1024:.2f} KB")

    assert audio_obj['ContentLength'] > 10000
    assert thumb_obj['ContentLength'] > 1000

    print("\n✓✓✓ FULL PIPELINE TEST PASSED! ✓✓✓")
    return result

@app.local_entrypoint()
def test_pipeline():
    server = MusicGenServer()
    server.test_full_pipeline.remote()
```

```bash
modal run backend/main.py::test_pipeline
# Expected: Complete generation workflow
# Audio file uploaded to S3
# Thumbnail uploaded to S3
# Categories extracted
# All assertions pass
# Note: This may take 3-5 minutes for 30s audio
```

---

## PHASE 3: API Endpoints

### STEP 3.1: Create FastAPI Endpoints

#### Micro-steps:
1. Add @modal.fastapi_endpoint decorators
2. Create generate_from_description endpoint
3. Create generate_with_lyrics endpoint
4. Create generate_with_described_lyrics endpoint
5. Enable API docs
6. Deploy to Modal
7. Test endpoints

#### Tests:
```python
# Add to MusicGenServer class in backend/main.py
@modal.fastapi_endpoint(method="POST", docs=True)
def generate_from_description(self, request: GenerateFromDescriptionRequest):
    """
    Generate music from a simple text description.
    Both prompt and lyrics are auto-generated by LLM.
    """
    # Generate prompt from description
    prompt = self.generate_prompt(request.full_described_song)

    # Generate lyrics if not instrumental
    lyrics = ""
    if not request.instrumental:
        lyrics = self.generate_lyrics(request.full_described_song)

    # Generate and upload
    return self.generate_and_upload_to_s3(
        prompt=prompt,
        lyrics=lyrics,
        description_for_categorization=request.full_described_song,
        **request.model_dump(exclude={"full_described_song"})
    )

@modal.fastapi_endpoint(method="POST", docs=True)
def generate_with_lyrics(self, request: GenerateWithCustomLyricsRequest):
    """
    Generate music with custom prompt and lyrics.
    User provides both the style tags and the lyrics.
    """
    return self.generate_and_upload_to_s3(
        prompt=request.prompt,
        lyrics=request.lyrics,
        description_for_categorization=request.prompt,
        **request.model_dump(exclude={"prompt", "lyrics"})
    )

@modal.fastapi_endpoint(method="POST", docs=True)
def generate_with_described_lyrics(self, request: GenerateWithDescribedLyricsRequest):
    """
    Generate music with custom prompt and LLM-generated lyrics.
    User provides style tags and describes what lyrics should be about.
    """
    # Generate lyrics from description
    lyrics = ""
    if not request.instrumental:
        lyrics = self.generate_lyrics(request.described_lyrics)

    return self.generate_and_upload_to_s3(
        prompt=request.prompt,
        lyrics=lyrics,
        description_for_categorization=request.prompt,
        **request.model_dump(exclude={"described_lyrics", "prompt"})
    )
```

```bash
# Deploy backend to Modal
cd backend
modal deploy main.py
# Expected: Deployment successful
# Note deployment URL from output

# Check Modal dashboard
# Navigate to: https://modal.com/apps
# Expected: See music-generator-v2 app

# Access API docs
# Navigate to: https://your-app--music-generator-v2.modal.run/docs
# Expected: Swagger UI with 3 POST endpoints

# Test endpoint manually via Swagger UI
# 1. Click "Try it out" on generate_from_description
# 2. Enter test data:
{
  "full_described_song": "energetic rock song",
  "audio_duration": 30,
  "guidance_scale": 15,
  "instrumental": false,
  "seed": 42
}
# 3. Execute
# Expected: Returns S3 keys and categories after 3-5 minutes
```

---

### STEP 3.2: Test Endpoints Programmatically

#### Micro-steps:
1. Get deployed endpoint URLs
2. Create test script
3. Test each endpoint
4. Verify responses
5. Check S3 uploads
6. Test error handling

#### Tests:
```python
# Create backend/test_endpoints.py
import requests
import json
import time

# IMPORTANT: Update this with your deployed URL
BASE_URL = "https://your-username--music-generator-v2.modal.run"

def test_description_endpoint():
    """Test /generate_from_description"""
    print("\n=== Testing Description Endpoint ===")

    url = f"{BASE_URL}/generate_from_description"
    payload = {
        "full_described_song": "chill lofi hip-hop beat",
        "audio_duration": 30.0,
        "guidance_scale": 15.0,
        "instrumental": True,
        "seed": 42
    }

    print(f"Sending request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    start = time.time()
    response = requests.post(url, json=payload, timeout=600)
    duration = time.time() - start

    print(f"Response time: {duration:.2f}s")
    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success!")
        print(f"  Audio S3 key: {result['s3_key']}")
        print(f"  Thumbnail S3 key: {result['cover_image_s3_key']}")
        print(f"  Categories: {result['categories']}")
        return result
    else:
        print(f"✗ Failed: {response.text}")
        return None

def test_custom_lyrics_endpoint():
    """Test /generate_with_lyrics"""
    print("\n=== Testing Custom Lyrics Endpoint ===")

    url = f"{BASE_URL}/generate_with_lyrics"
    payload = {
        "prompt": "hip-hop, 90BPM, chill",
        "lyrics": "[verse]\nTest custom lyrics\n[chorus]\nThis is a test",
        "audio_duration": 30.0,
        "guidance_scale": 15.0,
        "instrumental": False
    }

    response = requests.post(url, json=payload, timeout=600)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success: {result['s3_key']}")
        return result
    else:
        print(f"✗ Failed: {response.text}")
        return None

def test_described_lyrics_endpoint():
    """Test /generate_with_described_lyrics"""
    print("\n=== Testing Described Lyrics Endpoint ===")

    url = f"{BASE_URL}/generate_with_described_lyrics"
    payload = {
        "prompt": "pop, upbeat, 120BPM",
        "described_lyrics": "lyrics about summer and happiness",
        "audio_duration": 30.0,
        "guidance_scale": 15.0,
        "instrumental": False
    }

    response = requests.post(url, json=payload, timeout=600)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success: {result['s3_key']}")
        return result
    else:
        print(f"✗ Failed: {response.text}")
        return None

def test_error_handling():
    """Test invalid requests"""
    print("\n=== Testing Error Handling ===")

    url = f"{BASE_URL}/generate_from_description"

    # Missing required field
    bad_payload = {"audio_duration": 180}
    response = requests.post(url, json=bad_payload)

    print(f"Invalid request status: {response.status_code}")
    assert response.status_code in [400, 422], "Should reject invalid request"
    print("✓ Error handling works")

if __name__ == "__main__":
    print("="*60)
    print("MODAL API ENDPOINT TESTS")
    print("="*60)

    # Run tests
    test_description_endpoint()
    test_custom_lyrics_endpoint()
    test_described_lyrics_endpoint()
    test_error_handling()

    print("\n" + "="*60)
    print("✓✓✓ ALL ENDPOINT TESTS COMPLETED! ✓✓✓")
    print("="*60)
```

```bash
# Update BASE_URL in script with your deployed URL
# Then run tests
python backend/test_endpoints.py
# Expected: All tests pass
# Note: Each test takes 3-5 minutes due to generation time
```

---

## PHASE 4: Frontend Setup

### STEP 4.1: Neon PostgreSQL Setup

#### Micro-steps:
1. Go to neon.tech
2. Create free account
3. Create new project "music-generator"
4. Wait for provisioning
5. Copy connection string
6. Save DIRECT_URL and pooled URL
7. Note database credentials

#### Tests:
```bash
# Test connection with psql (if installed)
psql "postgresql://username:password@host/database"
# Expected: Connects successfully

# Or test with Node.js
# Create test_neon.js
const { neonConfig } = require('@neondatabase/serverless');
const { Pool } = require('@neondatabase/serverless');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL
});

async function test() {
  const result = await pool.query('SELECT NOW()');
  console.log('✓ Connected to Neon');
  console.log('Server time:', result.rows[0].now);
  await pool.end();
}

test().catch(console.error);
```

```bash
node test_neon.js
# Expected: Shows current server time
```

---

### STEP 4.2: Next.js Project Setup

#### Micro-steps:
1. Create frontend directory
2. Initialize Next.js project
3. Install dependencies
4. Set up Shadcn UI
5. Add UI components
6. Configure Tailwind
7. Test dev server

#### Tests:
```bash
# Create Next.js project
npx create-next-app@latest frontend \
  --typescript \
  --tailwind \
  --app \
  --use-npm \
  --no-src-dir

cd frontend

# Install dependencies
npm install @neondatabase/serverless
npm install drizzle-orm
npm install drizzle-kit -D
npm install better-auth
npm install inngest
npm install zod
npm install zustand
npm install sonner
npm install lucide-react
npm install next-themes
npm install aws-sdk
npm install @aws-sdk/client-s3
npm install dotenv

# Initialize Shadcn UI
npx shadcn@latest init
# Choose:
# - Style: Default
# - Base color: Slate
# - CSS variables: Yes

# Add Shadcn components
npx shadcn@latest add button
npx shadcn@latest add input
npx shadcn@latest add textarea
npx shadcn@latest add card
npx shadcn@latest add dialog
npx shadcn@latest add dropdown-menu
npx shadcn@latest add tabs
npx shadcn@latest add slider
npx shadcn@latest add switch
npx shadcn@latest add sidebar
npx shadcn@latest add badge
npx shadcn@latest add separator
npx shadcn@latest add tooltip
npx shadcn@latest add breadcrumb
npx shadcn@latest add sheet
npx shadcn@latest add skeleton
npx shadcn@latest add sonner

# Test 1: Verify structure
ls -la
# Expected: node_modules, package.json, app/, components/, etc.

# Test 2: Check UI components
ls components/ui/
# Expected: button.tsx, input.tsx, card.tsx, etc.

# Test 3: Start dev server
npm run dev
# Expected: Server starts on http://localhost:3000

# Test 4: Build project
npm run build
# Expected: Build succeeds

# Test 5: Check for errors
npm run lint
# Expected: No linting errors
```

---

### STEP 4.3: Database Schema with Drizzle

#### Micro-steps:
1. Create lib/db directory
2. Create Drizzle config
3. Define database schema
4. Create migration
5. Push schema to Neon
6. Verify tables created

#### Tests:
```typescript
// Create lib/db/index.ts
import { drizzle } from 'drizzle-orm/neon-serverless';
import { Pool } from '@neondatabase/serverless';

const pool = new Pool({ connectionString: process.env.DATABASE_URL! });
export const db = drizzle(pool);

// Create lib/db/schema.ts
import { pgTable, uuid, varchar, integer, boolean, text, timestamp, real, primaryKey } from 'drizzle-orm/pg-core';

export const users = pgTable('users', {
  id: uuid('id').primaryKey().defaultRandom(),
  email: varchar('email', { length: 255 }).unique().notNull(),
  name: varchar('name', { length: 255 }),
  credits: integer('credits').default(100),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
});

export const songs = pgTable('songs', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: uuid('user_id').references(() => users.id, { onDelete: 'cascade' }),
  title: varchar('title', { length: 255 }).notNull(),
  s3Key: varchar('s3_key', { length: 500 }),
  coverImageS3Key: varchar('cover_image_s3_key', { length: 500 }),
  status: varchar('status', { length: 50 }).default('queued'),
  instrumental: boolean('instrumental').default(false),
  prompt: text('prompt'),
  lyrics: text('lyrics'),
  fullDescribedSong: text('full_described_song'),
  describedLyrics: text('described_lyrics'),
  guidanceScale: real('guidance_scale'),
  inferStep: integer('infer_step'),
  audioDuration: real('audio_duration'),
  seed: integer('seed'),
  published: boolean('published').default(false),
  listenCount: integer('listen_count').default(0),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
});

export const categories = pgTable('categories', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 100 }).unique().notNull(),
});

export const songCategories = pgTable('song_categories', {
  songId: uuid('song_id').references(() => songs.id, { onDelete: 'cascade' }),
  categoryId: uuid('category_id').references(() => categories.id, { onDelete: 'cascade' }),
}, (table) => ({
  pk: primaryKey(table.songId, table.categoryId),
}));

export const likes = pgTable('likes', {
  userId: uuid('user_id').references(() => users.id, { onDelete: 'cascade' }),
  songId: uuid('song_id').references(() => songs.id, { onDelete: 'cascade' }),
  createdAt: timestamp('created_at').defaultNow(),
}, (table) => ({
  pk: primaryKey(table.userId, table.songId),
}));

// Create drizzle.config.ts
import type { Config } from 'drizzle-kit';
import * as dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

export default {
  schema: './lib/db/schema.ts',
  out: './drizzle',
  driver: 'pg',
  dbCredentials: {
    connectionString: process.env.DIRECT_URL!,
  },
} satisfies Config;
```

```bash
# Create .env.local
cat > .env.local << 'EOF'
# Neon Database
DATABASE_URL="your-pooled-connection-string"
DIRECT_URL="your-direct-connection-string"

# AWS S3
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="your-frontend-key"
AWS_SECRET_ACCESS_KEY="your-frontend-secret"
S3_BUCKET_NAME="your-bucket-name"

# Modal Backend
MODAL_API_URL="your-deployed-modal-url"

# Better Auth (generate random string)
BETTER_AUTH_SECRET="your-32-char-random-string"
BETTER_AUTH_URL="http://localhost:3000"
EOF

# Generate migration
npx drizzle-kit generate
# Expected: Creates migration files in drizzle/

# Push to database
npx drizzle-kit push
# Expected: Schema pushed to Neon

# Verify in Neon dashboard
# Go to neon.tech → Your project → Tables
# Expected: users, songs, categories, song_categories, likes tables exist
```

---

## PHASE 5: Authentication with BetterAuth

### STEP 5.1: BetterAuth Setup

#### Micro-steps:
1. Create auth configuration
2. Set up auth API route
3. Create auth client
4. Add middleware for protected routes
5. Test authentication flow

#### Tests:
```typescript
// Create lib/auth.ts
import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import { db } from "@/lib/db";

export const auth = betterAuth({
  database: drizzleAdapter(db, {
    provider: "pg",
  }),
  emailAndPassword: {
    enabled: true,
  },
  session: {
    expiresIn: 60 * 60 * 24 * 7, // 7 days
    updateAge: 60 * 60 * 24, // 1 day
  },
});

// Create lib/auth-client.ts
import { createAuthClient } from "better-auth/client";

export const authClient = createAuthClient({
  baseURL: process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
});

// Create app/api/auth/[...all]/route.ts
import { auth } from "@/lib/auth";

export const { GET, POST } = auth.handler;

// Create middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function middleware(request: NextRequest) {
  const protectedPaths = ['/create', '/dashboard'];
  const isProtected = protectedPaths.some(path =>
    request.nextUrl.pathname.startsWith(path)
  );

  if (isProtected) {
    // Check session
    const sessionCookie = request.cookies.get('better-auth.session_token');

    if (!sessionCookie) {
      return NextResponse.redirect(new URL('/auth/sign-in', request.url));
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
};
```

```bash
# Test auth setup
npm run dev
# Navigate to: http://localhost:3000/api/auth/session
# Expected: Returns session data (null if not logged in)

# Try to access protected route
# Navigate to: http://localhost:3000/create
# Expected: Redirects to /auth/sign-in
```

---

### STEP 5.2: Create Auth Pages

#### Micro-steps:
1. Create auth layout
2. Create sign-up page
3. Create sign-in page
4. Add form validation
5. Test authentication flow

#### Tests:
```typescript
// Create app/auth/layout.tsx
export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="w-full max-w-md">
        {children}
      </div>
    </div>
  );
}

// Create app/auth/sign-up/page.tsx
'use client';

import { useState } from 'react';
import { authClient } from '@/lib/auth-client';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

export default function SignUpPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      await authClient.signUp.email({
        email,
        password,
        name,
      });

      router.push('/create');
    } catch (error) {
      console.error('Sign up failed:', error);
      alert('Sign up failed. Please try again.');
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sign Up</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            type="text"
            placeholder="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
          <Input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <Input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={8}
          />
          <Button type="submit" className="w-full">
            Sign Up
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

// Create app/auth/sign-in/page.tsx
'use client';

import { useState } from 'react';
import { authClient } from '@/lib/auth-client';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import Link from 'next/link';

export default function SignInPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      await authClient.signIn.email({
        email,
        password,
      });

      router.push('/create');
    } catch (error) {
      console.error('Sign in failed:', error);
      alert('Sign in failed. Please check your credentials.');
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sign In</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <Input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <Button type="submit" className="w-full">
            Sign In
          </Button>
          <p className="text-sm text-center">
            Don't have an account?{' '}
            <Link href="/auth/sign-up" className="text-primary hover:underline">
              Sign up
            </Link>
          </p>
        </form>
      </CardContent>
    </Card>
  );
}
```

**Manual Tests:**
```bash
# Start dev server
npm run dev

# Test 1: Sign up
1. Navigate to http://localhost:3000/auth/sign-up
   ✓ Form displays correctly
2. Fill in: name, email, password
3. Click "Sign Up"
   ✓ Redirects to /create
   ✓ Check Neon dashboard - user record created
   ✓ User has 100 credits

# Test 2: Sign in
1. Navigate to http://localhost:3000/auth/sign-in
2. Enter credentials from sign up
3. Click "Sign In"
   ✓ Redirects to /create
   ✓ Session created

# Test 3: Protected routes
1. Sign out (or use incognito window)
2. Try to access http://localhost:3000/create
   ✓ Redirects to /auth/sign-in

# Test 4: Session persistence
1. Sign in
2. Refresh page
   ✓ Still logged in
3. Close browser, reopen
   ✓ Still logged in (within 7 days)
```

---

## PHASE 6: Queue System with Inngest

### STEP 6.1: Inngest Setup

#### Micro-steps:
1. Create Inngest account at inngest.com
2. Get event key and signing key
3. Create Inngest client
4. Set up Inngest serve endpoint
5. Start local dev server
6. Verify connection

#### Tests:
```typescript
// Create lib/inngest/client.ts
import { Inngest } from "inngest";

export const inngest = new Inngest({
  id: "music-generator",
  eventKey: process.env.INNGEST_EVENT_KEY,
});

// Create app/api/inngest/route.ts
import { serve } from "inngest/next";
import { inngest } from "@/lib/inngest/client";
import { generateMusicFunction } from "@/lib/inngest/functions/generate-music";

export const { GET, POST, PUT } = serve({
  client: inngest,
  functions: [
    generateMusicFunction,
  ],
});
```

```bash
# Add to .env.local
INNGEST_EVENT_KEY="your-event-key"
INNGEST_SIGNING_KEY="your-signing-key"

# Start Inngest dev server
npx inngest-cli@latest dev
# Expected: Opens http://localhost:8288

# In separate terminal, start Next.js
npm run dev

# Test 1: Check Inngest UI
# Navigate to http://localhost:8288
# Expected: Shows "music-generator" app

# Test 2: Verify function registration
# Click on "Functions" tab
# Expected: (will show function after we create it)
```

---

### STEP 6.2: Create Music Generation Function

#### Micro-steps:
1. Create generateMusicFunction
2. Implement step-by-step logic
3. Check credits
4. Call Modal backend
5. Update database
6. Deduct credits
7. Handle errors
8. Test function

#### Tests:
```typescript
// Create lib/inngest/functions/generate-music.ts
import { inngest } from "../client";
import { db } from "@/lib/db";
import { songs, users } from "@/lib/db/schema";
import { eq } from "drizzle-orm";

export const generateMusicFunction = inngest.createFunction(
  {
    id: "generate-music",
    concurrency: {
      // Only 1 job per user at a time
      key: "event.data.userId",
      limit: 1,
    },
  },
  { event: "music/generate.requested" },
  async ({ event, step }) => {
    const { songId, userId } = event.data;

    // Step 1: Check credits
    const user = await step.run("check-credits", async () => {
      const [user] = await db.select().from(users).where(eq(users.id, userId));

      if (!user || user.credits < 1) {
        throw new Error("Insufficient credits");
      }

      return user;
    });

    // Step 2: Update song status to processing
    await step.run("set-status-processing", async () => {
      await db
        .update(songs)
        .set({ status: "processing" })
        .where(eq(songs.id, songId));
    });

    // Step 3: Get song details
    const song = await step.run("get-song-details", async () => {
      const [song] = await db.select().from(songs).where(eq(songs.id, songId));
      return song;
    });

    // Step 4: Call Modal backend
    const result = await step.run("generate-audio", async () => {
      const modalUrl = process.env.MODAL_API_URL!;
      const endpoint = `${modalUrl}/generate_from_description`;

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          full_described_song: song.fullDescribedSong,
          audio_duration: song.audioDuration || 180,
          guidance_scale: song.guidanceScale || 15,
          instrumental: song.instrumental || false,
          seed: song.seed || -1,
          infer_step: song.inferStep || 60,
        }),
      });

      if (!response.ok) {
        throw new Error(`Generation failed: ${await response.text()}`);
      }

      return await response.json();
    });

    // Step 5: Update song with results
    await step.run("update-song-results", async () => {
      await db
        .update(songs)
        .set({
          status: "completed",
          s3Key: result.s3_key,
          coverImageS3Key: result.cover_image_s3_key,
        })
        .where(eq(songs.id, songId));
    });

    // Step 6: Deduct credits
    await step.run("deduct-credits", async () => {
      await db
        .update(users)
        .set({ credits: user.credits - 1 })
        .where(eq(users.id, userId));
    });

    return { success: true, songId, s3Key: result.s3_key };
  }
);
```

**Manual Test:**
```typescript
// Create test-inngest.ts
import { inngest } from "./lib/inngest/client";

async function testQueue() {
  // First, manually create a test song in database
  // Then send event

  await inngest.send({
    name: "music/generate.requested",
    data: {
      songId: "your-test-song-id",
      userId: "your-test-user-id",
    },
  });

  console.log("✓ Event sent to Inngest");
}

testQueue();
```

```bash
# Run test
npx tsx test-inngest.ts
# Expected: Event sent

# Check Inngest UI at http://localhost:8288
# Navigate to "Runs" tab
# Expected: See new run in progress

# Monitor steps:
# ✓ check-credits
# ✓ set-status-processing
# ✓ get-song-details
# ✓ generate-audio (takes 3-5 min)
# ✓ update-song-results
# ✓ deduct-credits

# Verify in Neon database:
# - Song status changed to "completed"
# - Song has s3_key and cover_image_s3_key
# - User credits decreased by 1
```

---

## PHASE 7: Core UI - Create Page

### STEP 7.1: Server Action for Queueing Songs

#### Micro-steps:
1. Create actions directory
2. Create queueSong server action
3. Implement database insert
4. Send Inngest event
5. Generate 2 songs (guidance scales 7.5 and 15)
6. Test action

#### Tests:
```typescript
// Create app/actions/generation.ts
"use server";

import { db } from "@/lib/db";
import { songs } from "@/lib/db/schema";
import { inngest } from "@/lib/inngest/client";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { redirect } from "next/navigation";

export async function queueSong(data: {
  fullDescribedSong?: string;
  prompt?: string;
  lyrics?: string;
  describedLyrics?: string;
  instrumental: boolean;
  audioDuration: number;
  seed: number;
  guidanceScale: number;
}) {
  // Check authentication
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session?.user) {
    redirect("/auth/sign-in");
  }

  // Create song record
  const [song] = await db
    .insert(songs)
    .values({
      userId: session.user.id,
      title: `Untitled ${Date.now()}`,
      status: "queued",
      fullDescribedSong: data.fullDescribedSong,
      prompt: data.prompt,
      lyrics: data.lyrics,
      describedLyrics: data.describedLyrics,
      instrumental: data.instrumental,
      audioDuration: data.audioDuration,
      seed: data.seed,
      guidanceScale: data.guidanceScale,
      inferStep: 60,
    })
    .returning();

  // Send to queue
  await inngest.send({
    name: "music/generate.requested",
    data: {
      songId: song.id,
      userId: session.user.id,
    },
  });

  return song;
}

export async function generateSong(data: {
  fullDescribedSong?: string;
  prompt?: string;
  lyrics?: string;
  describedLyrics?: string;
  instrumental: boolean;
  audioDuration: number;
  seed: number;
}) {
  // Generate 2 songs with different guidance scales for variety
  await queueSong({ ...data, guidanceScale: 7.5 });
  await queueSong({ ...data, guidanceScale: 15.0 });
}
```

**Manual Test:**
```typescript
// Test in Next.js page or create test file
// This will be tested through UI once we build it
```

---

### STEP 7.2: Create Page with Three Modes

#### Micro-steps:
1. Create (main) layout with sidebar
2. Create /create page
3. Add three-tab layout
4. Create description form component
5. Create custom lyrics form component
6. Create described lyrics form component
7. Add shared controls (instrumental toggle, duration slider)
8. Wire up to server actions
9. Test all three modes

#### Tests:
```typescript
// Create app/(main)/layout.tsx
import { AppSidebar } from "@/components/sidebar/app-sidebar";
import { SidebarProvider } from "@/components/ui/sidebar";

export default function MainLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full">
        <AppSidebar />
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </SidebarProvider>
  );
}

// Create app/(main)/create/page.tsx
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DescriptionForm } from "@/components/create/description-form";
import { CustomLyricsForm } from "@/components/create/custom-lyrics-form";
import { DescribedLyricsForm } from "@/components/create/described-lyrics-form";

export default function CreatePage() {
  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-8">Create Music</h1>

      <Tabs defaultValue="description" className="w-full">
        <TabsList className="grid w-full grid-cols-3 mb-8">
          <TabsTrigger value="description">Description</TabsTrigger>
          <TabsTrigger value="custom">Custom Lyrics</TabsTrigger>
          <TabsTrigger value="described">Described Lyrics</TabsTrigger>
        </TabsList>

        <TabsContent value="description">
          <DescriptionForm />
        </TabsContent>

        <TabsContent value="custom">
          <CustomLyricsForm />
        </TabsContent>

        <TabsContent value="described">
          <DescribedLyricsForm />
        </TabsContent>
      </Tabs>
    </div>
  );
}

// Create components/create/description-form.tsx
"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { generateSong } from "@/app/actions/generation";
import { toast } from "sonner";

export function DescriptionForm() {
  const [description, setDescription] = useState("");
  const [instrumental, setInstrumental] = useState(false);
  const [duration, setDuration] = useState([180]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!description.trim()) {
      toast.error("Please enter a description");
      return;
    }

    setLoading(true);

    try {
      await generateSong({
        fullDescribedSong: description,
        instrumental,
        audioDuration: duration[0],
        seed: -1,
      });

      toast.success("Songs queued for generation!");
      setDescription("");
    } catch (error) {
      toast.error("Failed to queue songs");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Generate from Description</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium mb-2">
              Describe your song
            </label>
            <Textarea
              placeholder="e.g., upbeat electronic dance music with synthesizers"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={4}
              maxLength={500}
            />
            <p className="text-xs text-muted-foreground mt-1">
              {description.length}/500 characters
            </p>
          </div>

          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Instrumental</label>
            <Switch checked={instrumental} onCheckedChange={setInstrumental} />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Duration: {duration[0]}s
            </label>
            <Slider
              value={duration}
              onValueChange={setDuration}
              min={60}
              max={240}
              step={30}
            />
          </div>

          <Button type="submit" disabled={loading} className="w-full">
            {loading ? "Queueing..." : "Generate Song"}
          </Button>

          <p className="text-xs text-center text-muted-foreground">
            This will generate 2 songs with different styles
          </p>
        </form>
      </CardContent>
    </Card>
  );
}

// Create similar components for:
// - components/create/custom-lyrics-form.tsx
// - components/create/described-lyrics-form.tsx
// (Follow same pattern, adjust fields)
```

**Manual Tests:**
```bash
# Start servers
npm run dev # Terminal 1
npx inngest-cli dev # Terminal 2

# Test 1: Description mode
1. Navigate to http://localhost:3000/create
2. Enter description: "chill lofi hip-hop beat"
3. Toggle instrumental: ON
4. Set duration: 60s
5. Click "Generate Song"
   ✓ Toast shows "Songs queued"
   ✓ Check Inngest UI - 2 events queued
   ✓ Check Neon - 2 song records created
   ✓ Wait 3-5 min - songs complete

# Test 2: Custom lyrics mode
1. Switch to "Custom Lyrics" tab
2. Enter prompt: "rap, 90BPM, aggressive"
3. Enter lyrics: "[verse]\nTest\n[chorus]\nTest"
4. Click "Generate Song"
   ✓ 2 songs queued

# Test 3: Described lyrics mode
1. Switch to "Described Lyrics" tab
2. Enter prompt: "pop, upbeat"
3. Enter lyrics description: "lyrics about summer"
4. Click "Generate Song"
   ✓ 2 songs queued
```

---

## PHASE 8: Dashboard & Playback

### STEP 8.1: Display User's Songs

#### Micro-steps:
1. Query user's songs from database
2. Create song list component
3. Display status badges
4. Show thumbnails
5. Add loading states
6. Auto-refresh for pending songs
7. Test display

#### Tests:
```typescript
// Create app/(main)/dashboard/page.tsx
import { db } from "@/lib/db";
import { songs } from "@/lib/db/schema";
import { eq, desc } from "drizzle-orm";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { SongCard } from "@/components/dashboard/song-card";

export default async function DashboardPage() {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session?.user) {
    redirect("/auth/sign-in");
  }

  const userSongs = await db
    .select()
    .from(songs)
    .where(eq(songs.userId, session.user.id))
    .orderBy(desc(songs.createdAt));

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold">My Songs</h1>
        <Button asChild>
          <Link href="/create">Create New</Link>
        </Button>
      </div>

      {userSongs.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground mb-4">
            You haven't created any songs yet
          </p>
          <Button asChild>
            <Link href="/create">Create Your First Song</Link>
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {userSongs.map((song) => (
            <SongCard key={song.id} song={song} />
          ))}
        </div>
      )}
    </div>
  );
}

// Create components/dashboard/song-card.tsx
"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Play, Download, Trash2, Edit } from "lucide-react";
import Image from "next/image";

export function SongCard({ song }: { song: any }) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-500";
      case "processing":
        return "bg-yellow-500";
      case "queued":
        return "bg-blue-500";
      case "failed":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  return (
    <Card>
      <CardContent className="p-4">
        {/* Thumbnail */}
        <div className="relative aspect-square mb-4 bg-muted rounded-lg overflow-hidden">
          {song.coverImageS3Key ? (
            <Image
              src={`https://${process.env.NEXT_PUBLIC_S3_BUCKET_NAME}.s3.amazonaws.com/${song.coverImageS3Key}`}
              alt={song.title}
              fill
              className="object-cover"
            />
          ) : (
            <div className="flex items-center justify-center h-full">
              <span className="text-muted-foreground">Generating...</span>
            </div>
          )}
        </div>

        {/* Title & Status */}
        <div className="mb-4">
          <h3 className="font-semibold truncate">{song.title}</h3>
          <Badge className={getStatusColor(song.status)}>
            {song.status}
          </Badge>
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <Button
            size="sm"
            disabled={song.status !== "completed"}
            className="flex-1"
          >
            <Play className="h-4 w-4 mr-1" />
            Play
          </Button>
          <Button size="sm" variant="outline">
            <Edit className="h-4 w-4" />
          </Button>
          <Button size="sm" variant="outline">
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
```

**Manual Tests:**
```bash
# Test 1: Empty state
1. Sign in with new account
2. Navigate to /dashboard
   ✓ Shows "You haven't created any songs yet"
   ✓ Shows "Create Your First Song" button

# Test 2: With songs
1. Create some songs
2. Navigate to /dashboard
   ✓ Shows all user's songs
   ✓ Status badges correct colors
   ✓ Queued songs show "Generating..." thumbnail
   ✓ Completed songs show actual thumbnail

# Test 3: Auto-refresh
1. Create a song
2. Stay on dashboard
3. Manually refresh page every 30s
   ✓ Status updates from queued → processing → completed
   ✓ Thumbnail appears when completed

# Test 4: Grid layout
1. Resize browser window
   ✓ Mobile: 1 column
   ✓ Tablet: 2 columns
   ✓ Desktop: 3 columns
```

---

## PHASE 9: Deployment

### STEP 9.1: Deploy Backend to Modal

#### Micro-steps:
1. Review backend code
2. Ensure all secrets configured
3. Deploy to Modal
4. Get production URLs
5. Test production endpoints
6. Monitor logs

#### Tests:
```bash
# Deploy backend
cd backend
modal deploy main.py
# Expected: Deployment successful
# Copy the endpoint URLs from output

# Test production endpoint
curl -X POST "https://your-username--music-generator-v2.modal.run/generate_from_description" \
  -H "Content-Type: application/json" \
  -d '{
    "full_described_song": "test song",
    "audio_duration": 30,
    "guidance_scale": 15,
    "instrumental": true,
    "seed": 42
  }'
# Expected: Returns S3 keys after 3-5 minutes

# Check Modal dashboard
# Navigate to modal.com/apps
# Expected: See deployed app, view logs, monitor GPU usage

# Verify volumes
modal volume list
# Expected: Shows model volumes with cached data
```

---

### STEP 9.2: Deploy Frontend to Vercel

#### Micro-steps:
1. Push code to GitHub
2. Create Vercel account
3. Import project
4. Configure environment variables
5. Deploy
6. Test production site
7. Configure Inngest production

#### Tests:
```bash
# Push to GitHub
git add .
git commit -m "Ready for production"
git push origin main

# Deploy to Vercel:
# 1. Go to vercel.com
# 2. Click "Import Project"
# 3. Select your repository
# 4. Configure build settings:
#    - Framework: Next.js
#    - Root Directory: frontend/
#    - Build Command: npm run build
#    - Output Directory: .next

# 5. Add environment variables:
#    Copy all from .env.local
#    Update MODAL_API_URL to production URL
#    Update BETTER_AUTH_URL to Vercel URL

# 6. Deploy
# Expected: Build succeeds, site live

# Test production:
# 1. Visit Vercel URL
#    ✓ Site loads
# 2. Sign up
#    ✓ Account created
# 3. Create song
#    ✓ Queued successfully
# 4. Check Inngest
#    ✓ Event received

# Configure Inngest for production:
# 1. Go to inngest.com dashboard
# 2. Add production app
# 3. Configure webhook endpoint: https://your-site.vercel.app/api/inngest
# 4. Test event delivery
#    ✓ Events reach production
```

---

## PHASE 10: Polish & Testing

### STEP 10.1: Add Remaining Features

#### Features to implement:
1. **Audio Player**: Global player component with play/pause, progress bar
2. **Rename Song**: Dialog to rename songs
3. **Delete Song**: Confirmation dialog to delete songs
4. **Publish Toggle**: Make songs public/private
5. **Home Feed**: Display all published songs
6. **Like System**: Like/unlike songs
7. **Credits Display**: Show remaining credits in sidebar
8. **Download Song**: Download generated audio files

#### Tests:
```bash
# Manual testing checklist for each feature
# Test on desktop and mobile
# Verify all CRUD operations work
# Check responsive design
# Verify error handling
```

---

### STEP 10.2: End-to-End Testing

#### Complete User Journey Test:
```bash
# Full E2E Test
1. Sign up new account
   ✓ Account created with 100 credits

2. Create song (Description mode)
   - Description: "upbeat electronic dance music"
   - Duration: 60s
   - Instrumental: OFF
   ✓ 2 songs queued
   ✓ Credits: 98

3. Wait for completion (5-8 minutes)
   ✓ Status: completed
   ✓ Thumbnails generated

4. Play song
   ✓ Audio plays in player
   ✓ Progress bar works
   ✓ Volume control works

5. Rename song
   ✓ New name saved

6. Publish song
   ✓ Marked as published

7. View in home feed
   ✓ Song appears in feed

8. Like song (from different account)
   ✓ Like count increases

9. Download song
   ✓ File downloads correctly

10. Delete song
    ✓ Confirmation shown
    ✓ Song removed
    ✓ S3 files deleted

✓✓✓ ALL E2E TESTS PASSED! ✓✓✓
```

---

## SUCCESS CRITERIA

### Backend ✅
- [  ] All three generation modes working
- [  ] Audio generation completes in < 5 minutes
- [  ] Thumbnails generated successfully
- [  ] Categories extracted accurately
- [  ] Files uploaded to S3
- [  ] Modal deployment stable
- [  ] API docs accessible

### Frontend ✅
- [  ] Users can sign up/sign in
- [  ] Users can create songs (all 3 modes)
- [  ] Songs queue and process correctly
- [  ] Audio playback works
- [  ] Users can publish/rename/delete songs
- [  ] Feed displays published songs
- [  ] Like system works
- [  ] Credits system works
- [  ] Responsive design works

### Infrastructure ✅
- [  ] Database schema correct
- [  ] Inngest queue processing reliably
- [  ] Concurrency control (1 per user)
- [  ] Production deployment stable
- [  ] Error monitoring active

---

## MAINTENANCE

### Daily
- Check Inngest dashboard for failed jobs
- Monitor Modal GPU usage
- Check Neon database size

### Weekly
- Review user feedback
- Check S3 storage costs
- Review error logs
- Update dependencies if needed

### Monthly
- Optimize database queries
- Review and improve prompts
- Analyze usage patterns
- Consider model upgrades

---

## FUTURE ENHANCEMENTS

Based on the transcript, here are recommended features to add:

1. **User Profiles**: Public pages showing user's published songs
2. **Song Pages**: Dedicated page per song with lyrics, prompts, comments
3. **Playlists**: Users can create and share playlists
4. **Search**: Search songs, users, genres
5. **Follow System**: Follow users and get notifications
6. **Profile Photos**: Upload and display user avatars
7. **Variable Song Count**: Let users choose how many songs to generate
8. **Model Parameters UI**: Expose more generation parameters
9. **Model Switching**: Support multiple music generation models

---

## TROUBLESHOOTING

### Common Issues

**Queue not processing**
- Check Inngest dev server running
- Verify INNGEST_EVENT_KEY in .env
- Check function registered in Inngest UI

**Modal timeout**
- Increase timeout in @modal.cls decorator
- Check GPU availability
- Review Modal logs for errors

**S3 upload fails**
- Verify AWS credentials in Modal secret
- Check bucket permissions
- Ensure bucket name correct

**Database connection errors**
- Verify Neon connection strings
- Check DIRECT_URL for migrations
- Ensure pooled URL for queries

**Authentication issues**
- Clear cookies and try again
- Check BETTER_AUTH_SECRET set
- Verify auth API route exists

---

## CONCLUSION

This plan follows the proven implementation path from the video tutorial. By following each phase, step, and micro-step with the provided tests, you'll recreate a fully functional AI Music Generation SaaS.

**Estimated Timeline**: 4-6 weeks (part-time)
- Phase 1-3 (Backend): 1-2 weeks
- Phase 4-6 (Frontend Infrastructure): 1 week
- Phase 7-8 (Core Features): 2 weeks
- Phase 9-10 (Deployment & Polish): 1 week

Remember:
- Test after each step
- Don't skip tests
- Use Inngest and Modal dashboards for debugging
- Start with short audio durations (30-60s) for testing
- Keep Modal GPU warm during active development

Good luck building your AI Music Generator! 🎵🚀
