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
        "transformers==4.50.0",
        "diffusers==0.33.0",
        "torch==2.5.1",
        "accelerate==1.6.0",
        "peft==0.14.0",
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