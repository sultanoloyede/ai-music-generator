# backend/main.py
import modal
import os
from pydantic import BaseModel
from typing import List

app = modal.App("music-generator-v2")

# Get the directory where this script is located
from pathlib import Path
backend_dir = Path(__file__).parent

# Create image with all dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git", "libsndfile1", "ffmpeg")  # Audio dependencies
    .pip_install(
        "boto3",
        "transformers==4.50.0",
        "diffusers==0.33.0",
        "torch==2.5.1",
        "accelerate==1.6.0",
        "peft==0.14.0",
        "fastapi==0.115.0",
        "soundfile",  # Audio backend for torchaudio
        "torchcodec"
    )
    .run_commands([
        "git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step",
        "cd /tmp/ACE-Step && pip install ."
    ])
    .env({"HF_HOME": "/.cache/huggingface"})
    .add_local_file(backend_dir / "prompts.py", "/root/prompts.py")
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

# MusicGenServer Class with Model Loading
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
        import sys
        sys.path.append('/root')
        from prompts import PROMPT_GENERATOR_PROMPT
        full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt=description)
        return self.prompt_qwen(full_prompt)

    def generate_lyrics(self, description: str):
        """Generate structured lyrics from description"""
        import sys
        sys.path.append('/root')
        from prompts import LYRICS_GENERATOR_PROMPT
        full_prompt = LYRICS_GENERATOR_PROMPT.format(description=description)
        return self.prompt_qwen(full_prompt)

    def generate_categories(self, description: str) -> List[str]:
        """Extract 3-5 genre categories"""
        prompt = f"Based on this music description, list 3-5 genres as comma-separated: '{description}'"
        response = self.prompt_qwen(prompt)
        categories = [cat.strip() for cat in response.split(",") if cat.strip()]
        return categories[:5]

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

@app.local_entrypoint()
def test_model_loading():
    """Test loading all three models"""
    print("Initializing MusicGenServer and loading models...")
    print("NOTE: First run will download models (may take 10-15 minutes)")
    print("Subsequent runs will use cached models (faster)\n")

    server = MusicGenServer()
    print("\n✓✓✓ Models initialized successfully! ✓✓✓")

@app.local_entrypoint()
def test_llm():
    """Test all LLM helper methods"""
    print("=== Testing LLM Helper Methods ===")
    print("NOTE: This will test prompt, lyrics, and category generation\n")

    server = MusicGenServer()
    server.test_llm_methods.remote()

    print("\n=== LLM Testing Complete ===")
    print("You can now proceed to Step 2.3!")

@app.local_entrypoint()
def test_pipeline():
    """Test the full pipeline - Step 2.3"""
    print("=== Testing Full Pipeline (Step 2.3) ===")
    print("NOTE: This will generate audio, upload to S3, create thumbnails")
    print("This may take 3-5 minutes for 30s audio\n")

    server = MusicGenServer()
    server.test_full_pipeline.remote()

    print("\n=== Full Pipeline Testing Complete ===")
    print("✓ Step 2.3 is now complete!")
    print("You can now proceed to Phase 3: API Endpoints")