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