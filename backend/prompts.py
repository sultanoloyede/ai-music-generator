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