#!/usr/bin/env python3
"""
Audio Transcriber using Groq's Whisper API

A robust audio transcription tool that converts audio files to text using
Groq's whisper-large-v3-turbo model with comprehensive error handling,
input validation, and security best practices.
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import ffmpeg
from groq import Groq
from dotenv import load_dotenv
from jinja2 import Template
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn


# Configure logging with Rich
console = Console()
logging.basicConfig(
    level=logging.INFO,  # Standard logging
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger("audio_transcriber")

# Supported audio formats based on Groq documentation
SUPPORTED_FORMATS = {'.flac', '.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.ogg', '.wav', '.webm'}

# File size limits (in bytes) - using conservative limit for reliability
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB conservative limit for chunking
CHUNK_DURATION = 300  # 5 minutes per chunk in seconds


class TranscriptionError(Exception):
    """Custom exception for transcription-related errors."""
    pass


class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


class AnalysisError(Exception):
    """Custom exception for analysis-related errors."""
    pass


def check_system_dependencies() -> None:
    """Check if required system dependencies are available."""
    if not shutil.which("ffmpeg"):
        raise ValidationError(
            "FFmpeg is not installed or not in PATH. Please install FFmpeg to continue.\n"
            "Visit: https://ffmpeg.org/download.html"
        )
    logger.info("✓ FFmpeg found")


def validate_api_key(api_key: str) -> None:
    """Validate Groq API key format."""
    if not api_key:
        raise ValidationError("GROQ_API_KEY environment variable is required")
    
    if not api_key.startswith("gsk_"):
        raise ValidationError("Invalid GROQ_API_KEY format. Key should start with 'gsk_'")
    
    if len(api_key) < 20:
        raise ValidationError("GROQ_API_KEY appears to be too short")
    
    logger.info("✓ API key validated")


def validate_input_file(file_path: Path) -> None:
    """Validate input audio file."""
    if not file_path.exists():
        raise ValidationError(f"File does not exist: {file_path}")
    
    if not file_path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")
    
    # Check file extension
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        supported = ', '.join(sorted(SUPPORTED_FORMATS))
        raise ValidationError(
            f"Unsupported file format: {file_path.suffix}\n"
            f"Supported formats: {supported}"
        )
    
    # Check if file is empty
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValidationError("File is empty")
    
    logger.info(f"✓ Input file validated: {file_size / (1024*1024):.1f}MB")


def convert_audio_to_flac(input_path: str, output_path: str) -> None:
    """
    Convert audio file to FLAC format with optimal settings for transcription.
    
    Uses 16kHz sample rate and mono channel as recommended by Groq.
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Converting to FLAC...", total=None)
            
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_path,
                    ar=16000,      # Sample rate 16kHz (optimal for Whisper)
                    ac=1,          # Mono channel
                    acodec='flac', # FLAC codec for lossless compression
                    map_metadata=-1  # Remove metadata for privacy
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            
            progress.update(task, completed=True)
            
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise TranscriptionError(f"Audio conversion failed: {error_msg}")


def transcribe_audio(file_path: str, api_key: str, model: str = "whisper-large-v3-turbo") -> str:
    """Transcribe audio file using Groq's Whisper API."""
    try:
        client = Groq(api_key=api_key)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Transcribing audio...", total=None)
            
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model=model,
                    response_format="verbose_json",
                    temperature=0.0  # Deterministic output
                )
            
            progress.update(task, completed=True)
        
        if not transcription.text:
            raise TranscriptionError("Transcription returned empty text")
            
        return transcription.text
        
    except Exception as e:
        if "authentication" in str(e).lower():
            raise TranscriptionError(
                "Authentication failed. Please check your GROQ_API_KEY."
            )
        elif "rate limit" in str(e).lower():
            raise TranscriptionError(
                "Rate limit exceeded. Please try again later."
            )
        elif "file too large" in str(e).lower():
            raise TranscriptionError(
                "File too large for API. Consider splitting into smaller segments."
            )
        else:
            raise TranscriptionError(f"Transcription failed: {str(e)}")


def needs_chunking(file_path: str) -> bool:
    """Check if file needs to be split into chunks."""
    if not os.path.exists(file_path):
        return False
    
    file_size = os.path.getsize(file_path)
    return file_size > MAX_FILE_SIZE


def split_audio_into_chunks(input_path: str, chunk_duration: int = CHUNK_DURATION) -> list[str]:
    """Split audio file into smaller chunks for processing."""
    chunks = []
    chunk_paths = []
    
    try:
        # Get audio duration
        probe = ffmpeg.probe(input_path)
        duration = float(probe['streams'][0]['duration'])
        
        logger.info(f"Audio duration: {duration/60:.1f} minutes, splitting into {chunk_duration/60} minute chunks")
        
        # Calculate number of chunks needed
        num_chunks = int(duration / chunk_duration) + (1 if duration % chunk_duration > 0 else 0)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Creating {num_chunks} audio chunks...", total=num_chunks)
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                chunk_path = f"/tmp/chunk_{i:03d}.flac"
                
                (
                    ffmpeg
                    .input(input_path, ss=start_time, t=chunk_duration)
                    .output(
                        chunk_path,
                        ar=16000,
                        ac=1,
                        acodec='flac',
                        map_metadata=-1
                    )
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
                
                # Only add chunk if it was created and has content
                if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 1000:  # At least 1KB
                    chunk_paths.append(chunk_path)
                
                progress.advance(task)
        
        logger.info(f"✓ Created {len(chunk_paths)} audio chunks")
        return chunk_paths
        
    except Exception as e:
        # Clean up any created chunks on error
        for chunk_path in chunk_paths:
            if os.path.exists(chunk_path):
                os.unlink(chunk_path)
        raise TranscriptionError(f"Failed to split audio: {e}")


def transcribe_chunks(chunk_paths: list[str], api_key: str, model: str = "whisper-large-v3-turbo") -> str:
    """Transcribe multiple audio chunks and combine results."""
    transcripts = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Transcribing {len(chunk_paths)} chunks...", total=len(chunk_paths))
        
        for i, chunk_path in enumerate(chunk_paths):
            try:
                progress.update(task, description=f"Transcribing chunk {i+1}/{len(chunk_paths)}...")
                
                client = Groq(api_key=api_key)
                
                with open(chunk_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        file=audio_file,
                        model=model,
                        response_format="verbose_json",
                        temperature=0.0
                    )
                
                if transcription.text:
                    transcripts.append(transcription.text)
                
                progress.advance(task)
                
            except Exception as e:
                logger.warning(f"Failed to transcribe chunk {i+1}: {e}")
                continue
    
    if not transcripts:
        raise TranscriptionError("Failed to transcribe any audio chunks")
    
    # Combine all transcripts
    combined_transcript = ' '.join(transcripts)
    logger.info(f"✓ Combined {len(transcripts)} transcripts")
    
    return combined_transcript


def save_transcript(transcript: str, output_file: Path) -> None:
    """Save transcript to file with error handling."""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcript)
            f.write('\n')  # Ensure file ends with newline
        
        logger.info(f"✓ Transcript saved: {output_file}")
        
    except PermissionError:
        raise TranscriptionError(
            f"Permission denied writing to: {output_file}\n"
            "Check directory permissions."
        )
    except OSError as e:
        raise TranscriptionError(f"Failed to save transcript: {e}")


def load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    template_path = Path(__file__).parent / "prompts" / f"{template_name}.txt"
    
    if not template_path.exists():
        raise AnalysisError(f"Prompt template not found: {template_path}")
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise AnalysisError(f"Failed to load prompt template: {e}")


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get a boolean value from environment variables."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def correct_transcript(transcript: str, api_key: str, model: str) -> str:
    """Use LLM to correct transcription errors in the transcript."""
    try:
        # Load the correction prompt template
        prompt_template = load_prompt_template("transcript_correction")
        
        # Render the prompt with the transcript
        template = Template(prompt_template)
        prompt = template.render(transcript=transcript)
        
        logger.info("🔧 Correcting transcript errors...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Applying transcript corrections...", total=None)
            
            client = Groq(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise transcript editor. Only correct obvious errors. Do not add, change, or fabricate any information. If you cannot make meaningful corrections, return the original transcript unchanged."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Zero temperature for maximum consistency
                max_tokens=8000,   # Allow for longer transcripts
                top_p=0.1,        # Very focused sampling
                frequency_penalty=0.2  # Reduce repetition
            )
            
            progress.update(task, completed=True)
        
        corrected_text = response.choices[0].message.content.strip()
        
        if not corrected_text:
            logger.warning("Transcript correction returned empty result, using original")
            return transcript
        
        # Basic validation - corrected text should be similar length
        length_ratio = len(corrected_text) / len(transcript)
        if length_ratio < 0.5 or length_ratio > 2.0:
            logger.warning(f"Corrected transcript length seems unusual (ratio: {length_ratio:.2f}), using original")
            return transcript
        
        logger.info("✓ Transcript corrections applied")
        return corrected_text
        
    except Exception as e:
        logger.warning(f"Transcript correction failed: {e}, using original transcript")
        return transcript


def analyze_transcript(transcript: str, api_key: str, model: str) -> str:
    """Perform comprehensive analysis of the conference transcript."""
    try:
        # Load the analysis prompt template  
        prompt_template = load_prompt_template("conference_analysis")
        
        # Render the prompt with the transcript
        template = Template(prompt_template)
        prompt = template.render(transcript=transcript)
        
        # Debug: Check if transcript is being passed correctly
        logger.info(f"Transcript preview: {transcript[:200]}...")
        logger.info(f"Template contains transcript placeholder: {'transcript' in prompt_template}")
        logger.info(f"Rendered prompt contains transcript: {'RAG' in prompt.lower() or 'retrieval' in prompt.lower()}")
        
        logger.info("🔍 Performing comprehensive analysis...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing conference content...", total=None)
            
            client = Groq(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst. Base your analysis STRICTLY on the provided transcript. Do not fabricate names, companies, technologies, or details not explicitly mentioned. If information is unclear or missing, state that explicitly rather than guessing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature but allow some analysis flexibility
                max_tokens=8192,   # Maximum allowed for this model
                top_p=0.3,        # Conservative sampling
                frequency_penalty=0.1,  # Slight penalty for repetition
                presence_penalty=0.1    # Encourage staying on topic
            )
            
            progress.update(task, completed=True)
        
        analysis = response.choices[0].message.content.strip()
        
        if not analysis:
            raise AnalysisError("Analysis returned empty result")
        
        logger.info("✓ Comprehensive analysis completed")
        return analysis
        
    except Exception as e:
        if "rate limit" in str(e).lower():
            raise AnalysisError("Rate limit exceeded during analysis. Please try again later.")
        elif "authentication" in str(e).lower():
            raise AnalysisError("Authentication failed during analysis. Check your API key.")
        else:
            raise AnalysisError(f"Analysis failed: {e}")


def create_env_example() -> None:
    """Create .env.example file if it doesn't exist."""
    env_example_path = Path(".env.example")
    if not env_example_path.exists():
        with open(env_example_path, 'w') as f:
            f.write("# Copy this file to .env and add your actual API key\n")
            f.write("GROQ_API_KEY=gsk_your_api_key_here\n")


def main():
    """Main application entry point."""
    # Load environment variables
    load_dotenv()
    
    # Create .env.example for reference
    create_env_example()
    
    # Parse command line arguments
    if len(sys.argv) != 2:
        console.print("Usage: python transcriber.py <audio_file_path>", style="red")
        console.print("\nExample: python transcriber.py audio.mp3", style="dim")
        sys.exit(1)
    
    input_file = Path(sys.argv[1]).resolve()
    
    try:
        # System and input validation
        console.print("🔍 Validating system and inputs...", style="blue")
        check_system_dependencies()
        
        api_key = os.getenv("GROQ_API_KEY")
        validate_api_key(api_key)
        
        # Get model configurations
        speech_model = os.getenv("SPEECH_MODEL", "whisper-large-v3-turbo")
        analysis_model = os.getenv("ANALYSIS_MODEL", "moonshotai/kimi-k2-instruct")
        
        # Get feature flags
        enable_correction = get_env_bool("ENABLE_TRANSCRIPT_CORRECTION", True)
        enable_analysis = get_env_bool("ENABLE_DETAILED_ANALYSIS", True)
        
        validate_input_file(input_file)
        
        # Processing
        console.print(f"\n🎵 Processing: {input_file.name}", style="green")
        
        # Create temporary FLAC file
        with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
            temp_flac_path = temp_file.name
        
        try:
            # Convert audio to optimal format
            convert_audio_to_flac(str(input_file), temp_flac_path)
            
            # Check if we need to split into chunks
            if needs_chunking(temp_flac_path):
                console.print("🔪 File is large, splitting into chunks...", style="yellow")
                
                # Split into chunks and transcribe each
                chunk_paths = split_audio_into_chunks(temp_flac_path)
                
                try:
                    transcript = transcribe_chunks(chunk_paths, api_key, speech_model)
                finally:
                    # Clean up chunk files
                    for chunk_path in chunk_paths:
                        if os.path.exists(chunk_path):
                            os.unlink(chunk_path)
            else:
                # File is small enough to process directly
                file_size = os.path.getsize(temp_flac_path)
                logger.info(f"✓ Converted file size OK: {file_size / (1024*1024):.1f}MB")
                transcript = transcribe_audio(temp_flac_path, api_key, speech_model)
            
            # Post-processing workflow
            console.print(f"\n🚀 Starting analysis pipeline...", style="blue")
            
            # Save original transcript first
            original_output_file = input_file.parent / f"{input_file.stem}_transcript_original.txt"
            save_transcript(transcript, original_output_file)
            
            # Step 1: Transcript Correction (if enabled)
            corrected_transcript = transcript
            corrected_output_file = None
            if enable_correction:
                corrected_transcript = correct_transcript(transcript, api_key, analysis_model)
                
                # Save corrected transcript if it's different from original
                if corrected_transcript != transcript:
                    corrected_output_file = input_file.parent / f"{input_file.stem}_transcript_corrected.txt"
                    save_transcript(corrected_transcript, corrected_output_file)
                else:
                    logger.info("No corrections were applied to transcript")
            else:
                logger.info("Transcript correction disabled")
            
            # Save the final transcript (corrected or original) as main output
            main_output_file = input_file.parent / f"{input_file.stem}_transcript.txt"
            save_transcript(corrected_transcript, main_output_file)
            
            # Step 2: Comprehensive Analysis (if enabled)
            analysis_output = None
            if enable_analysis:
                try:
                    analysis = analyze_transcript(corrected_transcript, api_key, analysis_model)
                    
                    # Save analysis
                    analysis_file = input_file.parent / f"{input_file.stem}_analysis.txt"
                    save_transcript(analysis, analysis_file)
                    analysis_output = analysis_file
                    
                except AnalysisError as e:
                    console.print(f"⚠️  Analysis failed: {e}", style="yellow")
                    logger.warning(f"Analysis failed: {e}")
            else:
                logger.info("Detailed analysis disabled")
            
            # Success summary
            word_count = len(corrected_transcript.split())
            console.print(f"\n✅ Processing Complete!", style="bold green")
            console.print(f"📝 Words transcribed: {word_count:,}")
            console.print(f"💾 Main transcript: {main_output_file}")
            console.print(f"📄 Original transcript: {original_output_file}")
            
            if corrected_output_file:
                console.print(f"🔧 Corrected transcript: {corrected_output_file}")
            
            if analysis_output:
                console.print(f"🔍 Analysis: {analysis_output}")
                console.print("\n📊 Analysis includes:", style="dim")
                console.print("   • Speaker profile & expertise", style="dim")
                console.print("   • Technology mentions & trends", style="dim")
                console.print("   • Key processes & methodologies", style="dim")
                console.print("   • Actionable insights & recommendations", style="dim")
            
            if word_count < 10:
                console.print("⚠️  Very short transcript - check audio quality", style="yellow")
            
        finally:
            # Always clean up temporary file
            if os.path.exists(temp_flac_path):
                os.unlink(temp_flac_path)
                logger.debug("Temporary file cleaned up")
    
    except (ValidationError, TranscriptionError, AnalysisError) as e:
        console.print(f"\n❌ Error: {e}", style="red")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n⏹️  Operation cancelled", style="yellow")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        console.print(f"\n💥 Unexpected error: {e}", style="red")
        console.print("Please report this issue if it persists.", style="dim")
        sys.exit(1)


if __name__ == "__main__":
    main()