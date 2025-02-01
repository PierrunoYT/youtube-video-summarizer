# YouTube Video Summarizer

This tool allows you to get AI-generated summaries of YouTube videos using their transcripts.

## Features
- Extract transcripts from YouTube videos using OpenAI's Whisper API
- Generate concise summaries using AI
- Support for multiple AI models via OpenRouter
- Automatic audio transcription for videos without subtitles
- Simple web interface for easy use

## Prerequisites
- Python 3.8 or higher (Python 3.8 - 3.11 recommended for best compatibility)
- pip (Python package installer)
- FFmpeg (required for audio processing)
- OpenAI API key (for Whisper transcription)
- OpenRouter API key (for AI summarization)
- For GPU acceleration:
  - Windows/Linux: NVIDIA GPU with CUDA support (recommended for faster transcription)
    - NVIDIA drivers version 525.60.13 (Linux) or 527.41 (Windows) or newer
    - CUDA 11.8 compatible GPU (most RTX 20xx, 30xx, 40xx series cards)
  - Mac: Apple Silicon (M1/M2) for Metal acceleration support
    - Intel Macs will run in CPU-only mode

### Installing FFmpeg

#### Windows
1. Download FFmpeg from [FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases)
2. Extract the ZIP file
3. Add the `bin` folder to your system's PATH environment variable
4. Verify installation: `ffmpeg -version`

#### macOS
Using Homebrew:
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

### GPU Support Setup

#### Windows/Linux (NVIDIA GPUs)
1. Visit [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Select your GPU model and download the latest driver
3. Run the installer and follow the instructions
4. Verify installation: `nvidia-smi`

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install nvidia-driver-525  # or newer version
```

#### macOS
- Apple Silicon (M1/M2) Macs: No additional setup needed, Metal support is built-in
- Intel Macs: Will run in CPU-only mode, no GPU acceleration available

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd youtube-video-summarizer
```

2. Set up Python virtual environment:

First, make sure you have the right Python version installed:
```bash
python --version  # Should show Python 3.8 or higher
```

#### Windows
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (choose one based on your shell):
# For Command Prompt:
venv\Scripts\activate.bat
# For PowerShell:
venv\Scripts\Activate.ps1
# For Git Bash:
source venv/Scripts/activate

# Verify Python version in venv:
python --version
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify Python version in venv:
python --version
```

If you see a "command not found" error, make sure Python and venv are installed:
- Ubuntu/Debian: `sudo apt install python3-venv`
- macOS: `brew install python@3.11`  # or your preferred version
- Windows: Download Python from python.org and check "Add Python to PATH" during installation

3. Install dependencies:
```bash
# Upgrade pip first (recommended)
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

Note on GPU Support:
- Windows/Linux with NVIDIA GPUs: PyTorch 2.1.0 with CUDA 11.8 support will be installed
- Apple Silicon Macs: PyTorch will use Metal Performance Shaders (MPS) for acceleration
- Intel Macs: PyTorch will run in CPU-only mode

If you encounter GPU-related issues:
- Windows/Linux (NVIDIA):
  - Verify your NVIDIA drivers are up to date
  - Check CUDA compatibility with `python -c "import torch; print(torch.cuda.is_available())"`
  - For NCCL errors, set this environment variable: `NCCL_P2P_DISABLE=1`
- Apple Silicon Macs:
  - Verify MPS availability: `python -c "import torch; print(torch.backends.mps.is_available())"`

4. Create a `.env` file in the root directory:
```
# Required
OPENAI_API_KEY=your_openai_api_key_here  # For Whisper transcription
OPENROUTER_API_KEY=your_openrouter_api_key_here  # For AI summarization

# Optional - defaults shown
SITE_URL=http://127.0.0.1:5000
SITE_NAME=YouTube Video Summarizer

# Optional - for custom Whisper model path
WHISPER_MODEL_PATH=/path/to/whisper/model  # If not set, will download to default location
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Usage
1. Copy a YouTube video URL
2. Paste it into the web interface
3. Select your preferred AI model
4. Click "Get Summary" to generate a summary

## Notes
- The application uses OpenAI's Whisper API for transcription
  - Maximum audio file size: 25MB
  - Supported formats: MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM
  - For longer videos, consider splitting into chunks (feature coming soon)
- Processing time depends on:
  - Video length (and resulting audio file size)
  - Chosen AI model for summarization
  - API response times
- API usage costs are determined by:
  - Whisper API usage (OpenAI pricing)
  - Selected summarization model (OpenRouter pricing)
- The video must have closed captions/subtitles available
- If subtitles are not available, the app will automatically transcribe the audio using Whisper
  - First time running may take longer as it downloads the Whisper model (~2GB)
  - Transcription speed depends on your hardware:
    - NVIDIA GPUs: Fastest with CUDA acceleration
    - Apple Silicon (M1/M2): Good performance with Metal acceleration
    - CPU only (Intel Macs/other): Slower but functional
- Processing time may vary depending on video length and chosen AI model

## Troubleshooting

### GPU/PyTorch Issues
- Windows/Linux (NVIDIA):
  - Run `python -c "import torch; print(torch.cuda.is_available())"` to verify CUDA setup
  - Check GPU detection: `nvidia-smi`
  - Ensure you have compatible NVIDIA drivers installed
  - For NCCL errors, try setting: `export NCCL_P2P_DISABLE=1`
- Mac:
  - Apple Silicon: Run `python -c "import torch; print(torch.backends.mps.is_available())"` to verify Metal support
  - Intel Macs: Will run in CPU-only mode, expect slower performance

### FFmpeg Issues
- Windows: Ensure FFmpeg is in your PATH
- macOS: Try `brew doctor` to check Homebrew installation
- Linux: Check FFmpeg installation with `ffmpeg -version`

### Virtual Environment Issues
- Windows: If `activate` fails, try running PowerShell as administrator
- macOS/Linux: If `source` fails, check Python installation and permissions

### API Issues
- Verify your OpenRouter API key is correct
- Check API credit balance and rate limits at https://openrouter.ai/keys
- Ensure your environment variables are properly set

### Whisper Issues
- Ensure you have enough disk space (~2GB) for the Whisper model
- Performance expectations:
  - NVIDIA GPU: Fastest transcription with CUDA
  - Apple Silicon: Good performance with Metal
  - CPU only (Intel Macs/other): Functional but slower
- If you encounter memory issues, try using a smaller Whisper model by modifying the code

## License
MIT License © 2025 PierrunoYT

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.