# YouTube Video Summarizer

This tool allows you to get AI-generated summaries of YouTube videos using their transcripts.

## Features
- Extract transcripts from YouTube videos using OpenAI's Whisper API
- Generate concise summaries using AI via the OpenRouter API
- Automatic audio transcription for videos without subtitles
- Simple web interface for easy use

## Prerequisites
- Python 3.8 or higher (Python 3.8 - 3.11 recommended)
- pip (Python package installer)
- FFmpeg (required for audio processing)
- OpenAI API key (for Whisper transcription via API)
- OpenRouter API key (for AI summarization via API)

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

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd youtube-video-summarizer
```

2. Set up a Python virtual environment.

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

3. Install dependencies:
```bash
# Upgrade pip first (recommended)
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:
```
# Required
OPENAI_API_KEY=your_openai_api_key_here  # For Whisper transcription via API
OPENROUTER_API_KEY=your_openrouter_api_key_here  # For AI summarization via API

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
- The application uses OpenAI's Whisper API for transcription via API
  - Maximum audio file size: 25MB
  - Supported formats: MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM
  - For longer videos, consider splitting into chunks (feature coming soon)
- Processing time depends on:
  - Video length (and resulting audio file size)
  - Chosen summarization model (via API)
  - API response times
- API usage costs are determined by:
  - Whisper API usage (OpenAI pricing)
  - Selected summarization model (OpenRouter pricing)
- The video must have closed captions/subtitles available; if not, the app automatically transcribes the audio using Whisper
  - First time running may take longer as it downloads the Whisper model (~2GB)
  - Transcription speed depends on your internet connection and hardware

## Troubleshooting

### FFmpeg Issues
- Windows: Ensure FFmpeg is in your PATH
- macOS: Try `brew doctor` to check Homebrew installation
- Linux: Check FFmpeg installation with `ffmpeg -version`

### Virtual Environment Issues
- Windows: If `activate` fails, try running PowerShell as administrator
- macOS/Linux: If `source` fails, check Python installation and permissions

### API Issues
- Verify your OpenRouter API key is correct
- Check API credit balance and rate limits at [OpenRouter Keys](https://openrouter.ai/keys)
- Ensure your environment variables are properly set

### Whisper Issues
- Ensure you have enough disk space (~2GB) for the Whisper model download (handled by the API)
- If you encounter memory issues, try modifying the code to use a smaller Whisper model variant

## License
MIT License © 2025 PierrunoYT

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.