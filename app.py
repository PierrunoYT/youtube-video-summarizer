from flask import Flask, render_template, request, jsonify, Markup
from flask_cors import CORS
import re
import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import urlparse, parse_qs
import requests
from functools import lru_cache
from datetime import datetime, timedelta
import tempfile
import markdown2
import time
import yt_dlp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
if not load_dotenv():
    logger.error("No .env file found. Please create one from .env.example")
    raise RuntimeError("No .env file found")

# Validate required environment variables
required_env_vars = {
    'OPENAI_API_KEY': 'OpenAI API key is required for transcription',
    'OPENROUTER_API_KEY': 'OpenRouter API key is required for summarization',
}

missing_vars = [var for var, message in required_env_vars.items() if not os.getenv(var)]
if missing_vars:
    for var in missing_vars:
        logger.error(f"Missing {var}: {required_env_vars[var]}")
    raise RuntimeError("Missing required environment variables")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

logger.info("Starting YouTube Video Summarizer application...")
logger.info(f"Environment: SITE_URL={os.getenv('SITE_URL', 'http://127.0.0.1:5000')}")

# Initialize OpenAI client for Whisper API
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

# Initialize OpenRouter client for summarization
router_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY'),
    default_headers={
        "HTTP-Referer": os.getenv('SITE_URL', "http://127.0.0.1:5000"),
        "X-Title": os.getenv('SITE_NAME', "YouTube Video Summarizer")
    }
)

# Cache models for 1 hour to avoid frequent API calls
@lru_cache(maxsize=1)
def get_models():
    """Fetch available models from OpenRouter API with caching."""
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}
        )
        if response.ok:
            models = response.json().get('data', [])
            # Filter and format models
            formatted_models = {}
            for model in models:
                model_id = model.get('id')
                if model_id:
                    formatted_models[model_id] = {
                        "id": model_id,
                        "name": model.get('name', model_id),
                        "context_length": model.get('context_length', 4096),
                        "description": model.get('description', ''),
                        "pricing": model.get('pricing', {}),
                    }
            logger.info(f"Fetched {len(formatted_models)} models from OpenRouter API")
            return formatted_models
        logger.error(f"Failed to fetch models: {response.status_code}")
        return {}
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        return {}

def refresh_models_cache():
    """Force refresh the models cache."""
    get_models.cache_clear()
    return get_models()

@app.route('/api/models')
def list_models():
    """API endpoint to list all available models."""
    models = get_models()
    return jsonify(models)

@app.route('/api/models/search')
def search_models():
    """API endpoint to search models."""
    query = request.args.get('q', '').lower()
    models = get_models()
    
    if not query:
        return jsonify(models)
    
    # Filter models based on search query
    filtered_models = {
        model_id: model_data
        for model_id, model_data in models.items()
        if query in model_id.lower() or 
           query in model_data['name'].lower() or 
           query in model_data.get('description', '').lower()
    }
    
    return jsonify(filtered_models)

def check_api_limits():
    """Check rate limits and credits remaining for the OpenRouter API key."""
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}
        )
        if response.ok:
            data = response.json().get('data', {})
            return {
                'usage': data.get('usage', 0),
                'limit': data.get('limit'),
                'is_free_tier': data.get('is_free_tier', False),
                'rate_limit': data.get('rate_limit', {}),
                'generation_id': data.get('id')  # For tracking costs
            }
        return None
    except Exception as e:
        logger.error(f"Error checking API limits: {str(e)}")
        return None

def get_generation_stats(generation_id):
    """Get detailed stats about a specific generation including costs and tokens."""
    try:
        response = requests.get(
            f"https://openrouter.ai/api/v1/generation?id={generation_id}",
            headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}
        )
        if response.ok:
            return response.json().get('data', {})
        return None
    except Exception as e:
        logger.error(f"Error getting generation stats: {str(e)}")
        return None

def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    try:
        logger.info(f"Attempting to extract video ID from URL: {url}")
        parsed_url = urlparse(url)
        logger.info(f"Parsed URL components: hostname={parsed_url.hostname}, path={parsed_url.path}, query={parsed_url.query}")
        
        if parsed_url.hostname == 'youtu.be':
            video_id = parsed_url.path[1:].split('?')[0]  # Remove query parameters
            logger.info(f"Extracted video ID from youtu.be URL: {video_id}")
            return video_id
            
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                video_id = parse_qs(parsed_url.query)['v'][0]
                logger.info(f"Extracted video ID from youtube.com/watch URL: {video_id}")
                return video_id
            if parsed_url.path.startswith('/embed/'):
                video_id = parsed_url.path.split('/')[2]
                logger.info(f"Extracted video ID from embed URL: {video_id}")
                return video_id
                
        logger.error("URL format not recognized")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting video ID: {str(e)}")
        return None

def get_transcript(video_id):
    """Get transcript using OpenAI Whisper API."""
    try:
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                logger.info(f"Attempting to process video {video_id} (attempt {retry_count + 1}/{max_retries})")
                
                # Configure yt-dlp options
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                    'quiet': True,
                    'no_warnings': True,
                }
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Update options with temporary output directory
                    ydl_opts['outtmpl'] = os.path.join(temp_dir, '%(id)s.%(ext)s')
                    
                    # Download audio
                    logger.info(f"Downloading audio from video {video_id}...")
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        url = f'https://www.youtube.com/watch?v={video_id}'
                        ydl.download([url])
                        
                        audio_file = os.path.join(temp_dir, f'{video_id}.mp3')
                        
                        if not os.path.exists(audio_file):
                            raise Exception("Audio file not found after download")
                        
                        file_size = os.path.getsize(audio_file) / (1024*1024)
                        logger.info(f"Audio downloaded successfully. File size: {file_size:.2f} MB")
                        
                        if file_size > 25:
                            logger.warning("Audio file exceeds 25MB limit, will process in chunks")
                            # TODO: Implement chunked processing for large files
                            raise Exception("Audio file too large (>25MB). Please use a shorter video.")
                        
                        logger.info("Starting transcription with Whisper API...")
                        start_time = time.time()
                        
                        with open(audio_file, "rb") as audio:
                            transcription = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio,
                                response_format="text"
                            )
                        
                        transcription_time = time.time() - start_time
                        logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
                        logger.info(f"Transcript length: {len(transcription)} characters")
                        
                        return transcription
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {retry_count + 1} failed: {last_error}")
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)  # Exponential backoff
        
        # If we get here, all retries failed
        error_msg = f"Failed to process video after {max_retries} attempts. Last error: {last_error}"
        logger.error(error_msg)
        return None
        
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return None

def format_summary_as_markdown(summary_data):
    """Convert the summary data to a formatted markdown string."""
    # Format the title and summary
    markdown = []
    markdown.append(f"# {summary_data.get('title', 'Video Summary')}")
    markdown.append(summary_data['summary'].strip())
    
    # Add key points
    if summary_data.get('key_points'):
        markdown.append("## Key Points")
        for point in summary_data['key_points']:
            markdown.append(f"- {point.strip()}")
    
    # Add topics as tags
    if summary_data.get('topics'):
        markdown.append("## Topics")
        topics = [f"`{topic.strip()}`" for topic in summary_data['topics']]
        markdown.append(" ".join(topics))
    
    # Add generation stats
    if summary_data.get('stats'):
        markdown.append("<details>")
        markdown.append("<summary>Generation Statistics</summary>")
        stats = summary_data['stats']
        total_tokens = stats.get('tokens_prompt', 0) + stats.get('tokens_completion', 0)
        if total_tokens:
            markdown.append(f"- **Total Tokens:** {total_tokens:,}")
        if stats.get('total_cost'):
            markdown.append(f"- **Cost:** ${stats.get('total_cost', 0):.4f}")
        markdown.append("</details>")
    
    # Join with single newlines for proper markdown rendering
    return "\n".join(markdown)

def generate_summary(transcript, model_name):
    """Generate summary using OpenRouter API with selected model."""
    try:
        logger.info(f"Generating summary using model: {model_name}")
        logger.info(f"Transcript length: {len(transcript)} characters")
        start_time = time.time()
        
        # Get model info from the API
        models = get_models()
        model_info = models.get(model_name)
        
        if not model_info:
            logger.warning(f"Model {model_name} not found in available models, using default context length")
            model_info = {
                "id": model_name,
                "context_length": 4096  # Safe default
            }
        
        model_id = model_info["id"]
        context_length = model_info["context_length"]
        
        logger.info(f"Using model ID: {model_id} with context length: {context_length}")
        
        # Check API limits before making the request
        limits = check_api_limits()
        if limits and limits.get('limit') is not None and limits.get('usage', 0) >= limits['limit']:
            raise Exception("API credit limit reached")

        messages = []
        messages.append({
            "role": "system",
            "content": """You are a helpful assistant that creates structured summaries of video transcripts.
            Your response must be a valid JSON object with the following structure:
            {
                "title": "A concise title for the video",
                "summary": "A detailed summary of the content (do not include any markdown headers)",
                "key_points": ["point 1", "point 2", ...],
                "topics": ["topic 1", "topic 2", ...]
            }
            The summary should be plain text without any markdown headers or formatting."""
        })
        
        # For long transcripts, split into chunks if needed
        max_transcript_length = context_length - 1000
        if len(transcript) > max_transcript_length:
            chunk_size = max_transcript_length // 3
            transcript = transcript[:chunk_size] + "\n...[middle section omitted]...\n" + transcript[-chunk_size:]

        messages.append({
            "role": "user",
            "content": f"Please analyze this transcript and provide a structured summary:\n\n{transcript}"
        })

        logger.info(f"Making API request to {model_id}")
        response = router_client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=min(800, context_length // 4),
            temperature=0.7,
            response_format={"type": "json_object"},
            stream=False  # Changed to non-streaming for simpler handling
        )

        logger.info("Received API response")
        
        # Extract the response content
        if not response.choices or not response.choices[0].message:
            raise Exception("No response content received from API")
            
        content = response.choices[0].message.content
        logger.info(f"Raw API response content: {content}")

        try:
            summary_data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response as JSON: {e}")
            logger.error(f"Raw content: {content}")
            raise Exception("Failed to parse summary response")

        # Validate required fields
        required_fields = ['title', 'summary', 'key_points', 'topics']
        missing_fields = [field for field in required_fields if field not in summary_data]
        if missing_fields:
            raise Exception(f"Missing required fields in API response: {missing_fields}")
            
        # Extract usage statistics directly from the response object
        if hasattr(response, 'usage'):
            summary_data['stats'] = {
                'tokens_prompt': response.usage.prompt_tokens,
                'tokens_completion': response.usage.completion_tokens,
                'total_cost': response.usage.total_tokens * 0.000002,  # Approximate cost per token
                'cache_discount': 0
            }
            logger.info(f"Usage statistics: {summary_data['stats']}")
        else:
            logger.warning("No usage statistics available in the response")
            summary_data['stats'] = {
                'tokens_prompt': 0,
                'tokens_completion': 0,
                'total_cost': 0,
                'cache_discount': 0
            }
        
        generation_time = time.time() - start_time
        logger.info(f"Summary generated in {generation_time:.2f} seconds")
        logger.info(f"Summary data keys: {list(summary_data.keys())}")
        
        return summary_data

    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return None

def render_summary_html(summary_data):
    """Convert the summary data into a structured HTML output."""
    output = []
    # Title
    title = summary_data.get("title", "Video Summary")
    output.append(f'<h1 class="text-3xl font-bold mb-4">{title}</h1>')

    # Summary paragraph
    summary = summary_data.get("summary", "")
    output.append(f'<section class="mb-6"><p class="text-gray-700">{summary}</p></section>')

    # Key Points - filter out any empty entries and only render if there is at least one non-empty point
    key_points = [point for point in summary_data.get("key_points", []) if point.strip()]
    if key_points:
        output.append('<section class="mb-6">')
        output.append('<h2 class="text-2xl font-semibold mb-2">Key Points</h2>')
        output.append('<ul class="list-disc pl-5 text-gray-700">')
        for point in key_points:
            output.append(f'<li>{point}</li>')
        output.append('</ul>')
        output.append('</section>')

    # Topics Covered - filter out any empty entries and only render if there is at least one non-empty topic
    topics = [topic for topic in summary_data.get("topics", []) if topic.strip()]
    if topics:
        output.append('<section class="mb-6">')
        output.append('<h2 class="text-2xl font-semibold mb-2">Topics Covered</h2>')
        output.append('<div class="flex flex-wrap gap-2">')
        for topic in topics:
            # Adding a simple badge style for topics
            output.append(f'<span class="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">{topic}</span>')
        output.append('</div>')
        output.append('</section>')

    # Generation Statistics
    stats = summary_data.get("stats", {})
    if stats:
        total_tokens = stats.get('tokens_prompt', 0) + stats.get('tokens_completion', 0)
        output.append('<section class="mb-6">')
        output.append('<details class="bg-gray-50 p-4 rounded-lg">')
        output.append('<summary class="cursor-pointer font-semibold mb-2">Generation Statistics</summary>')
        output.append('<div class="text-sm text-gray-700">')
        output.append(f'<p>Total Tokens: {total_tokens:,}</p>')
        if stats.get('total_cost'):
            output.append(f'<p>Cost: ${stats.get("total_cost", 0):.4f}</p>')
        output.append('</div>')
        output.append('</details>')
        output.append('</section>')

    # Join HTML segments
    return "\n".join(output)

@app.route('/')
def home():
    # Get API limits and models to display on the frontend
    limits = check_api_limits()
    models = get_models()
    return render_template('index.html', models=models, api_limits=limits)

@app.route('/api-status')
def api_status():
    """Endpoint to check API status and limits."""
    try:
        limits = check_api_limits()
        if limits:
            return jsonify({
                'status': 'success',
                'data': {
                    'usage': limits.get('usage', 0),
                    'limit': limits.get('limit', 'Unlimited'),
                    'is_free_tier': limits.get('is_free_tier', False),
                    'rate_limit': limits.get('rate_limit', {
                        'requests': 0,
                        'interval': 'N/A'
                    })
                }
            })
        return jsonify({
            'status': 'error',
            'message': 'Could not fetch API status',
            'data': {
                'usage': 0,
                'limit': 'Unknown',
                'is_free_tier': False,
                'rate_limit': {
                    'requests': 0,
                    'interval': 'N/A'
                }
            }
        })
    except Exception as e:
        print(f"API Status Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'data': {
                'usage': 0,
                'limit': 'Error',
                'is_free_tier': False,
                'rate_limit': {
                    'requests': 0,
                    'interval': 'N/A'
                }
            }
        })

@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle video summarization requests."""
    try:
        data = request.get_json()
        # Handle both 'url' and 'video_url' for compatibility
        video_url = data.get('video_url') or data.get('url')
        model_name = data.get('model', 'gpt-3.5')
        
        logger.info(f"Received summarization request for URL: {video_url}")
        logger.info(f"Selected model: {model_name}")
        
        if not video_url:
            logger.error("No URL provided in request")
            return jsonify({'error': 'No URL provided'}), 400
            
        video_id = extract_video_id(video_url)
        if not video_id:
            logger.error("Invalid YouTube URL provided")
            return jsonify({'error': 'Invalid YouTube URL'}), 400
            
        logger.info(f"Extracted video ID: {video_id}")
        
        # Get transcript
        transcript = get_transcript(video_id)
        if not transcript:
            logger.error("Failed to get transcript")
            return jsonify({'error': 'Failed to get transcript'}), 500
            
        # Generate summary
        summary_data = generate_summary(transcript, model_name)
        if not summary_data:
            logger.error("Failed to generate summary")
            return jsonify({'error': 'Failed to generate summary'}), 500
            
        # Render structured HTML output using new helper
        html = render_summary_html(summary_data)
        
        logger.info("Summary generated and rendered successfully")
        
        # Update: Return only the HTML version to avoid duplicate output
        response_data = {
            'summary_html': html
        }
        
        logger.info(f"Response data keys: {list(response_data.keys())}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask development server...")
    app.run(debug=True) 