YouTube Video Q&A Assistant
This is a Streamlit-based web application that allows users to ask questions about the content of a YouTube video by leveraging its transcript. The application fetches the video transcript using the youtube-transcript-api, processes it with LangChain for semantic search, and generates answers using a selected large language model (LLM) from Grok.
Features

YouTube Transcript Retrieval: Extracts transcripts from YouTube videos (supports English and Hindi).
Semantic Search: Uses FAISS and HuggingFace embeddings to retrieve relevant transcript segments.
LLM-Powered Answers: Answers user questions based on the transcript using a selected Grok model.
User-Friendly Interface: Built with Streamlit for an intuitive web-based experience.
Customizable LLM: Choose from multiple Grok models for generating responses.

Prerequisites

Python 3.8 or higher
A Grok API key (available from Grok Website)
A HuggingFace API token (optional, for embeddings)
A YouTube video with available transcripts (English or Hindi)

Installation

Clone the Repository:
git clone https://github.com/Subhajeet_Khawas/youtube-qna-assistant.git
cd youtube-qna-assistant


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Create a requirements.txt file with the following content:
streamlit==1.29.0
youtube-transcript-api==0.6.2
langchain==0.3.0
langchain-community==0.3.0
langchain-huggingface==0.1.0
langchain-groq==0.2.0
faiss-cpu==1.8.0
python-dotenv==1.0.1

Then run:
pip install -r requirements.txt


Set Up Environment Variables:Create a .env file in the project root with the following:
GROQ_API_KEY=your_grok_api_key
HF_TOKEN=your_huggingface_api_token

Replace your_grok_api_key and your_huggingface_api_token with your actual API keys.


Usage

Run the Application Locally:
streamlit run app.py

This will launch the app in your default web browser at http://localhost:8501.

Interact with the App:

Enter a YouTube video URL (e.g., https://www.youtube.com/watch?v=video_id).
Select a Grok model from the dropdown (e.g., gemma2-9b-it).
Input your question about the video content.
Click Get Answer to retrieve the response based on the video's transcript.



Deployment on Streamlit Cloud

Push to GitHub:

Ensure your code, requirements.txt, and .env (for local testing) are in your GitHub repository.
Do not include the .env file in the repository for security reasons. Instead, configure environment variables in Streamlit Cloud.


Deploy to Streamlit Cloud:

Log in to Streamlit Cloud.
Create a new app and link it to your GitHub repository.
Specify the main script as app.py.
Add environment variables (GROQ_API_KEY and HF_TOKEN) in the Streamlit Cloud app settings under "Secrets."


Troubleshooting Deployment:

If you encounter errors like Error fetching transcript: no element found: line 1, column 0, ensure the YouTube video has transcripts enabled.
Verify that package versions in requirements.txt match your local environment.
Check Streamlit Cloud logs for detailed error messages.



Known Issues

Transcript Errors: Some YouTube videos may not have transcripts available, or the youtube-transcript-api may fail due to network restrictions on Streamlit Cloud. Try using videos with confirmed transcript availability.
API Limits: The Grok API and YouTube Transcript API may have rate limits. Monitor usage to avoid hitting quotas.
Model Availability: Ensure the selected Grok model is supported by your API key.

Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bug fixes.


Built with Streamlit, LangChain, and Grok.
Powered by YouTube Transcript API and HuggingFace embeddings.

For questions or support, contact [subhkhawas044@gmail.com] or open an issue on GitHub.
