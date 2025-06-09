import os
import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from xml.parsers.expat import ExpatError
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Setup environment
load_dotenv()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
#os.environ['HUGGINGFACE_API_KEY'] = os.getenv("HUGGINGFACE_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Streamlit UI
st.set_page_config(page_title="YouTube QnA Assistant", layout="centered")
st.title("YouTube Video Q&A Assistant")
st.markdown("Ask questions based on a YouTube video.")

# Inputs
youtube_url = st.text_input("Enter YouTube Video URL")
llm_choice = st.selectbox("Choose LLM", [
    "qwen-qwq-32b",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "compound-beta",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    
    
])
user_question = st.text_area("Ask your question here")

# Get video ID from URL
def extract_video_id(url):
    parsed_url = urlparse(url)
    if 'youtube.com' in parsed_url.netloc:
        if parsed_url.path.startswith("/watch"):
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.path.startswith("/live/"):
            return parsed_url.path.split('/live/')[-1].split('?')[0]
    elif 'youtu.be' in parsed_url.netloc:
        return parsed_url.path.lstrip('/')
    return None

# Get transcript
@st.cache_data(show_spinner=False)
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "hi"])
        return " ".join(chunk["text"] for chunk in transcript_list)
    except (TranscriptsDisabled, NoTranscriptFound, ExpatError):
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

# Generate/reuse embeddings
@st.cache_resource(show_spinner="Creating and caching vector store...")
def get_vector_store(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# Prompt template
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.
    Always respond in English, regardless of the transcript language.

    {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)

# Format retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Main logic
if st.button("Get Answer"):
    if not youtube_url or not user_question:
        st.warning("Please provide both the YouTube URL and your question.")
    else:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL.")
        else:
            transcript = get_transcript(video_id)
            if not transcript:
                st.error("Transcript not available or couldn't be fetched.")
            else:
                vector_store = get_vector_store(transcript)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                llm = ChatGroq(model=llm_choice, temperature=0.2)

                chain = RunnableParallel({
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                }) | prompt | llm | StrOutputParser()

                with st.spinner("Thinking..."):
                    result = chain.invoke(user_question)

                st.markdown("### Answer:")
                st.markdown(
                    f"<div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px; "
                    f"border: 1px solid #ddd; font-size: 16px;'>{result}</div>",
                    unsafe_allow_html=True
                )