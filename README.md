# ChatPDF - Interactive PDF Document Chat

A Streamlit-based web application that allows users to chat with their PDF documents using AI. The application uses LangChain, OpenAI, and FAISS to create an interactive chat interface for PDF documents.

## Features

- Upload multiple PDF documents
- Interactive chat interface
- Document context-aware responses
- Support for large PDF files (up to 10MB each)
- Modern and responsive UI
- Error handling and user feedback
- Session management

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git (for deployment)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ChatPDF.git
cd ChatPDF
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Running Locally

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file (app.py)
6. Add your environment variables (OPENAI_API_KEY)
7. Deploy!

### Option 2: Heroku

1. Create a `Procfile` in the root directory:
```
web: streamlit run app.py
```

2. Create a `runtime.txt` file:
```
python-3.8.18
```

3. Deploy to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

4. Set your environment variables:
```bash
heroku config:set OPENAI_API_KEY=your_api_key_here
```

### Option 3: Docker

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

2. Build and run the container:
```bash
docker build -t chatpdf .
docker run -p 8501:8501 chatpdf
```

## Usage

1. Upload one or more PDF documents using the sidebar
2. Click "Process" to analyze the documents
3. Start asking questions about your documents
4. The AI will provide context-aware responses based on your documents

## Limitations

- Maximum file size: 10MB per PDF
- Maximum pages: 100 pages total
- Supported file format: PDF only

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [OpenAI](https://openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
