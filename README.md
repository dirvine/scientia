# Scientia AI 🧠

Scientia is an AI-powered knowledge exploration and management system that combines a powerful language model with a local knowledge base to provide intelligent responses and insights.

## Features

- 🤖 Advanced AI Chat Interface
- 📚 Local Knowledge Base Management
- 🔍 Intelligent Document Processing
- 📊 Topic Analysis and Exploration
- 🔄 RAG (Retrieval-Augmented Generation)
- 📝 Multi-format Document Support (PDF, DOCX, Images)
- 🔒 Privacy-focused (all data stays local)

## Installation

### Using Homebrew (recommended)

1. Install Homebrew if you haven't already:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Tap the Scientia repository:
   ```
   brew tap scientia-ai/scientia
   ```

3. Install Scientia:
   ```
   brew install scientia
   ```

### From Source

1. Clone the repository:
   ```
   git clone https://github.com/scientia-ai/scientia.git
   cd scientia
   ```

2. Create a virtual environment and activate it:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install uv:
   ```
   pip install uv
   ```

4. Install dependencies using uv:
   ```
   uv sync 
   ```

5. Set up the configuration:
   ```
   cp config.example.yml config.yml
   ```
   Edit `config.yml` with your preferred settings.

5. Run the application:
   ```
   python src/main.py
   ```

### Web Interface

The web interface provides several features:

1. **Chat Interface**
   - Interactive conversations with AI
   - Knowledge base integration
   - Suggested follow-up questions
   - Topic analysis mode

2. **Knowledge Base**
   - Add text or documents
   - Search existing knowledge
   - Manage privacy levels
   - Tag and organize information

3. **Advanced Tools**
   - Knowledge visualization (coming soon)
   - Concept mapping (coming soon)
   - Source analysis (coming soon)

## System Requirements

- Python 3.10+
- 8GB RAM (16GB recommended)
- Local storage for knowledge base
- Optional: NVIDIA GPU for faster processing

## Core Dependencies

- PyTorch: Machine learning framework
- Transformers: Language model support
- ChromaDB: Vector database for knowledge storage
- Streamlit: Web interface
- Tesseract: OCR support

## Development Setup

1. Set up your environment:

## Contributing

1. Fork the repository from [https://github.com/dirvine/scientia](https://github.com/dirvine/scientia)
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE)

## Support

- 📖 [Documentation](https://github.com/dirvine/scientia#readme)
- 🐛 [Issue Tracker](https://github.com/dirvine/scientia/issues)
- 💬 [Discussions](https://github.com/dirvine/scientia/discussions)

---

Built with ❤️ using [Streamlit](https://streamlit.io/), [Hugging Face](https://huggingface.co/), and [ChromaDB](https://www.trychroma.com/)
