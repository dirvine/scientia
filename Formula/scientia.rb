class Scientia < Formula
  include Language::Python::Virtualenv

  desc "AI-powered knowledge exploration and management system"
  homepage "https://github.com/dirvine/scientia"
  url "https://github.com/dirvine/scientia/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "179f62a9f6a944e9698ec5f44b48079066ba0550bd618276eae90f3904dc588c"
  license "MIT"

  depends_on "python@3.10"
  depends_on "pytorch"
  depends_on "tesseract"  # For OCR support
  depends_on "poppler"    # For PDF processing
  depends_on "pkg-config" # Build dependency
  depends_on "chromadb"   # Vector store

  resource "transformers" do
    url "https://files.pythonhosted.org/packages/transformers/transformers-4.36.0.tar.gz"
    sha256 "YOUR_SHA256"
  end

  resource "sentence-transformers" do
    url "https://files.pythonhosted.org/packages/sentence-transformers/sentence-transformers-2.2.2.tar.gz"
    sha256 "YOUR_SHA256"
  end

  resource "PyMuPDF" do
    url "https://files.pythonhosted.org/packages/PyMuPDF/PyMuPDF-1.23.8.tar.gz"
    sha256 "YOUR_SHA256"
  end

  resource "python-docx" do
    url "https://files.pythonhosted.org/packages/python-docx/python-docx-1.0.1.tar.gz"
    sha256 "YOUR_SHA256"
  end

  resource "Pillow" do
    url "https://files.pythonhosted.org/packages/Pillow/Pillow-10.1.0.tar.gz"
    sha256 "YOUR_SHA256"
  end

  resource "pytesseract" do
    url "https://files.pythonhosted.org/packages/pytesseract/pytesseract-0.3.10.tar.gz"
    sha256 "YOUR_SHA256"
  end

  resource "streamlit" do
    url "https://files.pythonhosted.org/packages/streamlit/streamlit-1.29.0.tar.gz"
    sha256 "YOUR_SHA256"
  end

  resource "protobuf" do
    url "https://files.pythonhosted.org/packages/protobuf/protobuf-3.20.0.tar.gz"
    sha256 "YOUR_SHA256"
  end

  resource "watchdog" do
    url "https://files.pythonhosted.org/packages/watchdog/watchdog-3.0.0.tar.gz"
    sha256 "YOUR_SHA256"
  end

  resource "tqdm" do
    url "https://files.pythonhosted.org/packages/tqdm/tqdm-4.66.1.tar.gz"
    sha256 "YOUR_SHA256"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/numpy/numpy-1.24.0.tar.gz"
    sha256 "YOUR_SHA256"
  end

  resource "aiohttp" do
    url "https://files.pythonhosted.org/packages/aiohttp/aiohttp-3.9.1.tar.gz"
    sha256 "YOUR_SHA256"
  end

  resource "python-dotenv" do
    url "https://files.pythonhosted.org/packages/python-dotenv/python-dotenv-1.0.0.tar.gz"
    sha256 "YOUR_SHA256"
  end

  def install
    # Create virtual environment
    venv = virtualenv_create(libexec, "python3.10")

    # Install all Python dependencies
    resources.each do |r|
      r.stage do
        system libexec/"bin/pip", "install", "."
      end
    end

    # Install the package itself
    venv.pip_install_and_link buildpath

    # Create necessary directories
    (var/"scientia/knowledge_base").mkpath
    (var/"scientia/exports").mkpath

    # Add post-installation message
    ohai "Installation complete!"
    ohai "Run 'scientia run' to start the web interface"
  end

  def caveats
    <<~EOS
      Scientia has been installed with all required dependencies.
      
      To start using Scientia:
        scientia run     - Start the web interface
        scientia add     - Add documents to the knowledge base
        scientia --help  - Show all available commands

      Data is stored in:
        #{var}/scientia/knowledge_base
        #{var}/scientia/exports

      Note: The first run will download required AI models.
    EOS
  end

  test do
    system bin/"scientia", "--version"
  end

  # Service support for running in background
  service do
    run [opt_bin/"scientia", "run"]
    keep_alive true
    working_dir var/"scientia"
    log_path var/"log/scientia.log"
    error_log_path var/"log/scientia.error.log"
  end
end
