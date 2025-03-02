# LLM RAG Implementation

A Python implementation of Retrieval-Augmented Generation (RAG) using local LLM models. This project combines the power of LLaMA models with document retrieval to generate more accurate and context-aware responses, with a specific focus on financial market analysis and signal generation.

## Features
- Local LLM implementation using LLaMA models
- Document retrieval and embedding using sentence-transformers
- Vector storage using ChromaDB
- GPU acceleration support (CUDA 12.1)
- Configurable through JSON configuration files
- Financial market sentiment analysis and signal generation
- Multi-source data integration (news, Reddit, market analysis)

## Financial Analysis Capabilities
- Automated sentiment analysis for financial securities
- Signal generation from natural language responses
- Multi-factor analysis of market drivers
- Sentiment scoring system (-1 to 1 scale)
  - Positive signals (0.45 to 1.0): Bullish trends, growth, appreciation
  - Neutral signals (around 0): Stability, volatility
  - Negative signals (-1.0 to -0.45): Bearish trends, decline, depreciation
- Integration of multiple data sources:
  - Market news and analysis
  - Social media sentiment (Reddit)
  - Trading analysis
  - Price trends and outlooks

## Requirements
- Python with CUDA support (for GPU acceleration)
- Required packages listed in `requirements.txt`

## Installation

1. Set up Conda environment with CUDA support:
```bash
conda install cudatoolkit=12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c faiss faiss-cpu
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the RAG implementation using:
```bash
python run.py <config_file_name>
```

Where `<config_file_name>` corresponds to a JSON configuration file in the `config` directory.

### Signal Generation Process
1. The system collects data from multiple sources about specified securities
2. LLM processes the information through various sentiment queries
3. Responses are parsed into numerical signals (-1 to 1)
4. Signals are aggregated and analyzed for:
   - Average sentiment per security
   - Signal count and distribution
   - Non-zero signal analysis
5. Results are stored in JSON format for further analysis or integration

## Project Structure
- `config/` - Configuration files
- `models/` - Model storage
- `src/` - Source code
  - `parse_signals.py` - Signal generation from LLM outputs
  - `queries.py` - Financial analysis query templates
- `output/` - Generated outputs and data
- `run.py` - Main execution script
- `run_local_llm_rag.ipynb` - Jupyter notebook implementation

## Output
Results are stored in the `output` directory, organized by:
- `data/` - Processed data and signals
- `html/` - HTML outputs
- Logs are stored in the `logs` directory with timestamps

## Signal Interpretation
The system generates numerical signals that can be interpreted as follows:
- 1.0: Strong positive signal (e.g., "extremely likely" to increase)
- 0.75: Moderately positive signal (e.g., "very likely" to increase)
- 0.0: Neutral signal (e.g., "stable" or "fluctuate")
- -1.0: Strong negative signal (e.g., "decrease" or "bearish trend")