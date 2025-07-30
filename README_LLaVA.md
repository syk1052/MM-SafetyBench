# LLaVA-1.5(7B) Querying System

This system allows you to query LLaVA-1.5(7B) with the MM-SafetyBench questions and collect real responses.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Setup
```bash
python test_llava_setup.py
```

This will:
- Check system requirements
- Download and load LLaVA-1.5(7B) model (first run only)
- Test basic text and image queries

## Usage

### Quick Start (5 questions for testing)
```bash
python add_model_responses.py
```

### Full Dataset Processing
Edit `add_model_responses.py` and change:
```python
max_questions=5  # Change to None for all questions
```

## Configuration Options

### Question Types
- `"original"`: Use original safety questions
- `"rephrased"`: Use rephrased questions designed for images (recommended)
- `"rephrased_sd"`: Use shorter rephrased questions

### Image Usage
- `use_images=True`: Create images with key phrases (recommended for LLaVA)
- `use_images=False`: Text-only queries

### Example Configuration
```python
add_real_llava_responses(
    use_images=True,           # Use images with key phrases
    question_type="rephrased", # Use questions designed for images
    max_questions=None         # Process all questions (97 total)
)
```

## Output

The system saves responses to:
```
data/MM-SafetyBench/question_and_answer/01-Illegal_Activitiy_with_llava_real_responses.json
```

## File Structure

```
MM-SafetyBench/
├── process_questions.py              # Create answer format structure
├── add_model_responses.py            # Query LLaVA and add responses
├── test_llava_setup.py              # Test LLaVA setup
├── requirements.txt                  # Dependencies
└── data/
    └── MM-SafetyBench/
        ├── processed_questions/      # Input questions
        ├── answer_format/           # Structured format
        └── question_and_answer/     # Final results
```

## Hardware Requirements

- **GPU recommended**: LLaVA-1.5(7B) is a 7B parameter model
- **RAM**: At least 16GB recommended
- **Storage**: ~14GB for model files (downloaded automatically)

## Safety Note

This system is designed for research purposes to evaluate model safety. The questions contain harmful content by design to test model robustness. Use responsibly and in accordance with your institution's ethics guidelines.

## Troubleshooting

### Out of Memory Errors
If you encounter GPU memory issues:

1. Reduce batch size in the model configuration
2. Use CPU instead (slower but works with less memory)
3. Process fewer questions at a time

### Model Download Issues
- Ensure stable internet connection
- Check HuggingFace token if required
- Models are cached in `~/.cache/huggingface/`

### Import Errors
```bash
pip install --upgrade transformers torch torchvision
``` 