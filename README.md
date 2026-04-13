# POS Tagging for Social Media

This project builds a part-of-speech tagging pipeline for informal social media text using a fine-tuned `distilbert-base-uncased` token classification model. The workflow is implemented in the notebook [POS_Tagging (1).ipynb](./POS_Tagging%20(1).ipynb) and is designed around Google Colab for training, evaluation, model export, and interactive inference.

## Overview

Traditional POS taggers often struggle with noisy text such as:

- slang
- abbreviations
- repeated characters
- hashtags, mentions, and URLs
- emojis and casual punctuation

This notebook adapts a transformer-based model to handle those patterns more robustly. It combines lightweight text cleaning, weak labeling heuristics, and DistilBERT fine-tuning for token-level POS prediction.

## What The Notebook Does

The notebook covers the full workflow end to end:

1. Installs dependencies in Colab.
2. Loads and prepares project/data paths.
3. Cleans social-media-style text.
4. Generates weak POS labels with `spaCy`.
5. Builds train, validation, and test datasets.
6. Fine-tunes `distilbert-base-uncased` for token classification.
7. Evaluates the model on held-out data.
8. Saves the trained model to Google Drive.
9. Reloads the saved model for inference.
10. Exports the model as a ZIP and includes a simple CLI-style prediction loop.

## Model And Approach

Core components used in the notebook:

- `transformers` for DistilBERT token classification
- `torch` for training and batching
- `spaCy` for linguistic processing and weak labeling support
- `nltk` TweetTokenizer for social-media-friendly tokenization
- `scikit-learn` for evaluation metrics
- `emoji` for emoji-aware preprocessing

The project uses a weak-labeling setup to bootstrap POS tags for noisy text, then fine-tunes a transformer model to improve generalization on social media language.

## Reported Results

The notebook already contains recorded metrics from a completed run:

- Validation accuracy: `0.9321`
- Validation F1: `0.9317`
- Test accuracy: `0.9035`
- Test F1: `0.9029`
- Test precision: `0.9035`
- Test recall: `0.9035`

These numbers come from the outputs saved in the notebook.

## Repository Structure

At the moment, the repository is notebook-centric:

```text
.
|-- POS_Tagging (1).ipynb
`-- README.md
```

If you later split the code into `src/`, `data/`, or `models/`, this README can be expanded to match that layout.

## Running The Project

### Option 1: Google Colab

This is the easiest way to run the current project.

1. Open the notebook in Google Colab.
2. Enable GPU if available.
3. Run the installation cells.
4. Update dataset paths if needed.
5. Run the training and evaluation cells.
6. Save the model to Google Drive using the notebook cells provided.

### Option 2: Local Jupyter Environment

Install the same dependencies locally:

```bash
pip install torch transformers spacy nltk scikit-learn tqdm pandas numpy emoji
python -m spacy download en_core_web_sm
```

Then open the notebook in Jupyter and update any Colab-specific paths such as:

- `/content/...`
- `/content/drive/MyDrive/...`

## Inference

The notebook includes examples for:

- loading a saved model from Google Drive
- packaging the model as a ZIP
- extracting and reloading the model
- predicting POS tags for custom input text

Example inputs shown in the notebook include short, informal social media phrases such as:

- `bro this was insane fr`
- `nah this ain t it`
- `yo that match was crazyyyy`
- `this is kinda sus ngl`

## Why This Project Matters

POS tagging on social media text is harder than standard news or formal writing because the language is shorter, noisier, and more creative. A stronger POS tagger for this domain can support downstream NLP tasks such as:

- sentiment analysis
- social media mining
- conversational AI preprocessing
- named entity recognition
- text normalization pipelines

## Current Limitations

- The repo currently stores the whole workflow in a single notebook.
- Some paths are Colab and Google Drive specific.
- The training data pipeline appears to rely on weak labeling rather than fully manual annotation.
- The notebook name could be cleaned up later for a more polished repository presentation.

## Next Improvements

Good next steps for this repository would be:

- rename the notebook to `POS_Tagging.ipynb`
- move reusable code into Python modules under `src/`
- add a `requirements.txt`
- document the dataset format explicitly
- save trained artifacts in a reproducible `models/` directory
- add a small standalone inference script

## Author

Created by `SaiRithvik07`.
