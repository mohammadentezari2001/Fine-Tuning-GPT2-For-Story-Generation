# Fine-Tuning GPT-2 for Short Story Generation

This project demonstrates how to fine-tune **GPT-2** on the **TinyStories** dataset to generate short, coherent stories. The workflow is implemented in a Jupyter Notebook and covers data loading, preprocessing, model training with early stopping, evaluation, and text generation.

## Project Overview

The goal of this project is to:

* Fine-tune a pretrained **GPT-2** language model
* Use a subset of the **TinyStories** dataset for faster experimentation
* Generate short, creative stories after training

This project is ideal for learning the basics of **language model fine-tuning** using the Hugging Face ecosystem.

## Model & Dataset

* **Model:** GPT-2 (from Hugging Face Transformers)
* **Dataset:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
* **Training Size:** 5% of the dataset (sampled for efficiency)

## Technologies Used

* Python
* Jupyter Notebook
* Hugging Face Transformers
* Hugging Face Datasets
* PyTorch
* Accelerate

## Project Structure

```
.
├── story_generation.ipynb   # Main notebook for training and generation
├── README.md                # Project documentation
```

## Setup & Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/mohammadentezari2001/Fine-Tuning-GPT2-For-Story-Generation.git
   cd Fine-Tuning-GPT2-For-Story-Generation
   ```

2. Install dependencies:

   ```bash
   pip install transformers datasets torch accelerate
   ```

> It is recommended to use a virtual environment or Google Colab for GPU support.

## How It Works

1. **Load Dataset** – TinyStories is loaded using the `datasets` library
2. **Sampling** – Only 5% of the data is used to reduce training time
3. **Tokenization** – GPT-2 tokenizer with EOS token padding
4. **Training** – Fine-tuning GPT-2 with early stopping
5. **Evaluation** – Model performance is evaluated during training
6. **Generation** – The trained model generates short stories from prompts

## Acknowledgements

* Hugging Face 🤗
* TinyStories Dataset by Ronen Eldan
