## **GPT-2 Fine-Tuning for Short Story Generation**

This repository contains the implementation for Project 8 of the Deep Learning and Neural Computation course, focusing on fine-tuning a pre-trained GPT-2 model for the task of short story generation. The project compares the performance of the base model versus the fine-tuned version using quantitative metrics: **Diversity**, **Fluency**, and **Coherence**.

The core implementation is in the `story_generation.ipynb` notebook.

## 1. Project Goals

The main objective of this exercise is to:
1.  **Fine-tune** a GPT-2 model (using the `gpt2` checkpoint) on a dataset of short stories.
2.  **Evaluate** the fine-tuned model's performance on text generation.
3.  **Compare** the output quality (in terms of diversity, fluency, and coherence) of the fine-tuned model against the base GPT-2 model.
4.  Investigate the impact of **early stopping** mechanisms.

## 2. Dataset

The project utilizes the **TinyStories** dataset, which consists of synthetic, simple stories suitable for training small language models to generate coherent narratives.

*   **Dataset Source:** `roneneldan/TinyStories` from the Hugging Face Datasets library.
*   **Preprocessing:** To reduce training time for demonstration purposes, a small fraction (approximately **2%**) of the original dataset was sampled for both the training and validation splits.

## 3. Model and Fine-Tuning

*   **Base Model:** `gpt2` (GPT-2 small)
*   **Library:** Hugging Face `transformers` and `datasets`.
*   **Training API:** `Trainer` API for simplified fine-tuning.
*   **Hyperparameters:**
    *   `num_train_epochs`: 5
    *   `learning_rate`: 5e-5
    *   `per_device_train_batch_size`: 4
    *   `load_best_model_at_end`: `True` (used for early stopping implementation)

## 4. Installation and Setup

To run the notebook locally, you need Python and the following libraries:

```bash
pip install transformers datasets torch accelerate sentence-transformers
```

## 5. Evaluation Metrics

The performance of the generated text is quantitatively evaluated using three key metrics:

| Metric | Calculation Method | Description |
| :--- | :--- | :--- |
| **Fluency** | **Perplexity (PPL)** | Measures how well the language model predicts the text. Lower PPL indicates higher fluency and better statistical fit to the language/domain. |
| **Diversity** | **Type-Token Ratio (TTR)** & **Entropy** | TTR measures lexical variation (unique words / total words). Entropy measures the randomness and variety of word distribution. Higher values suggest more diverse vocabulary. |
| **Coherence** | **Inter-Sentence Cosine Similarity** | Uses a Sentence Transformer (`all-MiniLM-L6-v2`) to embed adjacent sentences. The average cosine similarity between these embeddings serves as a proxy for how semantically connected the sentences are. Higher values indicate better coherence. |

## 6. Results and Analysis (Answering Assignment Questions)

### **Q1: Comparison of Generated Text Quality (Base vs. Fine-Tuned)**

The fine-tuning process resulted in a significant improvement in fluency and coherence, specializing the model for the short, simple narrative style of the TinyStories dataset.

#### **Average Evaluation Metrics Comparison**

A comparison of the average metrics calculated across 6 generated stories from both models:

| Metric | Base GPT-2 (Average) | Fine-Tuned GPT-2 (Average) | Observed Change |
| :--- | :--- | :--- | :--- |
| **Fluency (Perplexity)** | $\approx 17.2$ | $\mathbf{\approx 5.3}$ | **Significant Improvement (Lower)** |
| **Coherence** | $\approx 0.36$ | $\mathbf{\approx 0.38}$ | Slight Improvement |
| **Diversity (TTR)** | $\approx 0.70$ | $\approx 0.69$ | Maintained (Slight decrease is expected when specializing) |

#### **Qualitative Changes in Output**

| Quality Aspect | Base GPT-2 Output (Examples from Notebook) | Fine-Tuned GPT-2 Output (Examples from Notebook) |
| :--- | :--- | :--- |
| **Story Structure** | Often incoherent, rambling, or abruptly ending. Includes real-world references (e.g., website links) inappropriate for "TinyStories." | Consistently produces simple, multi-sentence stories with clear characters and actions (e.g., "Lily saw a big box... She climbed up the branches..."). |
| **Vocabulary/Tone** | Complex, dense, or overly dramatic words. | Simple, repetitive, and child-friendly language, directly matching the style of the TinyStories data. |
| **Fluency** | High Perplexity (e.g., $18.5$): Indicates the model is less confident in its word choices, leading to more jarring syntax. | Low Perplexity (e.g., $4.7$): Indicates high confidence and smooth, grammatically correct sentences that fit the target distribution. |

**Conclusion for Q1:** Fine-tuning successfully shifted the model's generation style from generic, large-scale web text towards the specific distribution of short, simple, and coherent stories, as evidenced by the dramatic drop in Perplexity (Fluency).

### **Q2: Generated Text Coherence Analysis**

**New Text Sample (Fine-Tuned Model):**

```
Once upon a time in a dark and feary forest, there was a curious little girl named Lily. She liked to explore the forest every day.

One morning, Lily saw a big box that she could play with. She was excited to see what it was! She climbed up the branches, carefully picked it out of the ground and carefully removed it from her hands. It smelled sweet and shiny.

Lily's mom told her that it was a special box, so Lily was very excited. The mom explained that many things can be made with other things. Lily was so excited, she had even made a jar for the box!

Her mom went outside to play with the weird things she was able to make. They
```

**Coherence Analysis:**
The generated story demonstrates good intra-story coherence. The narrative progresses logically: Lily explores the forest $\rightarrow$ finds a box $\rightarrow$ interacts with her mom about the box $\rightarrow$ continues playing. The sentences flow smoothly, and the vocabulary (e.g., "curious little girl," "big box," "smelled sweet and shiny") is consistent with a children's story. The average Sentence Similarity (Coherence) for this model ($\approx 0.38$) confirms that the thematic flow between sentences is stronger than the base model.

### **Q3: Impact of Early Stopping (Using `load_best_model_at_end`)**

The `Trainer` API was configured with `load_best_model_at_end=True` and `save_total_limit=2`, which implements an early stopping mechanism.

*   **Mechanism:** During training, the validation loss is monitored at regular steps. The model checkpoint associated with the *lowest* validation loss is saved.
*   **Impact on Performance:** By ensuring the best model (the one with the lowest validation loss) is loaded at the end of training, we mitigate the risk of **overfitting** to the specific training data. In the provided training run, the validation loss drops from the start but starts to plateau around step 2100. Loading the best model prevents the final, potentially overfit, state of the model from being used, guaranteeing the saved model offers the best generalization performance to unseen data (like the texts generated in Q2). This improves the robustness and generalization of the fine-tuned GPT-2.
