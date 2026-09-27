# Next-Word Prediction with a Stacked LSTM

A word-level language model that predicts the next word from the previous three, trained on L. M. Montgomery's novel *The Blue Castle*
(Project Gutenberg, public domain).

## How it works

- **Text:** about 403,000 characters, tokenized into 72,052 words, with a vocabulary of 8,413.
- **Training examples:** a sliding window of three words predicts the fourth (72,049 sequences).
- **Model:** Embedding(10) → LSTM(1000) → LSTM(1000) → Dense(1000, ReLU) → softmax over the vocabulary, trained 20 epochs with Adam.
  Training loss fell from 7.03 to 1.62.
- **Inference:** the notebook saves the model (`next_word.h5`) and tokenizer (`token.pkl`), then runs an interactive loop. Type a
  phrase and it suggests the next word.

## Limitations

- **No held-out text.** The loss is on training data only, so a falling loss partly reflects memorizing one novel. Holding out chapters
  and reporting perplexity or top-5 accuracy would show how well it generalizes.
- **Input handling.** Inputs shorter than three words, or with trailing spaces, raise an error in the prediction loop (visible in the
  notebook output). Stripping and padding the input fixes this.
- **Legacy format.** Keras now recommends the `.keras` format over HDF5 (`.h5`).

## Run it

Written for Google Colab (it uploads `blue_castle.txt` with `google.colab.files`). Locally, download the book's plain-text file from
Project Gutenberg and replace the upload cell with the file path.

Tools: TensorFlow / Keras, NumPy.

Notebook: [`next_word_prediction.ipynb`](next_word_prediction.ipynb)
