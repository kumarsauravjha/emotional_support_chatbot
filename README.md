# 🤖 Emotional Support Chatbot (T5-based)

A transformer-powered emotional support chatbot fine-tuned on the [ESConv](https://huggingface.co/datasets/taesiri/ESConv) dataset. This project adapts a T5 model (small variant) to generate empathetic, strategy-tagged responses in human–AI conversations for mental health support.

---

## Features

- Generates emotionally supportive responses
- Trained with strategy tags like `[Affirmation and Reassurance]`, `[Questions]`, etc.
- Interactive Gradio-based chatbot UI
- Fully trainable with custom `.tsv` formatted datasets
- TensorBoard-compatible training logs

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/emotional_support_chatbot.git
cd emotional_support_chatbot
```

To execute on your machine, first download the contents of the folder "t5_strategy_full". 
After that just execute EmotionalSupport.py from your terminal, open the generated gradio link and talk to the bot. 
