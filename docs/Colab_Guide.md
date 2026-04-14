# SAGE UI — Colab Quickstart Guide

So you started the SAGE server in Google Colab, got your Ngrok link, and loaded the webpage. Welcome to the **SAGE Browser IDE**!

Right now, you are looking at the "Control Plane." The AI is essentially a blank slate. To get a "proper agent" that can chat with you, you need to use this interface to prepare data and train the model step-by-step.

Here is exactly what to do.

---

### Step 1: Open the Terminal & Download Data

To train a model, you need text. We recently added a 5-Billion-Token downloader, but we don't need all 5 billion for a quick test.

1. In the SAGE IDE, open the **CLI Terminal** (click the `>_` icon on the left sidebar, or press `Ctrl + \``).
2. Type the following command and press Enter to download a small 1% slice (~50 Million tokens):
   ```bash
   python debug/download_5b_tokens.py --output-dir data/raw --scale 0.01
   ```
3. Watch the terminal. It will take a few minutes to download the General Web, Code, Math, Wikipedia, and Synthetic datasets into `data/raw/`.

### Step 2: Train Your Tokenizer

The AI doesn't read English words; it reads "Tokens". It needs to learn its vocabulary from your downloaded data.

1. Click the **Presets** tab (the rocket icon 🚀) on the left sidebar.
2. Select **Tokenizer Train** from the dropdown menu.
3. Click the purple **Run Job** button.
4. A new panel will slide out showing you the live logs. Wait until it says `Job finished successfully`.

### Step 3: Fast-Pack Your Data (Sharding)

Training directly from text files is too slow for a GPU. We need to tokenize the text and pack it into high-speed Parquet "shards".

1. Go back to the **Presets** tab.
2. Select **Build Data Shards**.
3. Set the `shard_size` to `2048`.
4. Click **Run Job**.
5. Wait for the logs to finish. When done, your data is packed and ready!

### Step 4: Begin Training the AI

Now it's time to put the GPU to work.

1. Go back to the **Presets** tab.
2. Select **Training Run**.
3. You can leave the steps at the default (e.g., `20` for a smoke test, or change it to `2000` for a real micro-run). Make sure `disable_wandb` is checked so it doesn't ask for a Weights & Biases login.
4. Click **Run Job**.
5. The live log viewer will now stream training metrics. You will see `loss` going down and `tokens_per_second` showing how fast your Colab T4 GPU is churning through data.
6. The trainer automatically saves checkpoints (e.g., `ckpt_step_1000.pt`) into the `runs/` folder.

### Step 5: Chat with Your New Agent

Once the training has run for a decent amount of steps and a checkpoint is saved, the model is ready to talk!

1. Click the **Chat** tab (the speech bubble icon 💬) on the left sidebar.
2. Type a message like _"What is Python?"_ and hit Enter.
3. The UI will send this prompt to the backend, run inference using your newly trained checkpoint and tokenizer, and stream the generated response back to your screen.

_(Note: If you only trained for 20 steps, the AI will probably respond with random gibberish. Real reasoning requires thousands of steps over billions of tokens!)_

---

### Pro-Tips for the IDE

- **Command Palette:** Press `Ctrl + K` anywhere to quickly jump between tools.
- **Function Inspector:** You can click the Book 📖 icon on the right to browse the actual Python codebase from within the browser while your model trains.
- **Stop a stray training job:** Go to the **Jobs** panel (the clipboard icon) and click the red "Stop" button on any running task to free up your GPU.
