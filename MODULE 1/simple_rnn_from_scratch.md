# Comprehensive Guide: Building a Simple RNN from Scratch

This document explains the end-to-end process of building a Simple Recurrent Neural Network (RNN) using only NumPy. It breaks down the code step-by-step so you understand **what** each part does, **why** it is necessary, and **how** it connects to the core concepts of sequence modeling.

---

## 1. Imports and Setup

**What this does**: We import NumPy for all the mathematical operations and Matplotlib for visualizing the results. We also set a random seed to ensure our results are reproducible.
**Why**: Since we are building the neural network from scratch without libraries like TensorFlow or PyTorch, NumPy is essential for handling the matrix multiplications (the core of neural networks).

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)
print("✅ Libraries loaded. NumPy:", np.__version__)
print("=" * 50)
print("🔁 Simple RNN — From Scratch")
print("   Slide 15: RNNs (Early 2000s)")
print("=" * 50)
```

---

## 2. Dataset and Vocabulary Building

**What this does**: We define a simple text string that we want the RNN to learn. We then find all unique characters to build a vocabulary, and create mappings from characters to numerical indices (and vice versa).
**Why**: Neural networks cannot read raw text. They need numbers. By creating a vocabulary and mapping each character to a unique index, we prepare the data to be converted into numerical vectors.

```python
text = "hello generative ai world"

chars      = sorted(set(text))
vocab_size = len(chars)

char_to_idx = {ch: i  for i,  ch in enumerate(chars)}
idx_to_char = {i:  ch for ch, i  in char_to_idx.items()}

print("📝 Text       :", text)
print("🔤 Vocabulary :", "".join(chars))
print(f"📊 Vocab size : {vocab_size}")
print(f"📏 Sequence   : {len(text)} characters")
```

---

## 3. Data Preprocessing (One-Hot Encoding)

**What this does**: We define a function to convert a character index into a "one-hot vector" (an array of zeros with a single `1` at the character's index). We then encode our entire input text `X` and define our targets `Y_idx` (the next character we want to predict).
**Why**: One-hot encoding is how we feed individual characters into the neural network so it can process them mathematically.

```python
def one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros((size, 1))
    v[idx] = 1.0
    return v

X     = [one_hot(char_to_idx[c], vocab_size) for c in text]
Y_idx = [char_to_idx[text[i + 1]] for i in range(len(text) - 1)]

print("\n✅ Data ready.")
print(f"   Inputs  : {len(X)} one-hot vectors, shape ({vocab_size}, 1)")
print(f"   Targets : {len(Y_idx)} next-character indices")
```

---

## 4. The RNN Architecture

**What this does**: This is the heart of the model. We define a class `SimpleRNN` that sets up the weight matrices (`Wxh`, `Whh`, `Why`) and biases (`bh`, `by`). It includes a `forward` method to process sequences step-by-step, a `train_step` to calculate gradients and update weights (using Stochastic Gradient Descent), and a `generate` function to produce new text.
**Why**: This manually implements the core RNN equation: `h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)`. It shows exactly how the previous hidden state (`h_{t-1}`) acts as the "memory" that is combined with the current input (`x_t`).

```python
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.05):
        self.H  = hidden_size
        self.lr = lr
        s = 0.1

        self.Wxh = np.random.randn(hidden_size, input_size)  * s
        self.Whh = np.random.randn(hidden_size, hidden_size) * s
        self.Why = np.random.randn(output_size, hidden_size) * s

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, xs):
        T = len(xs) - 1
        hs = {-1: np.zeros((self.H, 1))}
        ps = {}
        loss = 0.0

        for t in range(T):
            hs[t] = np.tanh(
                self.Wxh @ xs[t] +
                self.Whh @ hs[t - 1] +
                self.bh
            )
            logits = self.Why @ hs[t] + self.by
            ps[t]  = self.softmax(logits)

            loss += -np.log(ps[t][Y_idx[t], 0] + 1e-8)

        return loss / T, hs, ps

    def train_step(self, xs):
        loss, hs, ps = self.forward(xs)
        T = len(xs) - 1

        dWhy = np.zeros_like(self.Why)
        dby  = np.zeros_like(self.by)

        for t in range(T):
            dy = ps[t].copy()
            dy[Y_idx[t]] -= 1
            dWhy += dy @ hs[t].T
            dby  += dy

        self.Why -= self.lr * np.clip(dWhy / T, -5, 5)
        self.by  -= self.lr * np.clip(dby  / T, -5, 5)

        return loss

    def gradient_flow(self, steps=30):
        h = np.ones((self.H, 1)) * 0.5
        magnitudes = []

        for _ in range(steps):
            h   = np.tanh(self.Whh @ h)
            jac = 1.0 - h ** 2
            magnitudes.append(float(np.mean(np.abs(jac))))

        return magnitudes

    def generate(self, start_char, length=20):
        h   = np.zeros((self.H, 1))
        idx = char_to_idx.get(start_char, 0)
        out = start_char

        for _ in range(length):
            x   = one_hot(idx, vocab_size)
            h   = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            logits = self.Why @ h + self.by
            e   = np.exp(logits - np.max(logits))
            idx = int(np.argmax(e / e.sum()))
            out += idx_to_char[idx]

        return out
```

---

## 5. Model Training

**Where is the training?**: The training loops 1000 times (epochs), repeatedly calling `rnn.train_step(X)`.
**What this does**: It feeds our encoded text into the RNN. The RNN attempts to predict the next character, calculates how wrong it was (the loss), and adjusts its weights to do better next time. At the end, it uses the trained model to generate text.

```python
rnn    = SimpleRNN(vocab_size, hidden_size=32, output_size=vocab_size, lr=0.05)
losses = []

print("\n🔁 Training Simple RNN ...")
print("─" * 42)
for epoch in range(1000):
    loss = rnn.train_step(X)
    losses.append(loss)
    if epoch % 250 == 0:
        print(f"  Epoch {epoch:4d}  |  Loss: {loss:.4f}")

print("✅ Training complete!")
print(f"\n🔤 Generated text: '{rnn.generate('h', length=20)}'")
```

---

## 6. Visualizing the Training Loss

**What this does**: We plot the loss curve collected during training using Matplotlib.
**Why**: This helps us verify that the model is actually learning. A downward-trending curve means the model is getting better at predicting the next character over time.

```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(losses, color='#e74c3c', linewidth=2)
ax.fill_between(range(len(losses)), losses, alpha=0.1, color='#e74c3c')
ax.set_title("RNN Training Loss (Cross-Entropy)\n"
             "Loss decreases as the model learns character patterns",
             fontsize=12, pad=10)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 7. The Vanishing Gradient Problem

**What this does**: We simulate how the gradient magnitude shrinks over 30 time steps and plot it.
**Why**: This visually proves the major flaw of simple RNNs. The mathematical nature of the `tanh` activation function causes the "memory signal" (gradient) to exponentially shrink toward zero as it goes backward in time. After about 10 steps, the model essentially forgets what it saw.

```python
grad_mags = rnn.gradient_flow(steps=30)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, 31), grad_mags,
        color='#c0392b', linewidth=2.5, marker='o', markersize=5)
ax.fill_between(range(1, 31), grad_mags, alpha=0.12, color='#c0392b')
ax.axhline(0.01, color='gray', linestyle='--', alpha=0.7,
           label='Near-zero threshold (model has "forgotten")')
ax.set_title(
    "Vanishing Gradient Problem   [Slide 15]\n"
    "Gradient magnitude shrinks to near-zero after ~10 time steps",
    fontsize=12, pad=10)
ax.set_xlabel("Time Steps Back in Sequence", fontsize=11)
ax.set_ylabel("Gradient Magnitude", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nObservation:")
print(f"  Step  1: gradient = {grad_mags[0]:.4f}")
print(f"  Step  5: gradient = {grad_mags[4]:.4f}")
print(f"  Step 15: gradient = {grad_mags[14]:.4f}")
print(f"  Step 30: gradient = {grad_mags[29]:.6f}")
```

---

## 8. Architecture Diagram

**What this does**: Generates a schematic diagram of how the RNN processes sequences, explicitly pointing out where the gradient fades.
**Why**: Visualizing the architecture helps connect the math to the conceptual sequence flow shown in lectures.

```python
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title("RNN Architecture — Sequential Processing   [Slide 15]",
             fontsize=13, fontweight='bold', pad=14)

steps  = [1.5, 4.0, 6.5, 9.0]
labels = ['x_1', 'x_2', 'x_3', 'x_4']
h_lbls = ['h_1', 'h_2', 'h_3', 'h_4']

for i, (xp, lbl, hl) in enumerate(zip(steps, labels, h_lbls)):
    box = mpatches.FancyBboxPatch(
        (xp - 0.75, 3.1), 1.5, 1.5,
        boxstyle="round,pad=0.1",
        facecolor='#e74c3c', edgecolor='white', linewidth=2, alpha=0.88)
    ax.add_patch(box)
    ax.text(xp, 3.85, f"{hl}\ntanh",
            color='white', ha='center', va='center',
            fontsize=9.5, fontweight='bold')

    ax.annotate('', xy=(xp, 3.1), xytext=(xp, 1.6),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2))
    ax.text(xp, 1.25, lbl, ha='center', fontsize=12,
            color='#2c3e50', fontweight='bold')

    ax.annotate('', xy=(xp, 5.5), xytext=(xp, 4.6),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))
    ax.text(xp, 5.8, f'y_{i+1}', ha='center', fontsize=10, color='#555')

    if i < len(steps) - 1:
        ax.annotate('', xy=(steps[i+1] - 0.75, 3.85),
                    xytext=(xp + 0.75, 3.85),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.2))

for i, (xp, fade) in enumerate(zip(steps, ['100%', '~60%', '~15%', '~2%'])):
    ax.text(xp, 6.8, fade, ha='center', fontsize=9,
            color='#c0392b', fontweight='bold')
ax.text(5.25, 7.2, "Gradient size going backwards (BPTT):",
        ha='center', fontsize=9.5, color='#7f8c8d')

ax.text(6.5, 0.4,
        "Problem: Each tanh step shrinks the gradient."
        "  After ~10 steps, gradient ≈ 0  →  model forgets!",
        ha='center', fontsize=10, color='#c0392b',
        bbox=dict(boxstyle='round', facecolor='#fdecea', edgecolor='#e74c3c'))

plt.tight_layout()
plt.show()
```

---

## 9. Hidden State Heatmap

**What this does**: Plots the internal values of the hidden state (`h_t`) as a heatmap across the entire sequence of characters.
**Why**: The hidden state represents the RNN's "memory". This plot lets you visualize how that memory shifts and changes its pattern as it processes each new character.

```python
_, hs, _ = rnn.forward(X)
hidden_matrix = np.array([hs[t][:16, 0] for t in range(len(text) - 1)])

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(hidden_matrix.T, aspect='auto', cmap='RdBu_r',
               vmin=-1, vmax=1)
ax.set_title("RNN Hidden State Values Over Time\n"
             "Each row = one hidden unit, each column = one character step",
             fontsize=12, pad=10)
ax.set_xlabel("Character Position", fontsize=11)
ax.set_ylabel("Hidden Unit Index", fontsize=11)
ax.set_xticks(range(len(text) - 1))
ax.set_xticklabels([repr(c) for c in text[:-1]], fontsize=8)
plt.colorbar(im, ax=ax, label="Activation (-1 to 1)")
plt.tight_layout()
plt.show()
```

---

## What is an RNN?
A **Recurrent Neural Network (RNN)** is an architecture designed for processing sequential data (like text or time series). Unlike traditional networks, it has a "memory" in the form of a hidden state, allowing it to remember what happened in previous steps and use that context to make predictions.

## Where do we use this?
While basic RNNs are mostly a stepping stone today, their architecture laid the foundation for modern sequence processing:
1. **Early Language Translation**: First attempts at machine translation (like early Google Translate).
2. **Speech Recognition**: Converting audio sequences into text sequences.
3. **Time Series Forecasting**: Predicting stock prices or weather patterns based on historical data.

*Note: The major limitation (vanishing gradients) led to the development of LSTMs, and eventually Transformers (like GPT).*
