# Comprehensive Guide: Building a Simple LSTM from Scratch

This document explains the end-to-end process of building a Long Short-Term Memory (LSTM) network using only NumPy. It breaks down the code step-by-step so you understand **what** each part does, **why** it is necessary, and **how** it solves the major flaw of standard RNNs (the vanishing gradient problem).

---

## 1. Imports and Setup

**What this does**: We import NumPy for all mathematical operations and Matplotlib for visualizing the results. We also set a random seed to ensure our results are reproducible.
**Why**: Since we are building the neural network from scratch without libraries like TensorFlow or PyTorch, NumPy is essential for handling the matrix multiplications.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

np.random.seed(42)
print("✅ Libraries loaded. NumPy:", np.__version__)
print("=" * 50)
print("  Simple LSTM — From Scratch")
print("   Slide 16: LSTMs (2010s)")
print("=" * 50)
```

---

## 2. Dataset and Vocabulary Building

**What this does**: We define a simple text string that we want the LSTM to learn. We then find all unique characters to build a vocabulary, and create mappings from characters to numerical indices.
**Why**: Neural networks require numerical input. Building a vocabulary allows us to map each character into a unique index, preparing the data to be converted into mathematical vectors.

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

**What this does**: We define a function to convert a character index into a "one-hot vector" (an array of zeros with a single `1` at the character's index). We encode our entire input text `X` and define our targets `Y_idx` (the next character).
**Why**: One-hot encoding is how we represent individual characters mathematically so the network can process them.

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

## 4. The LSTM Architecture

**What this does**: We define a `SimpleLSTM` class containing the 4 gates (Forget, Input, Cell candidate, and Output) and the two distinct memory states (`c_t` for long-term memory, `h_t` for short-term memory). The `step` method calculates these gate values and updates the states.
**Why**: Unlike a simple RNN which overwrites its memory at every step, the LSTM uses these learned gates to decide exactly what information to keep, what to throw away, and what to output. The long-term cell state `c_t` acts as a "highway", allowing gradients to flow backward unchanged, which solves the vanishing gradient problem.

```python
class SimpleLSTM:
    def __init__(self, input_size, hidden_size, output_size, lr=0.05):
        self.H  = hidden_size
        self.lr = lr
        D = input_size + hidden_size
        s = 0.1

        self.Wf = np.random.randn(hidden_size, D) * s
        self.Wi = np.random.randn(hidden_size, D) * s
        self.Wg = np.random.randn(hidden_size, D) * s
        self.Wo = np.random.randn(hidden_size, D) * s

        self.bf = np.ones((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bg = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        self.Wy = np.random.randn(output_size, hidden_size) * s
        self.by = np.zeros((output_size, 1))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def step(self, x, h_prev, c_prev):
        comb = np.vstack([h_prev, x])

        f = self.sigmoid(self.Wf @ comb + self.bf)
        i = self.sigmoid(self.Wi @ comb + self.bi)
        g = np.tanh(self.Wg @ comb + self.bg)
        o = self.sigmoid(self.Wo @ comb + self.bo)

        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)

        return h_next, c_next, {
            'forget': float(f.mean()),
            'input':  float(i.mean()),
            'cell':   float(g.mean()),
            'output': float(o.mean()),
        }

    def forward(self, xs):
        T  = len(xs) - 1
        h  = np.zeros((self.H, 1))
        c  = np.zeros((self.H, 1))
        hs = {}
        gate_hist = []
        loss = 0.0

        for t in range(T):
            h, c, gates = self.step(xs[t], h, c)
            hs[t]       = h
            gate_hist.append(gates)

            logits = self.Wy @ h + self.by
            probs  = self.softmax(logits)
            loss  += -np.log(probs[Y_idx[t], 0] + 1e-8)

        return loss / T, hs, gate_hist

    def train_step(self, xs):
        loss, hs, gate_hist = self.forward(xs)
        T = len(xs) - 1

        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        for t in range(T):
            logits = self.Wy @ hs[t] + self.by
            e = np.exp(logits - np.max(logits))
            p = e / e.sum()
            dy = p.copy()
            dy[Y_idx[t]] -= 1
            dWy += dy @ hs[t].T
            dby += dy

        self.Wy -= self.lr * np.clip(dWy / T, -5, 5)
        self.by -= self.lr * np.clip(dby / T, -5, 5)

        return loss, gate_hist

    def generate(self, start_char, length=20):
        h   = np.zeros((self.H, 1))
        c   = np.zeros((self.H, 1))
        idx = char_to_idx.get(start_char, 0)
        out = start_char

        for _ in range(length):
            x       = one_hot(idx, vocab_size)
            h, c, _ = self.step(x, h, c)
            logits  = self.Wy @ h + self.by
            e       = np.exp(logits - np.max(logits))
            idx     = int(np.argmax(e / e.sum()))
            out    += idx_to_char[idx]

        return out
```

---

## 5. Model Training

**Where is the training?**: The training loops 1000 times (epochs), repeatedly calling `lstm.train_step(X)`.
**What this does**: It feeds our encoded text into the LSTM. The LSTM attempts to predict the next character, calculates the loss, and adjusts its weights to improve. We also use the trained model to generate text.

```python
lstm        = SimpleLSTM(vocab_size, hidden_size=32, output_size=vocab_size, lr=0.05)
lstm_losses = []
final_gates = None

print("\n  Training Simple LSTM ...")
print("─" * 42)
for epoch in range(1000):
    loss, gates = lstm.train_step(X)
    lstm_losses.append(loss)
    if epoch % 250 == 0:
        final_gates = gates
        print(f"  Epoch {epoch:4d}  |  Loss: {loss:.4f}")

print("✅ Training complete!")
print(f"\n🔤 Generated text: '{lstm.generate('h', length=20)}'")
```

---

## 6. Visualizing the Training Loss

**What this does**: We plot the loss curve collected during training using Matplotlib.
**Why**: This helps verify that the LSTM is successfully learning the sequence patterns over time.

```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(lstm_losses, color='#2980b9', linewidth=2)
ax.fill_between(range(len(lstm_losses)), lstm_losses, alpha=0.1, color='#2980b9')
ax.set_title("LSTM Training Loss (Cross-Entropy)\n"
             "More stable convergence than RNN due to cell state gradient highway",
             fontsize=12, pad=10)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 7. Visualizing the 4 Gate Activations

**What this does**: This plots exactly what the LSTM's four gates are doing at every character position.
**Why**: This is the core of understanding an LSTM. A Forget gate value near 1 means it's keeping old memory. An Input gate value near 1 means it is writing new data.

```python
gate_keys   = ['forget', 'input',  'cell',            'output']
gate_titles = [
    'Forget Gate (f)\n"What to erase from memory"',
    'Input Gate (i)\n"What new info to write"',
    'Cell Candidate (g)\n"What could be stored"',
    'Output Gate (o)\n"What to expose as h_t"',
]
gate_colors = ['#e74c3c', '#27ae60', '#2980b9', '#f39c12']

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(
    "LSTM — 4 Gate Activations Over the Sequence   [Slide 16]\n"
    "Each gate has a different job; all are LEARNED during training",
    fontsize=13, fontweight='bold')

time_steps  = list(range(len(final_gates)))
char_labels = [repr(c) for c in text[:-1]]

for ax, key, title, color in zip(axes.flat, gate_keys, gate_titles, gate_colors):
    values = [g[key] for g in final_gates]

    ax.bar(time_steps, values, color=color, alpha=0.55,
           edgecolor='white', width=0.6)
    ax.plot(time_steps, values, color=color,
            linewidth=2, marker='o', markersize=5)
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_xlabel("Character Position in Sequence", fontsize=9)
    ax.set_ylabel("Mean Gate Activation", fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(time_steps)
    ax.set_xticklabels(char_labels, fontsize=7.5, rotation=45)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.9)
    ax.grid(True, alpha=0.25, axis='y')

plt.tight_layout()
plt.show()

print("Reading the chart:")
print("  Forget gate close to 1  ->  model is KEEPING old memory")
print("  Forget gate close to 0  ->  model is ERASING old memory")
print("  Input  gate close to 1  ->  model is WRITING new information")
print("  Input  gate close to 0  ->  model is IGNORING the current input")
```

---

## 8. Solving the Vanishing Gradient

**What this does**: Simulates and compares the backward gradient flow of a Simple RNN versus an LSTM.
**Why**: In an RNN, gradients shrink via multiplication with the `tanh` derivative. In an LSTM, gradients travel over an additive cell state pathway (`c_t = f*c_{t-1} + i*g`). Because it's additive, the gradient survives for much longer, allowing the LSTM to learn from context hundreds of steps back!

```python
steps_back   = 30
rnn_grads    = []
lstm_grads   = []

h = np.ones((32, 1)) * 0.5
Whh_rnn = np.random.randn(32, 32) * 0.1
for _ in range(steps_back):
    h = np.tanh(Whh_rnn @ h)
    jac = 1.0 - h ** 2
    rnn_grads.append(float(np.mean(np.abs(jac))))

f_gate = 0.95
for step in range(steps_back):
    lstm_grads.append(f_gate ** step)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, steps_back+1), rnn_grads,
        color='#e74c3c', linewidth=2.5, marker='o', markersize=4,
        label='RNN (tanh Jacobian)')
ax.plot(range(1, steps_back+1), lstm_grads,
        color='#2980b9', linewidth=2.5, marker='s', markersize=4,
        label=f'LSTM (cell state, forget≈{f_gate})')
ax.axhline(0.01, color='gray', linestyle='--', alpha=0.6,
           label='Near-zero threshold')
ax.set_title(
    "Gradient Flow Comparison — RNN vs LSTM   [Slides 15 & 16]\n"
    "LSTM cell state lets gradient travel further back in time",
    fontsize=12, pad=10)
ax.set_xlabel("Time Steps Back", fontsize=11)
ax.set_ylabel("Gradient Magnitude", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 9. LSTM Architecture Diagram

**What this does**: Generates a schematic diagram showing the separate cell state and hidden state pathways.
**Why**: Visualizing the architecture illustrates the "long-term memory highway" in an easy-to-digest format.

```python
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis('off')
ax.set_title("LSTM Architecture — 4 Gates + Cell State Highway   [Slide 16]",
             fontsize=13, fontweight='bold', pad=14)

positions = [1.8, 5.5, 9.2]
gate_desc = [
    ['f', 'Forget', '#e74c3c'],
    ['i', 'Input',  '#27ae60'],
    ['g', 'Cell',   '#2980b9'],
    ['o', 'Output', '#f39c12'],
]

for i, xp in enumerate(positions):
    cbox = mpatches.FancyBboxPatch(
        (xp - 1.0, 6.4), 2.0, 1.0,
        boxstyle="round,pad=0.08",
        facecolor='#8e44ad', edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(cbox)
    ax.text(xp, 6.9, f"c_{i+1}  (cell state)",
            color='white', ha='center', va='center',
            fontsize=8.5, fontweight='bold')

    hbox = mpatches.FancyBboxPatch(
        (xp - 1.0, 3.8), 2.0, 1.4,
        boxstyle="round,pad=0.08",
        facecolor='#2980b9', edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(hbox)
    ax.text(xp, 4.5, f"h_{i+1}\nf / i / g / o",
            color='white', ha='center', va='center',
            fontsize=8.5, fontweight='bold')

    ax.annotate('', xy=(xp, 3.8), xytext=(xp, 2.2),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2))
    ax.text(xp, 1.85, f"x_{i+1}", ha='center', fontsize=12,
            color='#2c3e50', fontweight='bold')

    ax.annotate('', xy=(xp, 6.4), xytext=(xp, 5.2),
                arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2))
    ax.annotate('', xy=(xp, 8.0), xytext=(xp, 7.4),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))
    ax.text(xp, 8.3, f"y_{i+1}", ha='center', fontsize=10, color='#555')

    if i < len(positions) - 1:
        nx = positions[i + 1]
        ax.annotate('', xy=(nx - 1.0, 6.9), xytext=(xp + 1.0, 6.9),
                    arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2.5))
        ax.annotate('', xy=(nx - 1.0, 4.5), xytext=(xp + 1.0, 4.5),
                    arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2))

legend_items = [
    mpatches.Patch(color='#8e44ad', label='Cell State c_t  (long-term memory)'),
    mpatches.Patch(color='#2980b9', label='Hidden State h_t (short-term output)'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=9,
          framealpha=0.9)

ax.text(7.0, 0.5,
        "Key: c_t = f*c_{t-1} + i*g  -- additive update means gradient flows unchanged!",
        ha='center', fontsize=10, color='#27ae60',
        bbox=dict(boxstyle='round', facecolor='#eafaf1', edgecolor='#27ae60'))

plt.tight_layout()
plt.show()
```

---

## 10. Hidden State Heatmap

**What this does**: Plots the internal values of the short-term hidden state (`h_t`) as a heatmap across the entire sequence of characters.
**Why**: When compared to the RNN heatmap, you can see that the LSTM exhibits much richer, more complex memory patterns due to the gating mechanisms.

```python
_, hs_lstm, _ = lstm.forward(X)
lstm_matrix   = np.array([hs_lstm[t][:16, 0] for t in range(len(text) - 1)])

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(lstm_matrix.T, aspect='auto', cmap='RdBu_r',
               vmin=-1, vmax=1)
ax.set_title("LSTM Hidden State Values Over Time\n"
             "Richer patterns than RNN thanks to the gated cell state",
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

## What is an LSTM?
A **Long Short-Term Memory (LSTM)** network is an evolution of the traditional RNN. Instead of just overwriting its memory at every step, it relies on a long-term "cell state" that acts as a memory highway. It dynamically decides what to erase from memory (Forget gate), what new data to keep (Input gate), and what to output (Output gate). 

## Why did we need this?
LSTMs were created specifically to fix the **vanishing gradient problem** that plagued RNNs. Because the cell state update is additive (not multiplicative), gradients do not exponentially shrink, allowing the model to learn long-range patterns spanning hundreds of steps.

*Note: While LSTMs are extremely capable, they still process data sequentially (one step at a time). This bottleneck eventually led to the creation of Transformers.*
