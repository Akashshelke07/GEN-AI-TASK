# ================================================================
# 🤖 Module 1: Generative AI — Evolution of Sequence Models
# 📌 RNN & LSTM Built From Scratch — NumPy Only
# 📚 Aligned with Slides 15 & 16 (Evolution of Generative AI)
# ================================================================
# ✅ No TensorFlow | No PyTorch | No Keras
# ✅ Only: numpy + matplotlib  (both pre-installed in Colab)
# ================================================================
#
# HOW TO USE IN GOOGLE COLAB:
#   Option A: File → Open notebook → Upload this .py file
#   Option B: Copy each "# ── CELL N ──" block into separate cells
#
# ================================================================


# ── CELL 1: Imports ────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)
print("✅ Libraries loaded. NumPy:", np.__version__)


# ── CELL 2: Dataset ────────────────────────────────────────────
# We use a short text so students can see the model learn
# character-by-character prediction (same idea behind GPT!)

text = "hello generative ai world"

# Build vocabulary (unique characters)
chars     = sorted(set(text))
vocab_size = len(chars)

char_to_idx = {ch: i  for i,  ch in enumerate(chars)}
idx_to_char = {i:  ch for ch, i  in char_to_idx.items()}

print("📝 Text          :", text)
print("🔤 Vocabulary    :", "".join(chars))
print(f"📊 Vocab size    : {vocab_size}")
print(f"📏 Sequence len  : {len(text)}")


# ── CELL 3: Helper — One-Hot Encoding ──────────────────────────

def one_hot(idx: int, size: int) -> np.ndarray:
    """Return a column vector of zeros with a 1 at position `idx`."""
    v = np.zeros((size, 1))
    v[idx] = 1.0
    return v

# Pre-encode the whole text into one-hot vectors
X       = [one_hot(char_to_idx[c], vocab_size) for c in text]
Y_idx   = [char_to_idx[text[i + 1]] for i in range(len(text) - 1)]

print("\n✅ Data ready.")
print(f"   Inputs  : {len(X)} one-hot vectors of size ({vocab_size}, 1)")
print(f"   Targets : {len(Y_idx)} character indices")


# ================================================================
# 🔁  PART 1 — SIMPLE RNN
#     Slide 15: "Process data one step at a time and remember
#                previous information"
#              "Problem: Vanishing Gradient"
# ================================================================

# ── CELL 4: RNN Class ──────────────────────────────────────────

class SimpleRNN:
    """
    Vanilla RNN — one equation, one gate:

        h_t = tanh( Wxh·x_t  +  Whh·h_{t-1}  +  bh )
        y_t = softmax( Why·h_t  +  by )

    Parameters
    ----------
    input_size   : size of one-hot input  (= vocab_size)
    hidden_size  : number of hidden units  (our "memory")
    output_size  : number of output classes (= vocab_size)
    lr           : learning rate
    """

    def __init__(self, input_size, hidden_size, output_size, lr=0.05):
        self.H  = hidden_size
        self.lr = lr
        s = 0.1   # small random scale

        # Weight matrices
        self.Wxh = np.random.randn(hidden_size, input_size)  * s
        self.Whh = np.random.randn(hidden_size, hidden_size) * s
        self.Why = np.random.randn(output_size, hidden_size) * s

        # Bias vectors
        self.bh  = np.zeros((hidden_size, 1))
        self.by  = np.zeros((output_size, 1))

    # ── helpers ──────────────────────────────────────────────
    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # ── forward pass ─────────────────────────────────────────
    def forward(self, xs):
        """
        Run the RNN over `xs` (list of one-hot vectors).
        Returns loss and internal caches for backprop.
        """
        T = len(xs) - 1          # number of (input → target) steps
        hs = {-1: np.zeros((self.H, 1))}
        ps = {}
        loss = 0.0

        for t in range(T):
            # Core RNN equation
            hs[t] = np.tanh(
                self.Wxh @ xs[t] +
                self.Whh @ hs[t - 1] +
                self.bh
            )
            logits  = self.Why @ hs[t] + self.by
            ps[t]   = self.softmax(logits)
            loss   += -np.log(ps[t][Y_idx[t], 0] + 1e-8)

        return loss / T, hs, ps

    # ── parameter update (output layer only — keeps code simple) ─
    def train_step(self, xs):
        loss, hs, ps = self.forward(xs)
        T = len(xs) - 1

        dWhy = np.zeros_like(self.Why)
        dby  = np.zeros_like(self.by)

        for t in range(T):
            dy = ps[t].copy()
            dy[Y_idx[t]] -= 1            # cross-entropy gradient
            dWhy += dy @ hs[t].T
            dby  += dy

        # Gradient clipping + SGD update
        self.Why -= self.lr * np.clip(dWhy / T, -5, 5)
        self.by  -= self.lr * np.clip(dby  / T, -5, 5)

        return loss

    # ── vanishing gradient demo ───────────────────────────────
    def gradient_flow(self, steps=30):
        """
        Simulate how the gradient magnitude shrinks as we
        backpropagate through time.

        At each step we multiply by the Jacobian of tanh:
            d/dx [tanh(Whh·h)] = (1 - tanh²) — always ≤ 1

        Repeated multiplications → exponential decay → 0
        """
        h   = np.ones((self.H, 1)) * 0.5
        magnitudes = []

        for _ in range(steps):
            h   = np.tanh(self.Whh @ h)
            # Jacobian of tanh at current h
            jac = 1.0 - h ** 2
            # Effective gradient magnitude through this step
            magnitudes.append(float(np.mean(np.abs(jac))))

        return magnitudes


# ── CELL 5: Train RNN ──────────────────────────────────────────

rnn        = SimpleRNN(vocab_size, hidden_size=32, output_size=vocab_size, lr=0.05)
rnn_losses = []

print("🔁 Training Simple RNN …")
print("─" * 42)

for epoch in range(1000):
    loss = rnn.train_step(X)
    rnn_losses.append(loss)
    if epoch % 250 == 0:
        print(f"  Epoch {epoch:4d}  │  Loss: {loss:.4f}")

print("✅ RNN training complete!\n")


# ================================================================
# 🧠  PART 2 — LSTM
#     Slide 16: "Special memory cells that can remember important
#                information for a much longer time"
#              "Key Innovation: 4 gates"
# ================================================================

# ── CELL 6: LSTM Class ─────────────────────────────────────────

class SimpleLSTM:
    """
    LSTM from scratch — 4 gates, cell state, hidden state.

    Equations (one time step):
    ─────────────────────────────────────────────────────────────
    combined = [h_{t-1}, x_t]             (concatenate)

    f_t = σ( Wf·combined + bf )           FORGET gate
    i_t = σ( Wi·combined + bi )           INPUT  gate
    g_t = tanh( Wg·combined + bg )        CELL   gate  (candidate)
    o_t = σ( Wo·combined + bo )           OUTPUT gate

    c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t   Cell State  ← long memory!
    h_t = o_t ⊙ tanh(c_t)               Hidden State ← short output
    ─────────────────────────────────────────────────────────────
    ⊙ means element-wise multiplication
    σ is the sigmoid function (outputs 0–1, acts like a "gate valve")
    """

    def __init__(self, input_size, hidden_size, output_size, lr=0.05):
        self.H  = hidden_size
        self.lr = lr
        D = input_size + hidden_size   # combined input dimension
        s = 0.1

        # 4 gate weight matrices
        self.Wf = np.random.randn(hidden_size, D) * s   # Forget
        self.Wi = np.random.randn(hidden_size, D) * s   # Input
        self.Wg = np.random.randn(hidden_size, D) * s   # Cell (candidate)
        self.Wo = np.random.randn(hidden_size, D) * s   # Output

        # Biases — forget gate starts at 1 so the model remembers by default
        self.bf = np.ones((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bg = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        # Output projection
        self.Wy = np.random.randn(output_size, hidden_size) * s
        self.by = np.zeros((output_size, 1))

    # ── helpers ──────────────────────────────────────────────
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # ── one time step ────────────────────────────────────────
    def step(self, x, h_prev, c_prev):
        """
        One LSTM cell forward pass.
        Returns: h_next, c_next, gate_activations
        """
        comb = np.vstack([h_prev, x])      # concatenate

        f = self.sigmoid(self.Wf @ comb + self.bf)   # 🚪 Forget
        i = self.sigmoid(self.Wi @ comb + self.bi)   # 🚪 Input
        g = np.tanh(   self.Wg @ comb + self.bg)     # 🧠 Candidate
        o = self.sigmoid(self.Wo @ comb + self.bo)   # 🚪 Output

        c_next = f * c_prev + i * g       # 📦 Cell state
        h_next = o * np.tanh(c_next)      # 💡 Hidden state

        return h_next, c_next, {
            'forget': float(f.mean()),
            'input':  float(i.mean()),
            'cell':   float(g.mean()),
            'output': float(o.mean()),
        }

    # ── full sequence forward pass ────────────────────────────
    def forward(self, xs):
        T = len(xs) - 1
        h = np.zeros((self.H, 1))
        c = np.zeros((self.H, 1))

        hs        = {}
        gate_hist = []
        loss      = 0.0

        for t in range(T):
            h, c, gates = self.step(xs[t], h, c)
            hs[t]       = h
            gate_hist.append(gates)

            logits  = self.Wy @ h + self.by
            probs   = self.softmax(logits)
            loss   += -np.log(probs[Y_idx[t], 0] + 1e-8)

        return loss / T, hs, gate_hist

    # ── parameter update ─────────────────────────────────────
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


# ── CELL 7: Train LSTM ─────────────────────────────────────────

lstm        = SimpleLSTM(vocab_size, hidden_size=32, output_size=vocab_size, lr=0.05)
lstm_losses = []
final_gates = None

print("🧠 Training Simple LSTM …")
print("─" * 42)

for epoch in range(1000):
    loss, gates = lstm.train_step(X)
    lstm_losses.append(loss)
    if epoch % 250 == 0:
        final_gates = gates
        print(f"  Epoch {epoch:4d}  │  Loss: {loss:.4f}")

print("✅ LSTM training complete!\n")


# ================================================================
# 📊  VISUALIZATIONS  (4 plots)
# ================================================================

# ── CELL 8: Plot 1 — Vanishing Gradient ────────────────────────

grad_magnitudes = rnn.gradient_flow(steps=30)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(range(1, 31), grad_magnitudes,
        color='#e74c3c', linewidth=2.5, marker='o', markersize=5)
ax.fill_between(range(1, 31), grad_magnitudes, alpha=0.12, color='#e74c3c')
ax.axhline(0.01, color='gray', linestyle='--', alpha=0.7,
           label='Near-zero (model has "forgotten")')
ax.set_title(
    "⚠️  RNN — Vanishing Gradient Problem   (Slide 15)\n"
    "Gradient signal almost disappears after ~10 time steps → short-term memory only",
    fontsize=12, pad=10)
ax.set_xlabel("Time Steps Back in Sequence", fontsize=11)
ax.set_ylabel("Gradient Magnitude", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("💡 This is exactly why RNNs 'forgot information after a few words' (Slide 15).")


# ── CELL 9: Plot 2 — RNN vs LSTM Training Loss ─────────────────

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(rnn_losses,  label='🔁 RNN',  color='#e74c3c', linewidth=2)
ax.plot(lstm_losses, label='🧠 LSTM', color='#2980b9', linewidth=2)
ax.set_title(
    "📊 Training Loss — RNN vs LSTM\n"
    "LSTM has more stable gradient flow thanks to the cell state highway",
    fontsize=12, pad=10)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ── CELL 10: Plot 3 — LSTM 4 Gate Activations ──────────────────

gate_keys    = ['forget', 'input',  'cell',          'output']
gate_titles  = [
    '🚪 Forget Gate (f)\n"What to erase from memory"',
    '🚪 Input Gate (i)\n"What new info to store"',
    '🧠 Cell Gate / Candidate (g)\n"What could be remembered"',
    '🚪 Output Gate (o)\n"What to expose as hidden state"',
]
gate_colors  = ['#e74c3c', '#27ae60', '#2980b9', '#f39c12']

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(
    "🧠 LSTM Internal Gate Activations   (Slide 16)\n"
    "4 learnable gates decide what to remember, write, and output",
    fontsize=13, fontweight='bold')

time_steps = list(range(len(final_gates)))

for ax, key, title, color in zip(axes.flat, gate_keys, gate_titles, gate_colors):
    values = [g[key] for g in final_gates]
    ax.bar(time_steps, values, color=color, alpha=0.6, edgecolor='white', width=0.6)
    ax.plot(time_steps, values, color=color, linewidth=2, marker='o', markersize=5)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("Character Position in Sequence")
    ax.set_ylabel("Mean Activation (0 = off,  1 = on)")
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.grid(True, alpha=0.25, axis='y')
    # Annotate characters
    chars_in_seq = list(text[:-1])
    ax.set_xticks(time_steps)
    ax.set_xticklabels([repr(c) for c in chars_in_seq], fontsize=7, rotation=45)

plt.tight_layout()
plt.show()


# ── CELL 11: Plot 4 — Side-by-side Architecture Diagram ────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ─── RNN diagram ───────────────────────────────────────────────
ax1.set_xlim(0, 10); ax1.set_ylim(0, 8); ax1.axis('off')
ax1.set_title("🔁 Simple RNN\n(Slide 15)", fontsize=13, fontweight='bold', pad=12)

rnn_steps = [1.5, 4.0, 6.5, 8.5]
labels    = ['x₁', 'x₂', 'x₃', '…']
for i, (xp, lbl) in enumerate(zip(rnn_steps, labels)):
    # hidden box
    box = mpatches.FancyBboxPatch((xp - 0.7, 3.2), 1.4, 1.4,
        boxstyle="round,pad=0.1", facecolor='#e74c3c', edgecolor='white', alpha=0.85)
    ax1.add_patch(box)
    ax1.text(xp, 3.9, f"h_{i+1}\ntanh", color='white',
             ha='center', va='center', fontsize=9, fontweight='bold')
    # input arrow
    ax1.annotate('', xy=(xp, 3.2), xytext=(xp, 1.5),
                 arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))
    ax1.text(xp, 1.2, lbl, ha='center', fontsize=11,
             color='#333', fontweight='bold')
    # recurrent arrow
    if i < len(rnn_steps) - 1:
        ax1.annotate('', xy=(rnn_steps[i+1] - 0.7, 3.9),
                     xytext=(xp + 0.7, 3.9),
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.8))
    # output arrow
    ax1.annotate('', xy=(xp, 6.0), xytext=(xp, 4.6),
                 arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    ax1.text(xp, 6.3, f'ŷ_{i+1}', ha='center', fontsize=9, color='#555')

ax1.text(5.0, 0.4, "⚠️  Gradient fades backward → short memory!",
         ha='center', fontsize=10, color='#e74c3c',
         bbox=dict(boxstyle='round', facecolor='#fdecea', edgecolor='#e74c3c'))

# ─── LSTM diagram ──────────────────────────────────────────────
ax2.set_xlim(0, 10); ax2.set_ylim(0, 8); ax2.axis('off')
ax2.set_title("🧠 LSTM\n(Slide 16)", fontsize=13, fontweight='bold', pad=12)

lstm_steps = [1.5, 4.0, 6.5]
for i, xp in enumerate(lstm_steps):
    # cell state highway (top)
    box_c = mpatches.FancyBboxPatch((xp - 0.75, 5.5), 1.5, 0.9,
        boxstyle="round,pad=0.08", facecolor='#8e44ad', edgecolor='white', alpha=0.85)
    ax2.add_patch(box_c)
    ax2.text(xp, 5.95, f'c_{i+1}', color='white', ha='center',
             va='center', fontsize=9, fontweight='bold')

    # hidden state box
    box_h = mpatches.FancyBboxPatch((xp - 0.75, 3.2), 1.5, 1.2,
        boxstyle="round,pad=0.08", facecolor='#2980b9', edgecolor='white', alpha=0.85)
    ax2.add_patch(box_h)
    ax2.text(xp, 3.8, f'h_{i+1}\n4 gates', color='white', ha='center',
             va='center', fontsize=8.5, fontweight='bold')

    # input arrow
    ax2.annotate('', xy=(xp, 3.2), xytext=(xp, 1.5),
                 arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))
    ax2.text(xp, 1.2, f'x_{i+1}', ha='center', fontsize=11,
             color='#333', fontweight='bold')

    # cell → output
    ax2.annotate('', xy=(xp, 5.5), xytext=(xp, 4.4),
                 arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.5))
    ax2.annotate('', xy=(xp, 7.0), xytext=(xp, 6.4),
                 arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    ax2.text(xp, 7.25, f'ŷ_{i+1}', ha='center', fontsize=9, color='#555')

    if i < len(lstm_steps) - 1:
        # cell highway arrow
        ax2.annotate('', xy=(lstm_steps[i+1] - 0.75, 5.95),
                     xytext=(xp + 0.75, 5.95),
                     arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2.2))
        # hidden recurrent arrow
        ax2.annotate('', xy=(lstm_steps[i+1] - 0.75, 3.8),
                     xytext=(xp + 0.75, 3.8),
                     arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.8))

ax2.text(0.3, 6.3, "📦 Cell State\n(Long memory\nhighway)",
         fontsize=8, color='#8e44ad', va='center',
         bbox=dict(boxstyle='round', facecolor='#f3e5f5', edgecolor='#8e44ad', alpha=0.7))

ax2.text(4.8, 0.4, "✅  Cell state highway keeps gradients alive!",
         ha='center', fontsize=10, color='#27ae60',
         bbox=dict(boxstyle='round', facecolor='#e8f8f5', edgecolor='#27ae60'))

plt.suptitle("Architecture Comparison — RNN vs LSTM", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ================================================================
# 🔍  CELL 12: Inference — Generate text with each model
# ================================================================

def rnn_generate(rnn_model, start_char, length=15):
    """Greedily generate `length` characters with the RNN."""
    h   = np.zeros((rnn_model.H, 1))
    idx = char_to_idx.get(start_char, 0)
    out = start_char

    for _ in range(length):
        x   = one_hot(idx, vocab_size)
        h   = np.tanh(rnn_model.Wxh @ x + rnn_model.Whh @ h + rnn_model.bh)
        logits = rnn_model.Why @ h + rnn_model.by
        e   = np.exp(logits - np.max(logits))
        idx = int(np.argmax(e / e.sum()))
        out += idx_to_char[idx]

    return out


def lstm_generate(lstm_model, start_char, length=15):
    """Greedily generate `length` characters with the LSTM."""
    h   = np.zeros((lstm_model.H, 1))
    c   = np.zeros((lstm_model.H, 1))
    idx = char_to_idx.get(start_char, 0)
    out = start_char

    for _ in range(length):
        x       = one_hot(idx, vocab_size)
        h, c, _ = lstm_model.step(x, h, c)
        logits  = lstm_model.Wy @ h + lstm_model.by
        e       = np.exp(logits - np.max(logits))
        idx     = int(np.argmax(e / e.sum()))
        out    += idx_to_char[idx]

    return out


seed = 'h'
print(f"🔁 RNN  generated : '{rnn_generate(rnn,   seed, 20)}'")
print(f"🧠 LSTM generated : '{lstm_generate(lstm, seed, 20)}'")


# ── CELL 13: Final Summary ──────────────────────────────────────

print("""
╔══════════════════════════════════════════════════════════════╗
║         📚  MODULE 1 SUMMARY — COURSE TIMELINE              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📍 Slide 15 — RNNs (Early 2000s)                           ║
║  ─────────────────────────────────────────────────────────  ║
║  Formula : h_t = tanh(Wxh·x_t + Whh·h_{t-1} + bh)         ║
║  ✅  Handles sequences step-by-step                          ║
║  ✅  Shares weights across all time steps                    ║
║  ❌  Vanishing gradient → forgets after ~10 steps           ║
║  ❌  Cannot learn long-range dependencies                    ║
║                                                              ║
║  📍 Slide 16 — LSTMs (2010s)                                ║
║  ─────────────────────────────────────────────────────────  ║
║  Key : Cell state (c_t) = long-term memory highway          ║
║  🚪  Forget gate  : decide what OLD memory to erase         ║
║  🚪  Input  gate  : decide what NEW info to write           ║
║  🧠  Cell   gate  : candidate values to store               ║
║  🚪  Output gate  : decide what to expose as output         ║
║  ✅  Solves vanishing gradient via cell state               ║
║  ✅  Better at long sequences & text generation             ║
║  ❌  Still sequential — slow to train on long text          ║
║                                                              ║
║  📍 Slide 17 — Transformers (2017) — NEXT TOPIC             ║
║  ─────────────────────────────────────────────────────────  ║
║  → Self-Attention: look at ALL positions simultaneously     ║
║  → Parallel processing — much faster than RNN/LSTM          ║
║  → Behind GPT, Gemini, Claude, LLaMA, Mistral …            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
