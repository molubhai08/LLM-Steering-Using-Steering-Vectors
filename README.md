# LLM-Steering-Using-Steering-Vectors

This project explores **identity steering in large language models (LLMs)** using **activation steering vectors**.
The goal is to **systematically discover where, how, and how strongly** a model’s internal representations can be modified to induce a **stable identity shift** (e.g., from *AI assistant* → *Golden Retriever* 🐕).

---

## 📌 Key Contributions

✔ Layer-wise steering vector extraction
✔ Coefficient sweep to find effective steering strength
✔ Single-layer vs multi-layer steering comparison
✔ MLP hook vs residual hook analysis
✔ Discovery of **late-layer dominance** in identity control
✔ Final optimized function for stable identity steering

---

## 🧠 Core Concept

A **steering vector** is computed as the difference between hidden activations of two contrasting concepts:

```
steering_vector = mean(hidden_target − hidden_base)
```

These vectors are injected into transformer layers during generation using **forward hooks**, modifying the model’s internal computation **without fine-tuning**.

---

## 🧪 Experiment Phases

### **Phase 1 — Coefficient Sweep per Layer**

**Objective:**
Find how much steering strength (`coeff`) each layer can tolerate before becoming unstable.

**Method:**

* Extract AI ↔ Dog identity vectors
* Inject steering at one layer at a time
* Sweep coefficients from `5 → 30`
* Score outputs for “dog-like” behavior

**Finding:**

* Mid layers respond weakly
* Late layers respond strongly but require higher coefficients

---

### **Phase 2 — Multi-Layer Steering**

**Hypothesis:**
Single-layer steering is overwritten by later layers.

**Method:**

* Apply steering to **multiple layers simultaneously**
* Test different layer combinations:

  * Middle layers
  * Late layers
  * Early + Late
  * Wide ranges

**Finding:**

* Multi-layer steering preserves identity better
* Later layers dominate final behavior

---

### **Phase 3 — Pure Concept Extraction**

Instead of instruction prompts, **pure descriptive text** is used to extract cleaner concept vectors.

✔ Reduces instruction leakage
✔ Produces more stable identity vectors

---

### **Phase 4 — Late-Layer Optimization**

**Breakthrough Result**

Best configuration found:

```
Layers: [20, 22, 24, 26]
Coefficient: 30–35
Hook type: MLP (pre-residual)
```

This yields:

* Coherent
* Persistent
* Semantically consistent dog identity

---

### **Phase 5 — Residual vs MLP Hook Comparison**

| Hook Type     | Effect                              |
| ------------- | ----------------------------------- |
| MLP Hook      | Stronger identity control           |
| Residual Hook | More global but less precise        |
| Late MLP      | Best balance of coherence & control |

---

## ⚙️ Model & Environment

* **Model:** `microsoft/Phi-3-mini-4k-instruct`
* **Framework:** PyTorch + Hugging Face Transformers
* **Hardware:** CUDA GPU required
* **Cache:** Disabled (`use_cache=False`) for correct hooks

---

## 📂 File Structure

```
steering_vectors.py
├─ Layer-wise coefficient sweep
├─ Multi-layer steering experiments
├─ Pure concept vector extraction
├─ Late-layer optimization
├─ MLP hook steering
├─ Residual hook steering
└─ Interactive demo & comparison
```

---

## 🧪 Example Usage

```python
response = generate_as_dog(
    prompt="Who are you?",
    layers=[20, 22, 24, 26],
    coeff=32.0,
    temperature=0.8
)
print(response)
```

---

## 🔬 Key Findings

* Identity is **not localized** to a single layer
* Later layers have **higher semantic leverage**
* Steering strength must scale with layer depth
* Multi-layer steering prevents identity correction
* MLP hooks outperform residual hooks for concept control

---

## ⚠️ Limitations

* Model-specific (Phi-3 behavior may not generalize)
* Over-steering causes repetition or incoherence
* High compute cost (cache disabled)
* Ethical risks if used for manipulation
* Identity control ≠ factual control

---


