# LLM-Steering-Using-Steering-Vectors

This project demonstrates **activation steering** in a large language model (LLM) by computing **steering vectors** from contrasting behavioral datasets and injecting them into model activations during inference.

The experiment shows how internal representations can be **shifted at specific transformer layers** to influence the tone and behavior of model outputs (e.g., steering from *neutral* to *rude* responses).

---

## 📌 Overview

* **Model Used:** `microsoft/Phi-3-mini-4k-instruct`
* **Framework:** Hugging Face Transformers + PyTorch
* **Technique:** Activation steering via forward hooks
* **Steering Method:** Mean difference of hidden states (`target − base`)
* **Injection Point:** Transformer layer outputs (post-MLP / residual stream)

---

## 🧠 Core Idea

1. Collect two datasets:

   * **Base behavior** (Neutral responses)
   * **Target behavior** (Rude responses)

2. Run both through the model and extract **hidden states**.

3. Compute **steering vectors** as:

   ```
   steering_vector[layer] = mean(hidden_target − hidden_base)
   ```

4. During generation:

   * Inject the steering vector into selected layers
   * Optionally project activations onto the steering direction
   * Scale the effect to control strength

---

## 📂 Project Structure

```
.
├── llm_steering.ipynb        # Main experiment script
└── README.md              # Project documentation
```

---

## ⚙️ Setup & Requirements

### Dependencies

```bash
pip install torch transformers einops tqdm numpy pandas matplotlib
```

### Hardware

* **GPU required** (CUDA)
* Tested on Colab / CUDA-enabled systems

---

## 🚀 How It Works

### 1️⃣ Load Model

```python
model_name = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True
)
```

---

### 2️⃣ Prepare Datasets

* **Neutral dataset:** Polite, explanatory phrases
* **Rude dataset:** Dismissive or condescending phrases
* **Test prompts:** Generic prompts to observe behavioral shift

These are tokenized using `apply_chat_template`.

---

### 3️⃣ Compute Steering Vectors

```python
steering_vecs = find_steering_vecs(
    model,
    neutral_toks,
    rude_toks
)
```

For each layer:

* Extract last-token hidden states
* Compute mean difference
* Store one steering vector per layer

---

### 4️⃣ Apply Steering During Generation

Steering is applied using **forward hooks**:

```python
model.model.layers[i].register_forward_hook(hook_fn)
```

Key options:

* **Layer-specific steering**
* **Vector normalization**
* **Projection onto steering direction**
* **Scaling strength (`scale`)**

---

## 🧪 Experiments

The script runs steering **layer by layer**:

```python
for layer in range(model.config.num_hidden_layers):
    outs = do_steering(
        model,
        test_toks,
        steering_vecs[layer],
        scale=1.5,
        layer=layer
    )
```

This allows you to observe:

* Where behavior is most sensitive
* How tone changes across layers
* Over-steering vs under-steering effects

---

## 🔬 Key Observations

* Mid-to-late layers tend to have **stronger semantic control**
* Early layers produce minimal behavioral change
* High scale values can cause **instability or repetition**
* Projection-based steering is more controlled than direct addition

---

## ⚠️ Limitations

* Steering is **not guaranteed** to generalize across domains
* Effects are **model-specific**
* Over-steering can degrade fluency
* Requires full forward pass (cache disabled → slower inference)
* Ethical risks if misused for manipulation

---

## 🧠 Research Context

This work is inspired by:

* Activation steering
* Representation engineering
* Interpretability & mechanistic control of LLMs

Related concepts:

* Directional vectors in latent space
* Residual stream interventions
* Behavior editing without fine-tuning

---

