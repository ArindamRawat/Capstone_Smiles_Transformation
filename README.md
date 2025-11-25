# 📘 **Transformer-Based SMILES Correction and Validation Framework**

## 🔬 **Project Title**  
**Transformer-Based SMILES Correction and Validation Model for AI-Generated Molecules**

### **Author**  
**Arindam Rawat — Bennett University**

---

# 📄 **1. Project Overview**

Deep learning molecular generative models frequently produce **invalid or chemically inconsistent SMILES strings**. These errors break downstream tasks such as QSAR modeling, docking, ADMET prediction, and drug discovery pipelines.

This project presents a **Transformer-based correction system** that:

- Detects invalid or corrupted SMILES  
- Repairs them using an encoder–decoder Transformer  
- Validates chemical correctness via RDKit  
- Measures similarity using Tanimoto & edit-distance metrics  
- Ensures corrected SMILES remain chemically meaningful  

This repository includes:

- Full **research paper (LaTeX)**  
- End-to-end **model implementation**  
- Dataset instructions  
- Training and evaluation script  
- Academic research poster  
- All supporting utilities  

---

# 🧠 **2. Core Idea**

The model learns a supervised mapping:

Corrupted / Invalid SMILES → Valid SMILES

Inspired by grammar-correction Transformers in NLP, but adapted for **chemical language modeling**.

---

# 🏗️ **3. System Architecture**

## **3.1 High-Level Pipeline**
Valid SMILES → Corruption Engine → Paired Dataset  
→ Transformer Encoder–Decoder Model  
→ Greedy / Beam Search Output  
→ RDKit Validation  
→ Final Corrected SMILES  

## **3.2 Model Architecture**
- Transformer Encoder (6 layers)
- Transformer Decoder (6 layers)
- Multi-head attention: 8 heads  
- Hidden size: 512  
- Feed-forward dimension: 2048  
- Autoregressive decoding  
- Beam search decoding  

## **3.3 Tokenization**
Regex extraction of:
- Atoms (C, N, O…)
- Multi-character atoms (Cl, Br, [NH+])
- Bonds (=, #, -)
- Branches
- Ring indices

Vocabulary size: ~60–120 tokens.

## **3.4 Chemical Validation**
- RDKit sanitization  
- Morgan fingerprints  
- Tanimoto similarity scoring  

---

# 📊 **4. Results**

| Metric | Score |
|--------|--------|
| Validity Rate | 0.8118 |
| Reconstruction Accuracy | 0.803 |
| Tanimoto Similarity | 0.8003 |
| Normalized Edit Similarity | 0.8005 |

---

# 📁 **5. Repository Structure**
```
.
├── capstone_main_transformed.py
├── capstone_main.py
├── research_paper/
├── data/
│   ├── train_pairs.csv
│   ├── val_pairs.csv
│   └── test_pairs.csv
├── SMILES_Poster.png
├── test_predictions.csv
└── README.md
```

---

# 📦 **6. Installation**

```
pip install -r requirements.txt
```

requirements.txt:
```
torch
pandas
numpy
tqdm
rdkit-pypi
scikit-learn
```

---

# 🧪 **7. Dataset Format**

train_pairs.csv / val_pairs.csv / test_pairs.csv:
```
corrupted,correct
C(C(C,C(C)C
CN(Cl(,CN(Cl)
```

---

# 🚀 **8. Training**

```python
from capstone_main_transformed import train_and_validate
train_and_validate()
```

---

# 🔍 **9. Inference**

```python
from capstone_main_transformed import encode_smiles, beam_search_decode, decode_smiles_from_ids, model, stoi
src_ids = encode_smiles("C(C(C", stoi)
pred_ids = beam_search_decode(model, src_ids)
print(decode_smiles_from_ids(pred_ids))
```

---

# 📈 **10. Evaluation**

```python
from capstone_main_transformed import evaluate_on_test
evaluate_on_test()
```

---

# 🖼️ **11. Poster**

Included:
- SMILES_Poster.png

---

# 🧩 **12. Future Work**
- SELFIES-based constraints  
- Grammar-constrained decoding  
- ChemBERTa / SMILES-BART  
- Hybrid neural–symbolic models  
- Expansion to QSAR, docking, ADMET, toxicology  

---

# 🔖 **13. Citation**

```
@article{rawat2025smilescorrection,
  title={Transformer-Based SMILES Correction and Validation Model for AI-Generated Molecules},
  author={Arindam Rawat},
  year={2025},
  institution={Bennett University}
}
```

---

# 📬 **14. Contact**
**Arindam Rawat**  
Department of Computer Science  
Bennett University  
Email: e22cseu0599@bennett.edu.in
