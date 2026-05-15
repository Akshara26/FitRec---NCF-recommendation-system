# FitRec: Personalized Fitness Activity Recommender

Gym members often stick to familiar workout routines, missing activities better suited to their fitness goals. A personalized recommendation system can surface the right workout type for each member based on their behavior — increasing engagement, retention, and training outcomes. This project builds an end-to-end Neural Collaborative Filtering (NCF) recommendation system trained on gym member workout data, with a retraining loop that adapts to new user behavior over time.

---

## Architecture

The model is based on **Neural Matrix Factorization (NeuMF)**, which combines two pathways:

```
User ID ──► User Embedding (GMF) ──► Element-wise Product ──►
Item ID ──► Item Embedding (GMF)                             │
                                                             ├──► Concat ──► Sigmoid ──► Score
User ID ──► User Embedding (MLP) ──► Concat ──► Hidden ──►  │
Item ID ──► Item Embedding (MLP)    Layers                  │
```

- **GMF pathway** — element-wise product of user and item embeddings captures linear interactions
- **MLP pathway** — concatenated embeddings passed through hidden layers captures non-linear interactions
- **NeuMF** — both pathways concatenated into a single sigmoid output (probability of interaction)

Implicit feedback is derived from workout frequency — members who train 4+ days/week are treated as engaged (rating=1), others as non-engaged (rating=0). This is closer to how production recommendation systems work than explicit ratings.

---

## Dataset
- **Source**: [Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)
- **Size**: 973 members, 4 workout types (Yoga, HIIT, Cardio, Strength)
- **Interaction matrix sparsity**: 75%
- **Implicit feedback signal**: Workout frequency ≥ 4 days/week = 1, else 0

---

## Results

| Metric | Popularity Baseline | NeuMF |
|--------|-------------------|-------|
| HR@3 | 0.2045 | 0.7692 |
| NDCG@3 | — | 0.5302 |

NeuMF is **4x better** than the popularity baseline on HR@3, justifying the use of personalized collaborative filtering over a simple popularity-based approach.

---

## Retraining & Drift Monitoring

Production recommendation systems degrade over time as user behavior shifts. Notebook 3 implements a retraining loop that:

1. **Simulates a new interaction batch** — 200 new interactions with a shifted workout distribution
2. **Detects drift** using `scipy.stats.ks_2samp()` comparing the new batch against the original training distribution
3. **Fine-tunes the model** for 5 epochs at a 10x smaller learning rate (1e-4) to avoid catastrophic forgetting
4. **Logs before/after metrics** to CSV for tracking model performance over time

### Drift Detection Result
| Metric | Value |
|--------|-------|
| KS Statistic | 0.1550 |
| P-value | 0.0002 |
| Drift Detected | ✅ Yes |

A p-value below 0.05 triggers a retraining run. In this case, a shift toward Cardio workouts was successfully detected.

---

## Project Structure

```
neural-collaborative-filtering/
├── notebook0_eda.ipynb              # Exploratory data analysis
├── notebook1_preprocessing.ipynb   # Data preprocessing + S3 upload
├── notebook2_training.ipynb        # Model training + evaluation
├── notebook3_retraining.ipynb      # Retraining loop + drift detection
├── src/
│   ├── neumf.py                    # NeuMF architecture
│   ├── gmf.py                      # GMF pathway
│   ├── mlp.py                      # MLP pathway
│   ├── engine.py                   # Training engine
│   └── data/
│       ├── gym_members_exercise_tracking.csv
│       ├── fitness_ratings.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── checkpoints/
│   ├── neumf_fitness_final.pt
│   └── neumf_fitness_retrained.pt
└── requirements.txt
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yihong-chen/neural-collaborative-filtering.git
cd neural-collaborative-filtering

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch pandas numpy scikit-learn boto3 papermill jupyter tensorboardX scipy matplotlib

# Run notebooks in order
# notebook0_eda.ipynb
# notebook1_preprocessing.ipynb
# notebook2_training.ipynb
# notebook3_retraining.ipynb
```

---

## Requirements
- Python 3.10+
- PyTorch 2.x
- pandas, numpy, scikit-learn
- boto3 (S3 upload)
- scipy (drift detection)
- matplotlib (EDA visualizations)
- tensorboardX

