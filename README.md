
# AQER Simulation and Evaluation Framework

This repository contains the full implementation of the AQER framework and associated simulation tools used in the research paper:

Yahav, I., Goldstein, A., Geva, T., Shehory, O., & Meir, S. (2025). **Quality control for crowd workers and for language models: A framework for free-text response evaluation with no ground truth. Management Science. (Accepted for publication).**

## Overview

The AQER (Automated Quality Evaluation based on textual Responses) framework is designed to evaluate the quality of crowd worker and LLM responses without requiring access to ground-truth data. This repository includes:

- The **AQER framework code** for processing and evaluating real-world data.
- The **AQER simulation tool** to test the framework's robustness in synthetic and stress-testing scenarios.
- Example **datasets** and evaluation scripts used in the study.

## Repository Structure

```
.
├── Framework/                  # AQER core implementation (to be added)
│   ├── AQER.py
├── simulation/                # Simulation code used for stress tests
│   ├── AQER_Simulator.py
│   └── AQER_simulation_usage_examples.py
├── datasets/                   # Real-world and semi-synthetic datasets (to be added)
├── README.md                   # Project documentation
```


## Quick Start

### Framework

The `framework/` directory includes the core AQER grading engine used to evaluate worker or model responses to free-text questions.

#### `AQER.py`

Implements the `AQER` class, which performs unsupervised quality assessment using an Expectation-Maximization (EM) approach.

```python
from framework.AQER import AQER

aqer = AQER(answers_embeddings=df, max_iterations=1000, threshold=1e-5)
grades, skill_levels = aqer.grades_expectation_maximization()
```

**Input**: A DataFrame with columns `["question", "worker", "x1", ..., "xn"]`, where each row corresponds to a worker's answer embedding.

**Output**:
- `grades`: Estimated quality scores for each worker
- `skill_levels`: Skill levels inferred through iterative optimization


### Simulations

To run a basic synthetic simulation:

```python
from AQER_Simulator import Scenario, Simulator

scenario = Scenario(num_questions=10, answer_dim=300)
scenario.add_workers(num_workers=20, standard_deviation=1.0, bias=0)
scenario.add_correct_answers()
scenario.add_worker_answers()

simulator = Simulator(max_iterations=1000, threshold=1e-5, simulation_repetitions=30)
avg_epoch_0, _, avg_with_iter, _ = simulator.run_simulation(scenario)

print(f"Initial: {avg_epoch_0:.3f}, Final: {avg_with_iter:.3f}")
```

To replicate the full Appendix F experiments from the paper, run the script:
```bash
python AQER_simulation_usage_examples.py
```

### Datasets (to be added)

We will also provide example datasets (synthetic and real-world) in the `datasets/` directory for reproducibility.

## 🧪 Requirements

- Python 3.7+
- NumPy
- pandas
- scikit-learn 
- matplotlib


##  Contact
inbalyahav@tauex.tau.ac.il
anatgo@ariel.ac.il
ge.tomer1@gmail.com
For contributions, issues, or questions, please contact the authors or submit a pull request.

---
© 2025 — AQER Project Team. Academic and research use only.
