
# AQER Simulation and Evaluation Framework

This repository contains the full implementation of the AQER framework and associated simulation tools used in the research paper:

**"Quality Control for Crowd Workers and for Language Models: A Framework for Free-Text Response Evaluation with No Ground Truth"**

## 📄 Overview

The AQER (Automated Quality Evaluation based on textual Responses) framework is designed to evaluate the quality of crowd worker and LLM responses without requiring access to ground-truth data. This repository includes:

- The **AQER simulation tool** to test the framework's robustness in synthetic and stress-testing scenarios.
- The **AQER framework code** for processing and evaluating real-world data.
- Example **datasets** and evaluation scripts used in the study.

## 📦 Repository Structure

```
.
├── framework/                  # AQER core implementation (to be added)
├── simulations/                # Simulation code used for stress tests
│   ├── AQER_Simulator.py
│   └── AQER_simulation_usage_examples.py
├── datasets/                   # Real-world and semi-synthetic datasets (to be added)
├── README.md                   # Project documentation
```

## 🧠 Key Features

- **No Ground Truth Required**: AQER works with only worker- or model-generated responses.
- **Flexible Simulation Environment**: Generate test cases to probe AQER's limits.
- **EM-Based Skill Inference**: Uses Expectation-Maximization to estimate worker/model quality.
- **Modular Design**: Easy to plug in new datasets or scoring functions.

## ▶️ Quick Start

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

### Framework (to be added)

The `framework/` directory will include code for real-world data evaluation using AQER. This will support embedding models, response grading, and report generation.

### Datasets (to be added)

We will also provide example datasets (synthetic and real-world) in the `datasets/` directory for reproducibility.

## 📊 Output Metrics

Simulation and framework evaluations report:
- `avg_epoch_0`: Correlation of worker scores after one iteration
- `avg_with_iter`: Correlation after convergence
- Evaluation is based on cosine similarity with true or consensus embeddings

## 🧪 Requirements

- Python 3.7+
- NumPy
- pandas
- scikit-learn
- matplotlib

Install with:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## 📚 Citation

If you use this code, please cite our work:

```
[APA/BibTeX citation of the paper]
```

## 🤝 Contributing

For contributions, issues, or questions, please contact the authors or submit a pull request.

---
© 2025 — AQER Project Team. Academic and research use only.
