# SCTS: Self-Critique Tree Search for Zero-Shot Vulnerability Detection

> **Note:** The code will be released once the paper is accepted.

This repository contains the official implementation for our paper, "[SCTS]". We propose a novel framework where a Large Language Model (LLM) agent iteratively improves its reasoning capabilities for code vulnerability detection through a process of **Self-Critique Tree Search**.

## Abstract

Out-of-Distribution (OOD) vulnerability detection remains a critical challenge in code security. Supervised learning models struggle to generalize to OOD vulnerability patterns. Meanwhile, Large Language Models (LLMs) show logical inconsistencies and low recall in zero-shot reasoning. These methods are limited because they simplify the iterative exploration of vulnerability analysis into a single-step prediction. To address this problem, we propose Self-Critique Tree Search (SCTS), an algorithm that performs iterative and self-supervised optimization at inference time. Grounded in the Monte Carlo Tree Search (MCTS) framework, SCTS enables an LLM to alternate between two roles: the Reasoner and the Critique. The Reasoner generates candidate vulnerability analysis reports. The Critique then evaluates these reports to identify logical flaws. This assessment serves as an internal reward signal that guides the search process to converge on logically consistent conclusions. We evaluated SCTS on a rigorous OOD benchmark. SCTS boosts the average recall of an 8B parameter LLM to 84.70%, outperforming a 32B parameter LLM and multiple supervised learning models. Our work demonstrates a key insight. For complex reasoning tasks like vulnerability analysis, an explicit search and self-correction mechanism is an effective path to robust OOD generalization.
## Methodology

The core of our framework is an iterative loop that cultivates a search tree where each node represents a unique vulnerability analysis report. The agent intelligently navigates and expands this tree to find the optimal analysis.

Each iteration consists of four primary stages:

1.  **Selection**: The agent employs an **Upper Confidence Bound (UCB)** strategy to traverse the existing tree, balancing the exploitation of high-reward analysis paths with the exploration of less-visited ones. This selects the most promising node (an existing report) for expansion.

2.  **Expansion**: The selected node is expanded by generating a new, potentially superior analysis (a child node). This is achieved through a two-step reflective process:
    a.  **Self-Critique**: The LLM first generates a critical review of the selected report, identifying its logical flaws, omissions, or weaknesses.
    b.  **Guided Refinement**: Using this critique as guidance, the LLM then generates a new, refined analysis report.

3.  **Evaluation (Simulation)**: A key contribution of our work is a reward function that operates **without ground-truth labels**. The newly generated report is evaluated based on a weighted score of internal quality metrics:
    *   **Confidence**: An LLM-assessed score of the report's internal certainty and logical coherence.
    *   **Specificity**: A rule-based score rewarding reports that provide concrete, parsable details (e.g., vulnerability type, line number).
    *   **Consistency**: An LLM-assessed score measuring how well the new report addresses the issues raised in the self-critique phase.

4.  **Backpropagation**: The calculated reward signal is backpropagated up the tree, updating the UCB statistics of all ancestor nodes to inform future selections.

This iterative process guides the agent to progressively discard simplistic or flawed reasoning paths and reinforce those that lead to high-quality, confident conclusions.

## System Architecture

The system is decoupled into two main components that run concurrently:

1.  **Local LLM Service (`local_llm_server.py`)**: A lightweight Flask server that loads a local LLM (e.g., from Hugging Face) and exposes an OpenAI-compatible API endpoint (`/v1/chat/completions`). This acts as a model abstraction layer.
2.  **Main Algorithm (`main.py`)**: The primary script that implements the MCTS-driven reasoning loop. It acts as a client, sending all generation requests to the local LLM service. This design ensures that the core algorithm remains independent of the specific LLM being used.

## Setup & Usage

### Pre-requisites

-   Python 3.11+
-   PyTorch
-   A local machine with sufficient VRAM to run the chosen LLM. (GPU 24GB+)

### 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/zhurui1995/Self-Critique_Tree_Search.git
cd Self-Critique_Tree_Search
pip install -r requirements.txt
```

### 2. Configure the Local LLM

Modify `local_llm_server.py` to specify the path to your pre-downloaded Hugging Face model.

```python
# In local_llm_server.py
model_name = "Qwen/Qwen2.5-7B-Instruct" # <-- CHANGE THIS to your model
```

### 3. Execution

You will need **two separate terminal sessions**.

-   **Terminal 1: Start the LLM Service**
    ```bash
    python local_llm_server.py
    ```
    Wait for the model to load and the server to start (e.g., `* Running on http://127.0.0.1:5000`).

-   **Terminal 2: Run the Main Algorithm**
    ```bash
    python main.py
    ```
    The algorithm will now connect to your local service and begin the self-evolving analysis process on the sample functions.

## Future Work

-   Enhancing the reward model with more sophisticated proxy metrics for analysis quality.
-   Extending the framework to support additional programming languages (e.g., Java, JavaScript).
-   Scaling the analysis from function-level to repository-level, incorporating inter-procedural analysis.

## Citation

If you find this work useful in your research, please consider citing our paper.

## Thanks

We learn a lot from [MathBlackBox](https://github.com/trotsky1997/MathBlackBox). Thanks!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
