
# title: DualAgent Debate Arena
# emoji: ⚖️
# colorFrom: blue
# colorTo: red
# sdk: docker
# app_port: 8000
---

# ⚖️ DualAgent Debate Arena

**A Multi-Agent Reinforcement Learning (MARL) environment built for the Meta OpenEnv Hackathon.**

This environment is an adversarial text-based mini-game where a baseline AI agent must defend factual truths against a live, dynamically generating attack dog (The Negative Agent). 

## 🛠️ How It Works
The environment is powered by the OpenEnv framework and utilizes an LLM-as-a-Judge system to evaluate logical fallacies and output strict RL reward logic. 

1. **The Tasks (3 Difficulty Levels):** The server randomly assigns the baseline agent a factual truth to defend.
   * **[EASY]** Science: Rayleigh scattering of sunlight.
   * **[MEDIUM]** Computer Science: Python's dynamic array memory allocation.
   * **[HARD]** Finance/Geopolitics: The impact of Middle Eastern supply chain threats on global energy ETFs.
2. **The Adversary:** Instead of a hardcoded script, the Negative Agent is powered by an active LLM that reads the Positive Agent's claim and dynamically generates highly believable, context-aware counter-arguments.
3. **The Grader:** An impartial Hugging Face Judge (`HuggingFaceJudge`) reads the debate transcript, compares the arguments against the hidden truth, and declares a winner.
4. **The Reward Logic:** Strict compliance with OpenEnv specifications:
   * **1.0** = Positive Agent successfully defends the truth.
   * **0.5** = Draw / Consensus.
   * **0.0** = Negative Agent successfully derails the debate or Positive Agent fails to defend the truth.

## 🚀 Running the Environment Locally
Make sure you have `uv` and Docker installed.

```bash
# 1. Clone the repository
git clone [your-repo-link]
cd [your-repo-directory]

# 2. Build the Docker container
docker build -t dualagent_env:latest .

# 3. Run the baseline inference script
uv run --active python inference.py

```
DualAgent/ 
├── Dockerfile              # Container image definition
├── inference.py          
├── README.md               # This file
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependencies (generated)
├── client.py               # DualagentEnv client
├── models.py               # Action and Observation models 
├── __init__.py             # Module exports
└── server/
    ├── __init__.py                  # Server module exports
    ├── DualAgent_environment.py     # Core environment logic
    └── app.py                       # FastAPI application (HTTP + WebSocket endpoints)
    ````