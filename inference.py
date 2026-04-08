import asyncio
import os
import sys
import textwrap
import json
from typing import List, Optional

from openai import OpenAI, api_key

# It will import your newly updated models!
from DualAgent import DualagentAction, DualagentEnv

IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

if not API_KEY:
    sys.exit(
        "ERROR: No API key found. Set OPENAI_API_KEY, HF_TOKEN, or API_KEY in your environment. "
        "If you are using uv, run `uv run --active python inference.py` or activate your venv first."
    )
    
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "adversarial-debate")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "dualagent_env")
MAX_STEPS = 5
TEMPERATURE = 0.7
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

# Our environment gives max +1.0 per step if the Positive Agent wins perfectly.
MAX_TOTAL_REWARD = MAX_STEPS * 1.0

# UPDATED: We tell Qwen it is the Positive Agent defending the truth!
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the Positive Agent in a debate. 
    Your goal is to defend the truth based on the scenario provided.
    You MUST respond strictly in JSON format with two keys:
    {
        "claim": "Your main argument",
        "reasoning": "Your step-by-step logic"
    }
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, transcript: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current Debate Transcript: {transcript}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        
        Generate your next JSON response defending the truth.
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, transcript: str, last_reward: float, history: List[str]) -> DualagentAction:
    user_prompt = build_user_prompt(step, transcript, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            response_format={"type": "json_object"} 
        )
        # Parse the JSON safely
        text = completion.choices[0].message.content or "{}"
        data = json.loads(text)
        
        # Safely extract BOTH fields, providing defaults if Qwen hallucinates
        return DualagentAction(
            claim=data.get("claim", "The truth is undeniable."),
            reasoning=data.get("reasoning", "Basic facts support this.")
        )
        
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # CRITICAL FIX: If the API fails, we MUST provide BOTH fields to satisfy Pydantic
        return DualagentAction(
            claim="API Error: Could not connect to model.", 
            reasoning="Fallback reasoning triggered to prevent server crash."
        )
    
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# This dynamically starts your Docker container AND hands it the Meta API keys!
    env = await DualagentEnv.from_docker_image(
        IMAGE_NAME or "dualagent_env:latest", # Added a fallback just in case!
        env_vars={
            "HF_TOKEN": API_KEY,           # Uses the API_KEY variable from the top
            "API_BASE_URL": API_BASE_URL,  # Uses the dynamic variable from the top
            "MODEL_NAME": MODEL_NAME       # Uses the dynamic variable from the top
        }
    )
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        
        # Pulling the transcript instead of echoed_message
        last_transcript = result.observation.transcript
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Let Qwen AI generate the claim and reasoning automatically!
            action = get_model_action(client, step, last_transcript, last_reward, history)

            result = await env.step(action)           
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            
            # Tracking the debate transcript
            last_transcript = obs.transcript
            last_reward = reward

            print("\n=== SERVER DEBATE TRANSCRIPT ===")
            print(last_transcript)
            print("================================\n")

            # Format action for the log
            action_str = f"claim('{action.claim[:20]}...')"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action.claim[:30]} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())