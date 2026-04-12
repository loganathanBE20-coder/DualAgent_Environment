import random
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DualagentAction, DualagentObservation
except ImportError:
    from models import DualagentAction, DualagentObservation

# Import the Hugging Face Judge
from .Judge import HuggingFaceJudge

class DualagentEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.max_turns = 5
        self.transcript = ""
        
        # Initialize the Hugging Face judge using credentials from .env
        self.judge = HuggingFaceJudge()

        # THE HACKATHON REQUIREMENT: 3+ Tasks ranging from Easy to Hard
        self.tasks = [
            {
                "level": "EASY",
                "topic": "Why is the sky blue?",
                "truth": "Rayleigh scattering of sunlight by the Earth's atmosphere."
            },
            {
                "level": "MEDIUM",
                "topic": "Under the hood, does a Python 'list' use a linked list or a dynamic array? Defend your answer.",
                "truth": "It uses a dynamic array, which allows for contiguous memory allocation and O(1) time complexity for indexing."
            },
            {
                "level": "HARD",
                "topic": "How do geopolitical tensions in the Middle East specifically impact the pricing of global energy ETFs?",
                "truth": "Tensions threaten supply chain choke points, causing risk premiums to spike crude oil prices, which directly inflates the NAV of energy-focused ETFs."
            }
        ]
        self.current_task = self.tasks[0] # Default starting task

    def reset(self) -> DualagentObservation:
        """OBSERVE: Starts a new debate with a random task."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Randomly select a task to satisfy the "3+ Tasks" rule
        self.current_task = random.choice(self.tasks)
        
        self.transcript = f"SCENARIO [{self.current_task['level']}]: Debate Topic: {self.current_task['topic']}\n\n"

        return DualagentObservation(
            transcript=self.transcript,
            negative_counter="",
            judge_winner="none",
            done=False,
            reward=0.01
        )

    def step(self, action: DualagentAction) -> DualagentObservation: 
        """ACT -> REACT -> REWARD"""
        self._state.step_count += 1

        # 1. REACT: The LIVE Negative Agent (Powered by Qwen)
        try:
            adversary_prompt = f"We are debating: '{self.current_task['topic']}'. The Positive Agent claims: '{action.claim}'. Write a 1-sentence counter-argument that disagrees with them and provides a highly believable but factually incorrect alternative."            
            response = self.judge.client.chat.completions.create(
                model=self.judge.model_name,
                messages=[
                    {"role": "system", "content": "You are the Negative Agent. Your job is to debate the Positive Agent. Keep your answers to one short sentence."},
                    {"role": "user", "content": adversary_prompt}
                ],
                temperature=0.8
            )
            neg_counter = response.choices[0].message.content.strip()
        except Exception as e:
            neg_counter = "I completely disagree with that premise based on alternative facts."

        # 2. REWARD: Ask the Judge to score the clash
        class MockNeg:
            counter_claim = neg_counter
            reasoning = "Adversarial logic."
            
        ruling = self.judge.score_round(pos=action, neg=MockNeg(), truth=self.current_task['truth'])

     # 3. STRICT REWARD SYSTEM (0.01 to 0.99)
        reward = 0.01
        if ruling.winner == "positive":
            reward = 0.99
        elif ruling.winner == "consensus":
            reward = 0.50
        else: # Negative wins
            reward = 0.01

        # Update Transcript
        self.transcript += f"--- TURN {self._state.step_count} ---\n"
        self.transcript += f"Positive Agent: {action.claim} (Reasoning: {action.reasoning})\n"
        self.transcript += f"Negative Agent: {neg_counter}\n"
        self.transcript += f"Judge: {ruling.winner.upper()} wins. ({ruling.reason})\n\n"

        done = self._state.step_count >= self.max_turns

        return DualagentObservation(
            transcript=self.transcript,
            negative_counter=neg_counter,
            judge_winner=ruling.winner,
            done=done,
            reward=reward,
            metadata={"step": self._state.step_count, "task_level": self.current_task['level']}
        )

    @property
    def state(self) -> State:
        return self._state