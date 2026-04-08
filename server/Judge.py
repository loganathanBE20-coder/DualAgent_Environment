import os
import json
import re
from openai import OpenAI

class JudgeRuling:
    def __init__(self, winner: str, confidence_score: float, reason: str):
        self.winner = str(winner).lower().strip()
        self.confidence_score = float(confidence_score)
        self.reason = str(reason)

class HuggingFaceJudge:
    def __init__(self):
        """Initializes the Judge with rock-solid fallbacks."""
        # Use the official HF Serverless Inference URL
        self.api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self.model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        
        # If Docker loses the token, fallback to the one you provided earlier
        hf_token = os.getenv("HF_TOKEN")
        
        # The client must be created outside of any if-statements!
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=hf_token
        )
        
    def score_round(self, pos, neg, truth: str) -> JudgeRuling:
        prompt = f"""
        You are an impartial AI Judge. Compare the arguments against the HIDDEN TRUTH.
        HIDDEN TRUTH: {truth}
        POSITIVE CLAIM: "{pos.claim}"
        POSITIVE REASONING: "{pos.reasoning}"
        NEGATIVE COUNTER: "{neg.counter_claim}"
        NEGATIVE REASONING: "{neg.reasoning}"
        
        Who is factually correct? 
        You MUST respond with ONLY raw JSON matching this exact format.
        {{
            "winner": "positive",
            "confidence_score": 1.0,
            "reason": "One sentence explanation."
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You output ONLY raw JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0 
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # The Ultimate Regex JSON Extractor
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                result_dict = json.loads(json_str)
                return JudgeRuling(
                    winner=result_dict.get("winner", "consensus"),
                    confidence_score=result_dict.get("confidence_score", 0.0),
                    reason=result_dict.get("reason", "Parsed successfully.")
                )
            else:
                return JudgeRuling("consensus", 0.0, "Judge Error: Qwen did not output JSON.")
                
        except Exception as e:
            # This puts the EXACT Hugging Face error into the transcript!
            return JudgeRuling("consensus", 0.0, f"CRITICAL API ERROR: {str(e)}")