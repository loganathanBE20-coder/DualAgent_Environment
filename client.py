# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dualagent Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DualagentAction, DualagentObservation


class DualagentEnv(
    EnvClient[DualagentAction, DualagentObservation, State]
):
    """
    Client for the Dualagent Environment.
    Updated for the Adversarial Debate AI.
    """

    def _step_payload(self, action: DualagentAction) -> Dict:
        """
        Convert DualagentAction to JSON payload for step message.
        """
        # UPDATED: Sending the Claim and Reasoning instead of 'message'
        return {
            "claim": action.claim,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DualagentObservation]:
        """
        Parse server response into StepResult[DualagentObservation].
        """
        obs_data = payload.get("observation", {})
        
        # UPDATED: Mapping the new Debate fields from the server
        observation = DualagentObservation(
            transcript=obs_data.get("transcript", ""),
            negative_counter=obs_data.get("negative_counter", ""),
            judge_winner=obs_data.get("judge_winner", "none"),
            done=payload.get("done", False),
            reward=float(payload.get("reward") or 0.0)
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )