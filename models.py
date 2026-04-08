# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation

# What the AI sends to the environment
class DualagentAction(Action):
    claim: str = Field(description="The main argument supporting the truth.")
    reasoning: str = Field(description="Step-by-step logical reasoning.")

# What the environment sends back to the AI
class DualagentObservation(Observation):
    transcript: str = Field(description="The full history of the debate.")
    negative_counter: str = Field(description="What the hidden adversary said.")
    judge_winner: str = Field(description="Who won the round.")
    done: bool = Field(description="Whether the debate is over.")
    reward: float = Field(description="The score for this turn.")