#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The entire framework consists of four roles: the reasoner, the affirmative side, the negative side, and the referee
Debate process

The reasoner generates the result, the affirmative and negative sides debate, and finally the referee makes a judgment
The reasoner makes adjustments based on the judgment of the referee
"""
from typing import Any, Dict, List
from lwj_tools.llms.chain import LLMChain, ChainResult
from lwj_tools.llms.client import LLMClientGroup

from .constant import ATOMIC_RELATIONS
from .prompts import (
    RPSysPromptTemplate,
    RPUserPromptTemplate,
    RPChairmanSysPromptTemplate,
    RPChairmanUserPromptTemplate,
    RPAffirmativeSysPromptTemplate,
    RPAffirmativeUserPromptTemplate,
    RPNegativeSysPromptTemplate,
    RPNegativeUserPromptTemplate,
)


def _restore_messages(response: str, role: str, history: List[dict]):
    history.append({"role": role, "content": str(response)})


class DebateFramework:
    writer_sys_prompt = RPSysPromptTemplate()
    affirmative_sys_prompt = RPAffirmativeSysPromptTemplate()
    negative_sys_prompt = RPNegativeSysPromptTemplate()
    chairman_sys_prompt = RPChairmanSysPromptTemplate()

    writer_user_prompt = RPUserPromptTemplate()
    affirmative_user_prompt = RPAffirmativeUserPromptTemplate()
    negative_user_prompt = RPNegativeUserPromptTemplate()
    chairman_user_prompt = RPChairmanUserPromptTemplate()

    def __init__(
        self,
        writer_client: LLMClientGroup,
        affirmative_client: LLMClientGroup,
        negative_client: LLMClientGroup,
        chairman_client: LLMClientGroup,
        debate_rounds: int = 1,
        max_rewrite_num: int = 3,
        writer_generation_config: dict = None,
        affirmative_generation_config: dict = None,
        negative_generation_config: dict = None,
        chairman_generation_config: dict = None,
    ):
        assert debate_rounds > 0 and max_rewrite_num > 0

        if writer_generation_config is None:
            writer_generation_config = {}

        if affirmative_generation_config is None:
            affirmative_generation_config = {}

        if negative_generation_config is None:
            negative_generation_config = {}

        if chairman_generation_config is None:
            chairman_generation_config = {}

        self.writer_chain = LLMChain(
            client_group=writer_client,
            prompt_template=self.writer_user_prompt,
            **writer_generation_config,
        )
        self.affirmative_chain = LLMChain(
            client_group=affirmative_client,
            prompt_template=self.affirmative_user_prompt,
            **affirmative_generation_config,
        )
        self.negative_chain = LLMChain(
            client_group=negative_client,
            prompt_template=self.negative_user_prompt,
            **negative_generation_config,
        )
        self.chairman_chain = LLMChain(
            client_group=chairman_client,
            prompt_template=self.chairman_user_prompt,
            **chairman_generation_config,
        )

        self.max_rewrite_num = max_rewrite_num
        self.debate_rounds = debate_rounds

    def initial_writer_messages(self, data: Any, history: List[dict]):
        # add system prompt
        history.append(
            {"role": "system", "content": self.writer_sys_prompt.generate_prompt(data)}
        )

    def initial_affirmative_messages(self, data: Any, history: List[dict]):
        # add system prompt
        history.append(
            {
                "role": "system",
                "content": self.affirmative_sys_prompt.generate_prompt(data),
            }
        )

    def initial_negative_messages(self, data: Any, history: List[dict]):
        # add system prompt
        history.append(
            {
                "role": "system",
                "content": self.negative_sys_prompt.generate_prompt(data),
            }
        )

    def initial_chairman_messages(self, data: Any, history: List[dict]):
        # add system prompt
        history.append(
            {
                "role": "system",
                "content": self.chairman_sys_prompt.generate_prompt(data),
            }
        )

    def get_writer_result(self, writer_result: ChainResult) -> dict:
        return writer_result.result

    def chairman_pass_func(self, chairman_result: dict) -> bool:
        if chairman_result.get("result", 0) == 1:
            return True
        if "suggestion" in chairman_result:
            return len(chairman_result["suggestion"]) == 0
        return False

    def post_edit_result(
        self, writer_chain_result: ChainResult, chairman_chain_result: dict
    ):
        inference_dict = writer_chain_result.result
        need_to_revise = chairman_chain_result.get("suggestion", {})

        final_ans = {}
        for relation in ATOMIC_RELATIONS:
            if relation in need_to_revise or relation not in inference_dict:
                continue
            final_ans[relation] = inference_dict[relation]
        return final_ans

    def __call__(self, data: Any):
        all_messages: Dict[str, List[dict]] = {
            "writer": [],
            "affirmative": [],
            "negative": [],
            "chairman": [],
        }

        self.initial_writer_messages(data, all_messages["writer"])
        self.initial_affirmative_messages(data, all_messages["affirmative"])
        self.initial_negative_messages(data, all_messages["negative"])
        self.initial_chairman_messages(data, all_messages["chairman"])

        writer_chain_result, chairman_chain_result = None, None

        for i in range(1, self.max_rewrite_num + 1):
            writer_chain_result: ChainResult = self.writer_chain(
                data,
                history=all_messages["writer"],
            )
            _restore_messages(
                response=writer_chain_result.prompt,
                role="user",
                history=all_messages["writer"],
            )
            _restore_messages(
                response=writer_chain_result.response,
                role="assistant",
                history=all_messages["writer"],
            )

            # start debate
            debate_history = []
            for debate_round in range(1, self.debate_rounds + 1):
                affirmative_chain_result = self.affirmative_chain(
                    data,
                    str(writer_chain_result.result),
                    history=all_messages["affirmative"],
                )
                _restore_messages(
                    response=affirmative_chain_result.prompt,
                    role="user",
                    history=all_messages["affirmative"],
                )
                _restore_messages(
                    response=affirmative_chain_result.result,
                    role="assistant",
                    history=all_messages["affirmative"],
                )
                _restore_messages(
                    response=f"The Affirmative Side Spoke:\n{affirmative_chain_result.result}",
                    role="user",
                    history=all_messages["negative"],
                )

                negative_chain_result = self.negative_chain(
                    data,
                    str(writer_chain_result.result),
                    history=all_messages["negative"],
                )
                _restore_messages(
                    response=negative_chain_result.prompt,
                    role="user",
                    history=all_messages["negative"],
                )
                _restore_messages(
                    response=negative_chain_result.result,
                    role="assistant",
                    history=all_messages["negative"],
                )

                debate_history.append(
                    f"""**Round {debate_round}**\nThe Affirmative Side Spoke:\n{affirmative_chain_result.response}"""
                )

                # not the last round
                if debate_round != self.debate_rounds:
                    _restore_messages(
                        response=f"The Negative Side Spoke:\n{negative_chain_result.result}",
                        role="user",
                        history=all_messages["affirmative"],
                    )

            _restore_messages(
                response="**Debate History**\n" + "\n".join(debate_history),
                role="user",
                history=all_messages["chairman"],
            )

            chairman_chain_result = self.chairman_chain(
                data,
                str(writer_chain_result.result),
                history=all_messages["chairman"],
            )
            _restore_messages(
                response=chairman_chain_result.prompt,
                role="user",
                history=all_messages["chairman"],
            )
            _restore_messages(
                response=chairman_chain_result.response,
                role="assistant",
                history=all_messages["chairman"],
            )

            if self.chairman_pass_func(chairman_chain_result.result):
                return writer_chain_result.result, all_messages

            _restore_messages(
                response=chairman_chain_result.result.get(
                    "suggestion", chairman_chain_result.response
                ),
                role="user",
                history=all_messages["writer"],
            )

        final_ans = self.post_edit_result(
            writer_chain_result, chairman_chain_result.result
        )

        return final_ans, all_messages
