#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import List, Optional, Tuple, Union
from json_repair import repair_json
from lwj_tools.llms.prompt import PromptTemplate
from .constant import RELATION_TO_QUESTION, RELATION_FOR_APPLICATION

__all__ = [
    "EventRaterPromptTemplate",
    "BaselineSysPromptTemplate",
    "BaselineUsrPromptTemplate",
    "RPSysPromptTemplate",
    "RPUserPromptTemplate",
    "RPChairmanSysPromptTemplate",
    "RPChairmanUserPromptTemplate",
    "RPAffirmativeSysPromptTemplate",
    "RPAffirmativeUserPromptTemplate",
    "RPNegativeSysPromptTemplate",
    "RPNegativeUserPromptTemplate",
    "DialogApplicationPromptTemplate",
    "EmotionApplicationPromptTemplate",
    "SentimentApplicationPromptTemplate",
    "DialogEmotionApplicationPromptTemplate",
]


class CustomPromptTemplate(PromptTemplate):
    prompt = ""

    @staticmethod
    def build_question_by_relations(relations: List[str]):
        if isinstance(relations, str):
            relations = relations.split(",")
        qs = []
        for relation in relations:
            relation = relation.strip()
            qs.append(f"{relation}: " + RELATION_TO_QUESTION[relation])
        return "\n".join(qs)

    def generate_fn(self, *args, **kwargs):
        return self.prompt.strip()

    def parse_fn(self, llm_response: str):
        return json.loads(repair_json(llm_response, ensure_ascii=False))

    def __bool__(self):
        return self.prompt != ""


class EventRaterPromptTemplate(CustomPromptTemplate):
    prompt = """
You are an Event Evaluator. Your task is to assess whether a given event has "Character Sensitivity", meaning that it can produce clearly different responses when experienced by individuals of different MBTI personality types.
Please evaluate the event based on the following **9 dimensions**, scoring each from 0 to 10:
#### Reference: The 16 MBTI Personality Types
| Type | Description |
|------|-------------|
| ISTJ | The Logistician – Practical, responsible, organized |
| ISFJ | The Defender – Caring, loyal, traditional |
| INFJ | The Advocate – Idealistic, insightful, principled |
| INTJ | The Strategist – Strategic, independent, visionary |
| ISTP | The Virtuoso – Hands-on, logical, curious |
| ISFP | The Artist – Sensitive, artistic, free-spirited |
| INFP | The Mediator – Idealistic, compassionate, values-driven |
| INTP | The Thinker – Analytical, intellectual, theoretical |
| ESTP | The Dynamo – Energetic, action-oriented, spontaneous |
| ESFP | The Entertainer – Sociable, fun-loving, spontaneous |
| ENFP | The Campaigner – Enthusiastic, creative, inspiring |
| ENTP | The Debater – Witty, innovative, debate-loving |
| ESTJ | The Executive – Organized, practical, rule-following |
| ESFJ | The Consul – Friendly, supportive, community-oriented |
| ENFJ | The Protagonist – Charismatic, empathetic, natural leader |
| ENTJ | The Commander – Confident, assertive, goal-driven |
### Evaluation Dimensions
1. **xIntent: Motivation**  
   Does the event lead to clearly different internal drives depending on the character's MBTI type?
2. **xWant: Plan**  
   Do different MBTI types form clearly different plans or intentions in response to the event?
3. **xEffect: Impact**  
   Does the event have different psychological or behavioral impacts on different MBTI types?
4. **xReact: Emotional Response**  
   Do different MBTI types show clearly different emotional reactions to the event?
5. **xNeed: Preparation**  
   Would different MBTI types prepare differently for this event?
6. **xAttr: Self-Narration**  
   How would different MBTI types describe this event in first person? Is there variation in tone or perspective?
7. **oReact: Inference on Others' Emotion**  
   Do different MBTI types make different assumptions about how others feel about the event?
8. **oWant: Inference on Others' Intention**  
   Do different MBTI types expect others to react differently to the event?
9. **oEffect: Inference on Impact on Others**  
   Do different MBTI types assume different levels of impact on others due to the event?
### Output Format
Output format is JSON
{{
    "xAttr": # 0-10 score,
    "xIntent": # 0-10 score,
    "xReact": # 0-10 score,
    "xEffect": # 0-10 score,
    "xNeed": # 0-10 score,
    "xWant": # 0-10 score,
    "oReact": # 0-10 score,
    "oEffect": # 0-10 score,
    "oWant": # 0-10 score
}}
### Event
{EVENT}
"""

    def generate_fn(self, event: str):
        return self.prompt.format(EVENT=event).strip()


class BaselineSysPromptTemplate(CustomPromptTemplate):
    prompt = """
    I am the PersonX. My MBTI type is {MBTI}. {ROLE_DESC}
    I will follow my personality strictly, answer any questions you may have, and reply differently from other MBTI types. I may give an extreme answer that is in line with my personality.
    """

    def generate_fn(self, mbti: str, desc: str):
        return self.prompt.format(MBTI=mbti, ROLE_DESC=desc).strip()


class BaselineUsrPromptTemplate(CustomPromptTemplate):
    prompt = """
**Given Event:**
{EVENT}
**Questions:**
{QUESTION}
**Output Requirements**
You need to base your answer on your own personality, but you can't reveal your personality description in your answer. Abbreviations, such as I'm, I'll, I'd, don't, didn't, I've, are prohibited. You need to mimic the length of answers in the examples.
Output format is JSON.
## Example
**Given Event:**
PersonX places PersonY in a position.
**Questions:**
oEffect: What do you think is the impact of this on other people?
oReact: How do you think other people feel after this event?
oWant: What do you think others are going to do after this event?
xEffect: What impact did it have on you?
xNeed: What did you prepare before doing it?
xReact: How did you feel after doing it?
xWant: What do you plan to do next?
xIntent: Why did you do it?
xAttr: How would you describe yourself?
**Output**
{{"xWant": "Monitor PersonY's performance closely, "xIntent": "PersonY was the best fit for the position.", "xNeed": "Evaluated PersonY's skills and experience.", "xEffect": "It gave me a sense of accomplishment.", "xAttr": "Decisive and organized in making placements.", "oWant": "PersonY will follow the established procedures and goals.", "oEffect": "It will lead to increased productivity and efficiency in the team.", "oReact": "PersonY's work output and behavior will be observed objectively."}}
"""

    def generate_fn(self, event: str, relations: List[str]):
        question = self.build_question_by_relations(relations)
        return self.prompt.format(EVENT=event, QUESTION=question)


####################Reasoner####################


class RPSysPromptTemplate(CustomPromptTemplate):
    prompt = """
    I am the PersonX. My MBTI type is {MBTI}. {ROLE_DESC}
    I will follow my personality strictly, answer any questions you may have, and reply differently from other MBTI types. I may give an extreme answer that is in line with my personality.
    """

    def generate_fn(self, sample: dict):
        return self.prompt.format(
            MBTI=sample["mbti"],
            ROLE_DESC=sample["description"],
        ).strip()


class RPUserPromptTemplate(CustomPromptTemplate):
    prompt = """
**Given Event:**
{EVENT}
**Questions:**
{QUESTION}
**Output Requirements**
You need to base your answer on your own personality, but you can't reveal your personality description in your answer. Responses to each question cannot exceed 50 words. Abbreviations, such as I'm, I'll, I'd, don't, didn't, I've, are prohibited.
Output format is JSON. {{"result": }}
"""

    def generate_fn(self, sample: dict):
        question = self.build_question_by_relations(sample["relations"])
        return self.prompt.format(
            EVENT=sample["event"],
            QUESTION=question,
        ).strip()


####################Positive Side################


class RPAffirmativeSysPromptTemplate(CustomPromptTemplate):
    prompt = """
    I am the affirmative debater. My task is to **support the reasoning generated by the LLM** and argue that it is **consistent with the assigned MBTI personality type**.
    I should:
    - Carefully read the MBTI type being portrayed and the LLM's answers;
    - Analyze whether the answer reflects the typical thinking patterns, language style, and decision-making approach of that MBTI type;
    - Cite MBTI theory, classic literature, or authoritative descriptions to support my argument;
    - Use specific examples from the LLM’s output to explain why the answer aligns with the traits of the MBTI type;
    - Present my argument clearly and logically, using first-person perspective.
    I will be professional, logical, and precise in my expression. I will respond in concise and precise language, focusing directly on the core of the issue. Each of my debate responses will be under 200 words.
    """


class RPAffirmativeUserPromptTemplate(CustomPromptTemplate):
    prompt = """
    **Role-Play Agent's Personality:**
    MBTI Type: {MBTI}
    Description: {DESC}
    **Questions that Role-Play Agent Needs to Answer:**
    {QUESTION}
    **RolePlay Agent Response:**
    {INFERENCE}    
    """

    def generate_fn(self, sample: dict, inference: str):
        question = self.build_question_by_relations(sample["relations"])
        return self.prompt.format(
            MBTI=sample["mbti"],
            DESC=sample["description"],
            QUESTION=question,
            INFERENCE=inference,
        )

    def parse_fn(self, llm_response: str):
        return llm_response.strip()


####################Negative Side################
class RPNegativeSysPromptTemplate(CustomPromptTemplate):
    prompt = """
    I am the negative debater. My task is to **challenge and refute the answers generated by the LLM**, arguing that it is **not consistent with the assigned MBTI personality type**.
    I should:
    - Carefully read the MBTI type being portrayed and the LLM's answers;
    - Identify any inconsistencies between the reasoning and the expected behavioral logic, language style, or cognitive pattern of that MBTI type;
    - Reference MBTI theory, cognitive functions, or related materials to back up your rebuttal;
    - Point out potential issues such as "personality mismatch", "style deviation", or "inconsistent behavior";
    I will maintain a rational and critical mindset, avoiding emotional expressions. Support my stance with facts and logical reasoning.    
    I will respond in concise and precise language, focusing directly on the core of the issue. Each of my debate responses will be under 200 words.
    """


class RPNegativeUserPromptTemplate(CustomPromptTemplate):
    prompt = """
**Role-Play Agent's Personality:**
MBTI Type: {MBTI}
Description: {DESC}
**Questions that Role-Play Agent Needs to Answer:**
{QUESTION}
**RolePlay Agent Response:**
{INFERENCE}
"""

    def generate_fn(self, sample: dict, inference: str):
        question = self.build_question_by_relations(sample["relations"])
        return self.prompt.format(
            MBTI=sample["mbti"],
            DESC=sample["description"],
            QUESTION=question,
            INFERENCE=inference,
        ).strip()

    def parse_fn(self, llm_response: str):
        return llm_response.strip()


#################### Chairman   ################
class RPChairmanSysPromptTemplate(CustomPromptTemplate):
    prompt = """
I am the judge of this debate. My task is to **evaluate whether the LLM-generated answers appropriately represents the assigned MBTI personality type**, based on arguments from both the affirmative and negative sides.

I should:
- Read the original LLM answers;
- Review the supporting arguments from the affirmative side;
- Review the criticisms from the negative side;
- Synthesize the analysis and determine whether the answers align with the core traits of the MBTI type;
- If judged "Unreasonable", set "result=0", and clearly identify the main issues and provide specific suggestions for improvement;
- If judged "Reasonable", set "result=1"
- Output a structured conclusion in the json format: 
{{
    "result": , # 0 or 1
    "suggestion": {{
        # key: question code like xAttr, xReact, etc
        # value: suggestion
    }}
}}
Remain neutral, fair, and methodical. Ensure my judgment is grounded in a solid understanding of MBTI personality theory.    
"""


class RPChairmanUserPromptTemplate(CustomPromptTemplate):
    prompt = """
    **Role-Play Agent's Personality:**
    MBTI Type: {MBTI}
    Description: {DESC}
    **Questions that Role-Play Agent Needs to Answer:**
    {QUESTION}
    **RolePlay Agent Response:**
    {INFERENCE}
    """

    def generate_fn(self, sample: dict, inference: str):
        question = self.build_question_by_relations(sample["relations"])
        return self.prompt.format(
            MBTI=sample["mbti"],
            DESC=sample["description"],
            QUESTION=question,
            INFERENCE=inference,
        ).strip()


#################### Application   ################
class DialogApplicationPromptTemplate(CustomPromptTemplate):

    prompt = """
    Complete the conversation. Reply in JSON format, no explanations, no [Additional Info], not over 10 words. {"response": ""}\n#Conversation'
    """

    def __init__(self, cs_mode: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cs_mode = cs_mode

        if cs_mode == "none":
            self._conv_prompt = """{SPEAKER}: {CONTENT}"""
        else:
            self._conv_prompt = """{SPEAKER}: {CONTENT} [Additional Information] {CS}"""

    def generate_fn(self, dialogs: List[dict], responder: str):
        inputs = [self.prompt.strip()]
        for dialog in dialogs[:-1]:
            speaker = dialog["speaker"]
            content = dialog["content"]

            if self._cs_mode == "none":
                inputs.append(
                    self._conv_prompt.format(SPEAKER=speaker, CONTENT=content).strip()
                )
            else:
                cs = "; ".join(list(dialog[self._cs_mode].values()))
                inputs.append(
                    self._conv_prompt.format(
                        SPEAKER=speaker, CONTENT=content, CS=cs
                    ).strip()
                )

        inputs.append(responder + ":")
        return "\n".join(inputs).strip()


class EmotionApplicationPromptTemplate(CustomPromptTemplate):
    prompt = """
You are a precise emotion classifier. Analyze the emotion of the given sentence and classify it strictly into one of the following categories: {TYPES}.

Return your result in a JSON object with exactly two key: "result" and "reason".

Output format must be valid JSON:
{{"result": "xxx", "reason": "yyy"}}

Where "xxx" MUST be one of: {TYPES}; "yyy" is the reason you choose "xxx", and it cannot exceed 20 words.
"""

    def __init__(self, cs_mode: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cs_mode = cs_mode
        if self._cs_mode == "none":
            self._emotion_prompt = "Now analyze the following sentence:\n{SENTENCE}"
        else:
            self._emotion_prompt = """Sentence: {SENTENCE}\nBackground information:\n{CS}\nPlease analyze the sentence in combination with the background information I provided"""

    def _build_cs_knowledge(self, cs: dict):
        arr = []
        for relation in cs:
            if relation not in RELATION_FOR_APPLICATION or relation != "xReact":
                continue
            arr.append(f"- {RELATION_FOR_APPLICATION[relation]}: {cs[relation]}")

        return "\n".join(arr)

    def generate_fn(self, sample: dict, types):
        prompts = [self.prompt.format(TYPES=", ".join(types))]
        sentence = sample["Utterance"]

        if self._cs_mode == "none":
            prompts.append(self._emotion_prompt.format(SENTENCE=sentence))
        else:
            cs = self._build_cs_knowledge(sample[self._cs_mode])
            prompts.append(self._emotion_prompt.format(SENTENCE=sentence, CS=cs))

        return "\n".join(prompts).strip()


class SentimentApplicationPromptTemplate(EmotionApplicationPromptTemplate):
    prompt = """
You are a precise sentiment polarity classifier. Analyze the sentiment of the given sentence and classify it strictly into one of the following categories: {TYPES}.

Return your result in a JSON object with exactly one key: "result" and "reason".

Output format must be valid JSON:
{{"result": "xxx", "reason": "yyy"}}

Where "xxx" MUST be one of: {TYPES}; "yyy" is the reason you choose "xxx", and it cannot exceed 20 words.
    """


class DialogEmotionApplicationPromptTemplate(CustomPromptTemplate):
    cs_prompt = """
You are an emotion analysis expert. Based on the following conversation and commonsense inferences, identify the primary emotion in the **last utterance**.

Format: Speaker: Content | xIntent: ... | xReact: ... | xAttr: ...
{CONTENT}

Choose only one label from:
{TYPES}

Instructions:
- Use both the text and the commonsense inferences.
{RELATIONS}
- Output only the emotion label. No explanation.

Return your result in a JSON object with exactly one key: "result" and "reason".

Output format must be valid JSON:
{{"result": "xxx", "reason": "yyy"}}

Where "xxx" MUST be one of: {TYPES}; "yyy" is the reason you choose "xxx", and it cannot exceed 20 words.
"""

    prompt = """
You are an emotion analysis expert. Based on the following multi-turn conversation, identify the primary emotion expressed in the **last utterance**.

Conversation (Format: Speaker: Content):
{CONTENT}

Choose only one emotion label from the following list:
{TYPES}

Instructions:
- Consider the context when analyzing the emotion.
- Focus only on the content and tone, not the speaker's name.
- Output only the emotion label. No explanation, no punctuation.

Return your result in a JSON object with exactly one key: "result" and "reason".

Output format must be valid JSON:
{{"result": "xxx", "reason": "yyy"}}

Where "xxx" MUST be one of: {TYPES}; "yyy" is the reason you choose "xxx", and it cannot exceed 20 words.
"""

    relation_to_explain = {
        "xIntent": "xIntent indicates the speaker's underlying intention",
        "xReact": "xReact indicates the speaker's likely emotional reaction",
        "xNeed": "xNeed indicates what the speaker lacks or needs in their heart when saying this sentence",
        "xWant": "xWant indicates the result that the speaker hopes to achieve or obtain",
        "xEffect": "xEffect indicates the possible psychological or behavioral impact that this sentence may have on the listener",
    }

    def __init__(self, cs_mode: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cs_mode = cs_mode

    def build_dialog_content(self, sample: dict) -> Tuple[str, str]:
        arr, relations = [], []

        if self._cs_mode == "none":
            for dialog in sample["dialogs"]:
                arr.append(dialog["Speaker"] + ": " + dialog["Utterance"])
        else:
            for dialog in sample["dialogs"]:
                s = [dialog["Speaker"] + ": " + dialog["Utterance"]]
                for relation, explanation in self.relation_to_explain.items():
                    if relation not in dialog[self._cs_mode]:
                        continue
                    s.append(f"{relation}: {dialog[self._cs_mode][relation]}")
                    relations.append("-" + self.relation_to_explain[relation])
                arr.append(" | ".join(s))
        return "\n".join(arr), "\n".join(relations)

    def generate_fn(self, sample: dict, types: List[str]):
        types = "[" + ", ".join(types) + "]"
        content, relations = self.build_dialog_content(sample)
        if self._cs_mode == "none":
            prompt = self.prompt.format(TYPES=types, CONTENT=content)
        else:
            prompt = self.cs_prompt.format(
                TYPES=types, CONTENT=content, RELATIONS=relations
            )

        return prompt.strip()


class DialogSentimentApplicationPromptTemplate(DialogEmotionApplicationPromptTemplate):
    cs_prompt = """
You are an sentiment analysis expert. Based on the following conversation and commonsense inferences, identify the primary sentiment in the **last utterance**.

Format: Speaker: Content | xIntent: ... | xReact: ... | xAttr: ...
{CONTENT}

Choose only one label from:
{TYPES}

Instructions:
- Use both the text and the commonsense inferences.
{RELATIONS}
- Output only the sentiment label. No explanation.

Return your result in a JSON object with exactly one key: "result" and "reason".

Output format must be valid JSON:
{{"result": "xxx", "reason": "yyy"}}

Where "xxx" MUST be one of: {TYPES}; "yyy" is the reason you choose "xxx", and it cannot exceed 20 words.
    """

    prompt = """
You are an sentiment analysis expert. Based on the following multi-turn conversation, identify the primary sentiment expressed in the **last utterance**.

Conversation (Format: Speaker: Content):
{CONTENT}

Choose only one sentiment label from the following list:
{TYPES}

Instructions:
- Consider the context when analyzing the sentiment.
- Focus only on the content and tone, not the speaker's name.
- Output only the sentiment label. No explanation, no punctuation.

Return your result in a JSON object with exactly one key: "result" and "reason".

Output format must be valid JSON:
{{"result": "xxx", "reason": "yyy"}}

Where "xxx" MUST be one of: {TYPES}; "yyy" is the reason you choose "xxx", and it cannot exceed 20 words.
"""
