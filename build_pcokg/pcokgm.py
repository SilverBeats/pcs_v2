#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

ACCEPTABLE_RELATIONS = [
    "xReact",
    "xWant",
    "xIntent",
    "xNeed",
    "xEffect",
    "xAttr",
    "oWant",
    "oEffect",
    "oReact",
]

RELATION_TO_QUESTION = {
    "oEffect": "What do you think is the impact of this on other people?",
    "oReact": "How do you think other people feel after this event?",
    "oWant": "What do you think others are going to do after this event?",
    "xEffect": "What impact did it have on you?",
    "xNeed": "What did you prepare before doing it?",
    "xReact": "How did you feel after doing it?",
    "xWant": "What do you plan to do next?",
    "xIntent": "Why did you do it?",
    "xAttr": "How would you describe yourself?",
}

MBTI_TO_DESCRIPTION = {
    "ISTJ": "I am organized and detail-oriented. I value responsibility, rules, and traditions. I like to plan ahead and make sure things are done efficiently and correctly. I rely on past experiences to guide my decisions.",
    "ISFJ": "I care deeply about others and enjoy helping in practical ways. I'm loyal, patient, and sensitive to people's needs. I prefer harmony and stability over conflict, and I often put others' needs before my own.",
    "INFJ": "I am introspective and guided by deep values and visions. I understand people intuitively and want to help them grow. I believe in meaning and purpose, and I strive to live authentically according to my ideals.",
    "INTJ": "I am strategic and independent, always thinking several steps ahead. I trust logic and intuition to uncover patterns and long-term possibilities. I’m focused on self-improvement and achieving my goals efficiently.",
    "ISTP": "I enjoy solving problems with my hands and understanding how things work physically. I’m observant, practical, and action-oriented. I don’t seek attention, but I stay calm and capable under pressure.",
    "ISFP": "I am quiet and sensitive to beauty and emotion. I express myself creatively through art, music, or movement. I value personal freedom and dislike being controlled or rushed.",
    "INFP": "I am idealistic and deeply reflective. I search for authenticity and meaning in life. I’m compassionate and often inspired by poetry, philosophy, or nature. I want to live in alignment with my core values.",
    "INTP": "I love exploring ideas and understanding complex systems. I enjoy analyzing theories and questioning assumptions. I prefer abstract thinking over routine tasks, and I spend a lot of time in my own thoughts.",
    "ESTJ": "I am decisive and organized, good at managing teams and resources. I believe in structure, fairness, and accountability. I take responsibility seriously and expect the same from others.",
    "ESFJ": "I am warm and sociable, always looking out for the people around me. I enjoy connecting with others, resolving conflicts, and creating a sense of belonging. I thrive in supportive, harmonious environments.",
    "ENFJ": "I inspire and empower others with my words and presence. I naturally see potential in people and guide them toward growth. I communicate clearly and passionately, driven by a sense of purpose and connection.",
    "ENTJ": "I am a natural leader who enjoys setting bold goals and making strategic plans. I'm confident, assertive, and results-oriented. I challenge the status quo and lead teams to achieve excellence and success.",
    "ESTP": "I act quickly and adapt easily to new situations. I love excitement, competition, and hands-on challenges. I live in the moment and enjoy motivating others through energy and spontaneity.",
    "ESFP": "I am cheerful and enjoy being around people. I bring positivity and fun wherever I go. I appreciate sensory pleasures and live in the present, sharing joy and memorable experiences with those around me.",
    "ENFP": "I am curious, enthusiastic, and full of creative energy. I love brainstorming new ideas and bringing people together. I form deep connections and enjoy exploring life’s endless possibilities.",
    "ENTP": "I am sharp-minded, witty, and love debating ideas. I thrive on intellectual challenges and enjoy coming up with unconventional solutions. I'm always learning, growing, and pushing boundaries.",
}


class PCoKGM:
    SYS_PROMPT = """
    I am the PersonX. I am the initiator of the event. My MBTI type is {MBTI}. {MBTI_DESC}
    I will follow my personality strictly, answer any questions you may have, and reply differently from other MBTI types. I may give an extreme answer that is in line with my personality. I will avoid outputting the subject and use infinitive phrases more often, and answer your question in the most concise way.
    """
    USER_PROMPT = """
    {RELATION_QUESTION}
    Given Events: {EVENT}
    """

    def __init__(self, model_path: str, device: str = None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    @torch.no_grad()
    def generate(
        self,
        sentence: str,
        relation: str,
        mbti: str,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 20,
        **kwargs
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": self.SYS_PROMPT.format(
                    MBTI=mbti, MBTI_DESC=MBTI_TO_DESCRIPTION[mbti]
                ).strip(),
            }
        ]
        query = self.USER_PROMPT.format(
            RELATION_QUESTION=RELATION_TO_QUESTION[relation], EVENT=sentence
        ).strip()

        response, _ = self.model.chat(
            tokenizer=self.tokenizer,
            query=query,
            history=messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
        return response

    @staticmethod
    def build_messages(event: str, relation: str, mbti: str) -> List[dict]:
        messages = [
            {
                "role": "system",
                "content": PCoKGM.SYS_PROMPT.format(
                    MBTI=mbti, MBTI_DESC=MBTI_TO_DESCRIPTION[mbti]
                ).strip(),
            },
            {
                "role": "user",
                "content": PCoKGM.USER_PROMPT.format(
                    RELATION_QUESTION=RELATION_TO_QUESTION[relation], EVENT=event
                ).strip(),
            },
        ]
        return messages
