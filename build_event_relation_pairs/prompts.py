from easy_tools.llms.prompt import PromptTemplate
from tools import parse_to_json

class RateEventRelationPrompt(PromptTemplate):
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

    def parse_fn(self, llm_response):
        return parse_to_json(llm_response)
