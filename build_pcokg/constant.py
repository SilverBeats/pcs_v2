#!/usr/bin/env python3
# -*- coding: utf-8 -*-

ATOMIC_RELATIONS = [
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

RELATION_FOR_APPLICATION = {
    "xReact": "Current mood",
    "xEffect": "Influence",
    "xNeed": "Preparatory work",
    "xIntent": "Intention",
    "xWant": "Future plans",
}
