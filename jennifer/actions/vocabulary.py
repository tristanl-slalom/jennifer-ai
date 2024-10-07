from enum import Enum
from typing import Optional

from openai import OpenAI


class VocabularyWord(Enum):
    PROMPT = "prompt"
    ROLE = "role"
    CONTENT = "content"
    CONTEXT = "context"
    COMPLETION = "completion"
    CHAT_COMPLETION = "chat-completion"
    TOP_P = "top-p"
    TEMPERATURE = "temperature"


def vocabulary_action(word: VocabularyWord, age: int, temperature: Optional[float], top_p: Optional[float]):
    client = OpenAI()
    system_messages = [
        "I am trying to help the user understand generative AI terms",
        f"I know the user is {age} years old",
        "I want to speak in terms relevant to a person of that age",
        "I also want to use vocabulary familiar with a person of that age",
    ]
    user_messages_per_word = {
        VocabularyWord.PROMPT: "Help me understand what a 'prompt' is",
        VocabularyWord.ROLE: "Help me understand what the various 'roles' are in the open AI SDK messages",
        VocabularyWord.CONTENT: "Help me understand what a 'content' is, in terms of the open AI SDK messages",
        VocabularyWord.CONTEXT: "Help me understand what a 'context' is, in terms of the open AI SDK messages",
        VocabularyWord.COMPLETION: "Help me understand what a 'completion' is, not a 'chat completion'",
        VocabularyWord.CHAT_COMPLETION: "Help me understand what a 'chat completion' is, beyond a regular 'completion'",
        VocabularyWord.TOP_P: "Help me understand what 'top-p' is in comparison to 'temperature'",
        VocabularyWord.TEMPERATURE: "Help me understand what 'temperature' is in comparison to 'top-p'",
    }
    selected_user_message = user_messages_per_word[word]

    system_messages = [{
        "role": "system",
        "content": message
    } for message in system_messages]

    user_messages = [{
        "role": "user",
        "content": selected_user_message
    }]

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=system_messages + user_messages,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
