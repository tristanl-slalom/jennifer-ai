from openai import OpenAI


def haiku_action(topic="recursion in programming"):
    """
    Generate a haiku with the provided topic.
    """
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Write a haiku about {topic}."},
        ],
    )

    print(completion.choices[0].message.content)
