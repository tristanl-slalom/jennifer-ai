# Jennifer

A command line utility in Python against the latest versions of libraries and packages, as of 2024-10-01.

## Install and run

Clone the repo and use the following commands to get it working.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
jennifer --help
```

IMPORTANT: You must have your OPENAI_API_KEY set in your environment for the
OpenAI client to function!

Once installed, you'll find a series of commands, but "ask-question"
will do all the commands internally. Assuming it works on the first try,
it'll try to download the relevant data and process it for the domain
you select for your first question.

You may have to wait 5-10 minutes for that first request to gather all
the needed data, but further requests should go quickly.

# Main Commands

## Ask Question - Week 2

Scrapes a website and lets you ask questions against the site's 
contents. The first question takes a long time, but follow-up 
questions reuse the content it acquired. Ask a question of a new 
domain and the system will download data for that instead.

```bash
python main.py ask-question https://www.hot-dog.org \
"What can you tell me about the North American Meat Institute?"
```

Which should hopefully display something like:
```
The North American Meat Institute (NAMI) is the leading voice for
the meat and poultry industry. It was formed in 2015 from the
merger of the American Meat Institute (AMI) and North American
Meat Association (NAMA). NAMI provides essential member services
including legislative, regulatory, scientific, international, and
public affairs representation. The Institute's mission is to shape
a public policy environment that allows the meat and poultry industry
to produce wholesome products safely, efficiently, and profitably.
NAMI's members produce the majority of U.S. beef, pork, lamb, and
poultry, as well as the equipment, ingredients, and services
needed for high-quality products.
```

## Vocabulary - Week 3

The vocabulary command can use generative AI to explain various terms
of generative AI as if you were five years old.

The following terms can be described:

- prompt
- role
- content
- context
- completion
- chat-completion
- top-p
- temperature

```commandline
python main.py vocabulary temperature
```

```
Alright! Imagine you have a magical box that can make up stories!

**Temperature** is like a dial you can turn up or down. When you 
turn it up, the stories can be really wild and surprising, like a 
rainbow unicorn flying in space! But if you turn it down, the 
stories become more predictable, like a cat chasing a laser pointer.

**Top-p** is a little different. Think of it like giving the box a 
list of the best toys to choose from. If you pick a "top-p" of 0.8, 
the box can only choose from the top 80% of the toys. This way, it 
still makes fun stories but makes sure they’re not too strange and 
still pretty cool!

So, **temperature** makes the stories more crazy or safe, and **top-p** 
helps the box choose from the best ideas!%
```