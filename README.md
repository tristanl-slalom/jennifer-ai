# Jennifer

A command line utility in Python against the latest versions of libraries and packages, as of 2024-10-01.

## Upgrade to python 3.10.0 or later
```bash
# Install pyenv if not already installed
curl https://pyenv.run | bash

# Add pyenv to your shell
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Restart your shell
exec "$SHELL"

# Install Python 3.10 using pyenv
pyenv install 3.10.0

# Set Python 3.10 as the global version
pyenv global 3.10.0
```

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

## Haiku - Week 1

Uses the OpenAI Completion API to generate a haiku, defaulting to the
tutorial topic of recursive programming, but you can set your own topic
using the optional "--topic" argument.

```bash
jennifer haiku --topic "drinking coffee"
```

```bash
Warm cup in my hands,  
Steam rises like morning light—
Awake, dreams take flight.
```

## Ask Question - Week 2

Scrapes a website and lets you ask questions against the site's 
contents. The first question takes a long time, but follow-up 
questions reuse the content it acquired. Ask a question of a new 
domain and the system will download data for that instead.

```bash
jennifer ask-question https://www.hot-dog.org \
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
jennifer vocabulary temperature
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

## Baseball vs Hockey - Week 4

The goal for this week was to use sample data to train a model with
specific data, and then leverage that model with the trained data.
The problem is, the Jupyter notebook relied on old APIs and base
models that were retiring in just a few days (e.g. 'babbage').

> New fine-tuning training runs on babbage-002 and davinci-002 will 
> no longer be supported starting October 28, 2024.
> -- [Open AI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning/which-models-can-be-fine-tuned)

I've taken from the original materials a pretty faithful translation
but the project should try to create a new model from the data, wait
for its successful completion, do some analysis on the results, and
then attempt to use the model.

PLEASE NOTE: Creating a model can take a long time, like 30 minutes.
A happy progress spinner will let you know it's still going, but it
cannot gauge progress.

```bash
# should create a model and test it with some sample data.
jennifer training
```

```bash
# should use the model from an existing successful job instead.
jennifer training --existing-job ftjob-se4KpuhQ7rXOf5YcCUVOVNzp
```

```bash
# should use the model from an existing successful job and
# use the given test message instead.
jennifer training --existing-job ftjob-se4KpuhQ7rXOf5YcCUVOVNzp\
 --test-message "I love how the players smack pucks with their big sticks"

```
```
Provided test message determined to be hockey related
```

