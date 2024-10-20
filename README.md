# Jennifer

A command line utility in Python against the latest versions of libraries and packages, as of 2024-10-01.

## Upgrade to python 3.10.0 or later
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew update

brew install python@3.12

echo 'export PATH="/usr/local/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
echo 'alias python="python3.12"' >> ~/.zshrc
```

## Install and run

Clone the repo and use the following commands to get it working.

```bash
source ~/.zshrc
python -m venv venv
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

## Embeddings - Week 5

Gather the data and calculate embeddings with the latest models. Use 
arguments to determine whether we want to show various matplot 
graphs. Use an optional product description to show related reviews.
Specify a number of clusters to group the reviews into a number of high level 
commonalities. You can specify both, but this may not work with every 
search term, number of clusters, and the amount of data we have. Expect errors!

```bash
# Search for products matching a description
jennifer embeddings search "beverage" --num-results 3
```
```
⠧ Loading existing embeddings... 0:00:02
A sweet, refreshing drink!:  I've never been much of a soda drinker, 
so I'm always on the lookout for alternative beverages like juice and
tea.  Lately, I've been swept up in the craze of coconut wate...

Refreshing and healthy drink:  I was delighted to receive this new 
V8 drink as I've been a fan of V8 for a long time as it not only is
healthy, but it tastes really good.  I don't know if it gives mor...

Yum!:  I loved the V8 Pomegranate Blueberry energy drink! It was very
good and has some of the fruits and veggies that we try to pack in all
day. This is also an easy way to get the nutrition that mos...
```

```bash
# show clusters of reviews and cluster visualization (close the graph to continue!)
jennifer embeddings clusters --num-clusters 3 --show-clustering
```

```bash
# show clusters of reviews and various visualizations (close the graphs to continue!)
jennifer embeddings visualize
```
