# Jennifer

A command line utility in Python against the latest versions of libraries and packages, as of 2024-10-01.

## Install and run

Clone the repo and use the following commands to get it working.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --help
```

IMPORTANT: You must have your OPENAI_API_KEY set in your environment for the
OpenAI client to function!

Once installed, you'll find a series of commands, but "ask-question"
will do all the commands internally. Assuming it works on the first try,
it'll try to download the relevant data and process it for the domain
you select for your first question.

You may have to wait 5-10 minutes for that first request to gather all
the needed data, but further requests should go quickly.

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
