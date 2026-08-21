# cire-nlp

another NLP engine, except this one is one file, zero pip installs, and you can read it on a train.

layered resolution: normalize, meta intents, topic score, intent class, fallback. swap DOMAIN_SCHEMA and you have a constrained agent for your own topics.

not a transformer. not embeddings. inspectable string matching with tests.

## works today

- `python -m unittest test_cire.py`
- `python cire_engine.py` smoke
- resolve("rules for topic b") returns domain_b / rules

## does not work yet

- multilingual generative models
- anything that needs a GPU

## try it

```bash
python -m unittest test_cire.py -v
python cire_engine.py
```

## license

mit.
