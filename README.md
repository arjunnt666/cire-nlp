# cire-nlp

another NLP engine, except this one is one file, zero pip installs, and you can read it on a train.

layered resolution: normalize, meta intents, topic score, intent class, fallback. swap DOMAIN_SCHEMA and you have a constrained agent for your own topics.

not a transformer. not embeddings. inspectable string matching with tests.

word boundaries matter. `topic` is not `top`. ranking does not fire because a keyword contains three letters in a row.

## works today

- `python -m unittest test_cire.py`
- `python cire_engine.py "rules for topic b"`
- resolve("rules for topic b") returns domain_b / rules
- detect_intent("topic a") is info, not ranking

## does not work yet

- multilingual generative models
- anything that needs a GPU

## try it

```bash
python -m unittest test_cire.py -v
python cire_engine.py "rules for topic b"
python cire_engine.py --smoke
```

## license

mit.
