cire-nlp

yeah it's another NLP engine. this one is one file, zero pip installs, and you can read it on a train.

I did not want a transformer or embeddings. I wanted inspectable string matching with tests. swap DOMAIN_SCHEMA and you have a constrained agent for your own topics.

resolution order:
- normalize
- meta intents
- topic score
- intent class
- fallback

word boundaries matter. `topic` is not `top`. ranking does not fire because a keyword contains three letters in a row.

I already check these:
- resolve("rules for topic b") returns domain_b / rules
- detect_intent("topic a") is info, not ranking

```bash
python -m unittest test_cire.py -v
python cire_engine.py "rules for topic b"
python cire_engine.py --smoke
```

no GPU because there is no model. MIT.
