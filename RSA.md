---
layout: default
title: RSA
---

The Rational Speech Acts family of models describes language understanding as a listener reasoning about a speaker reasoning about a speaker.
It is a classic example of nested Bayesian inference: each agent model performs marginal inference, with the outer models calling the inner.
Because the inner models will be evaluated many times with different arguments, this is also a perfect *amortized inference* setting:
Many related queries must be evaluated, and we care about the overall speed and quality, not that of individual sub-queries.



These notes are about scaling 

# Three basic problems

Three basic problems:
big worlds
big languages
where do the lexicon & prior come from?


RSA with a large world.

RSA with a large, compositional language.

TFBT for RSA. (Connected to learning.)

## Bigger worlds
 More objects and relations (symmetries?)
 Continuous variables (cannot do enumeration)

## Larger language
  Compositional semantics
  More free vars
  Alternatives 

##The full Bayesian thing
BDA on top of RSA?

# Discourse

Longer horizon informativity and persistence of QUD.
  
# Realtime and incremental processing

Reactive? Learned heuristics?
  
# Learning
  
If we want to scale RSA to real-world applications, we will need to cover a much wider portion of language and world-knowledge.
It is likely that this can't be done without learning at least some aspects from data.
Perhaps the biggest challenge to scaling RSA is thus the *learning problem*: learning (aspects of) the lexicon, world knowledge, and perhaps linguistic system from (weakly supervised) data.

Learning: 
  lexicon, 
  qud dist,
  background knowledge

Parameter learning. Basically the same as BDA over RSA, though with a lot of params.

Structure learning. Hard. From structured language / interviews (ala whybot)?


# Some inference ideas

Observation: nested models are a perfect amortized setting
-caching (current method)
-faster/better inferences by generalizing from related (sub)queries?

Inference ideas
  caching / GP surrogate fn
  learned heuristic factors for better smc
  latent variable predictors for better smc
  pseudo-marginal MH? needs estimate of likelihood... smc? 
  lifted inference / symmetry-based techniques
  variational?


  