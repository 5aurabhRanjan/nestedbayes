___
title: RSA
___

The Rational Speech Acts family of models describes language understanding as a listener reasoning about a speaker reasoning about a speaker.
It is a classic example of nested Bayesian inference: each agent model performs marginal inference, with the outer models calling the inner.
Because the inner models will be evaluated many times with different arguments, this is also a perfect *amortized inference* setting:
Many related queries must be evaluated, and we care about the overall speed and quality, not that of individual sub-queries.

test