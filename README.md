nestedbayes
===========

Thinking through "nested-query" WebPPL models -- those where conditioning is nested inside conditioning. These are called doubly intractable in stats.

Examples include Bayesian social cognition models, Bayesian data analysis of Bayesian cognitive models, and posterior inference of model parameters for undirected models.

At the moment only exploring GP-based caching. It is worth thinking more generally. E.g. see [MCMC for doubly-intractable distributions](http://arxiv.org/pdf/1206.6848.pdf), and [this one](http://xxx.tau.ac.il/pdf/1306.4032.pdf).

Requires sylvester math library (used by gaussian process code). Do `npm install sylvester`.

Gaussian process code taken from  [gaussianprocess.js](https://github.com/scotthellman/gaussianprocess_js).
