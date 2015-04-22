---
layout: default
title: RSA
---

The Rational Speech Acts family of models describes language understanding as a listener reasoning about a speaker reasoning about a speaker.
It is a classic example of nested Bayesian inference: each agent model performs marginal inference, with the outer models calling the inner.
Because the inner models will be evaluated many times with different arguments, this is also a perfect *amortized inference* setting:
Many related queries must be evaluated, and we care about the overall speed and quality, not that of individual sub-queries.

~~~
var worldPrior = function() {
  var num_nice_people = randomInteger(4) //3 people.. 0-3 can be nice.
  return num_nice_people
}

var utterancePrior = function() {
  var utterances = ["some of the people are nice",
                    "all of the people are nice",
                    "none of the people are nice"]
  var i = randomInteger(utterances.length)
  return utterances[i]
}

var meaning = function(utt,world) {
  return utt=="some of the people are nice"? world>0 :
         utt=="all of the people are nice"? world==3 :
         utt=="none of the people are nice"? world==0 :
         true
}

var literalListener = function(utterance) {
  Enumerate(function(){
    var world = worldPrior()
    var m = meaning(utterance, world)
    factor(m?0:-Infinity)
    return world
  })
}

var speaker = function(world) {
  Enumerate(function(){
    var utterance = utterancePrior()
    var L = literalListener(utterance)
    factor(L.score([],world))
    return utterance
  })
}

var listener = function(utterance) {
  Enumerate(function(){
    var world = worldPrior()
    var S = speaker(world)
    factor(S.score([],utterance))
    return world
  })
}

print(listener("some of the people are nice"))
~~~

These notes are about scaling up the RSA model to handle more complex fragments of language. Ultimately this should support both open-ended natural language in an experimental context, and NLP-style applications.

# Three basic problems
We can identify three basic problem types: 
RSA with a large world, where the support of `worldPrior` is big. 
RSA with a large and compositional language, where the `utterancePrior` generates a big language and `meaning` involves semantic parsing.
Posterior (or MAP) parameter inference for RSA, which figures in both Bayesian data analysis and model learning.

## Bigger worlds
When the support of the `worldPrior` (or an implicit generative model version) gets large, exact inference becomes intractable for both `literalListener` and `listener`. Large worlds can come quickly from having many objects and relations, from having multiple free-variables, etc. The impact of large worlds is more acute for RSA than typical inference problems because each additional world implies an additional evaluation of the `speaker` (which implies many evaluations of the `literalListener`, which depends on world size).

**Alg:** One partial fix is to leverage symmetry in the model to do *lifted inference*. For instance, if there are multiple objects but their identities don't matter, then there is a permutation-group symmetry. By reducing the model space to equivalence classes before inference a lot of work can be saved. However, it can be had to find and harder to exploit symmetries. How many models we care about have exploitable symmetries?

Another way that we end up with big effective world sizes is when there are *continuous variables* (which we currently often discretize). In general, continuous variables suggest moving away from enumeration.


## Larger language
The next type of challenge comes from a larger language, particularly a compositional language. This enters into RSA in two places: the `utterancePrior` gets large and makes the speaker expensive, computation of meaning requires semantic parsing. These two obstacles can be put together by moving the computation of meaning and alternatives into the `listener`, which has the nice side effect of making free-variable creation and use happen in the same place. It's worth noting that free variables effectively increase the size of the world *and* the language.

A naive way to extend the `utterancePrior` is to generate possible utterances for the speaker from a PCFG. This makes the model tricky because most sentences are not informative about most worlds, but the prior doesn't take this into account. Heuristic factors (see below) may be one solution. Another solution may be to take a limited alternative set more seriously, deriving a smaller set of alternatives to use in `utterancePrior` from the utterance that `listener` hears. The question then becomes: how can alternatives be coherently derived? (And if the alternative generator is stochastic, should the `listener` or `speaker` reason about alternatives? This is interesting but runs the risk of re-introducing the big language into the speaker model....)

When we turn to the `meaning` function we have no choices but to include full semantic parsing to turn strings into potential denotations. Standard approaches like CCG work by looking up words in a lexicon and then using combinators to combine meanings of constituents together. On the bright side, semantic parsing people have had success with simple methods like beam search (which is closely related to particle filtering). On the dark side, there is a large search space of potential parses for a string, and this has to be done for at least the observed sentence and all alternatives.

##The full Bayesian thing
Next consider the special pains of trying to do Bayesian data analysis of RSA, even when the RSA model itself is fairly simple. For instance, inferring alternative weights from response data might look like:

~~~
var analysis = function(data) {
  MH(function(){
    var alts = repeat(numAlts, dirichletERP)
    map(function(d){factor(listener(d.utt).score([],d.resp))}, data)
    return alts
  })
~~~

This looks simple, but it implies yet another nesting of inference, and introduces a large continuous state space. It gets worse if we try to have item-wise or subject-wise effects (random values per item, such as different alternative weights, or per subject, such as dependent-measure calibration.)

The same problems crop up if we are trying to learn parameters of the RSA model from training data. Here the objective may be MAP parameters, not posterior inference, but the number of parameters will be large. This many-params approach is taken by many useful semantic parser learning algorithms; they tend to do stochastic gradient descent on the params.


# Other directions and their problems

## Discourse
Longer-horizon notions of speaker utility (such as expected future informativity) and persistence of variables (such as QUD) across multiple utterances can turn the simple utterance selection model into a complex planning problem. In general, dealing with sequences of utterances makes the model very complex, unless we make simplifying assumptions.

Note that when we assume the utterances are independent conditioned on the language, this again gives us a learning model (as taken by Smith, et al, and by Hamilton, et al).
  
## Realtime and incremental processing
People are really fast at language understanding and production. They also seem to work incrementally, processing the sentence as it is coming in. How can we achieve this speed and model this incremental processing?

Both reactive idioms and learned heuristic factors (etc) may have a role to play.
  
## Uncertain agents
For uncertain speakers the appropriate utility is not log-probability of the known state (because there isn't one); instead we have used KL-distance from the belief distribution of the speaker to the posterior distribution of the `literalListener`. This complicates matters somewhat for approximate inference techniques because KL can be hard to estimate (especially from samples).

  
# Learning
If we want to scale RSA to real-world applications, we will need to cover a much wider portion of language and world-knowledge.
It is likely that this can't be done without learning at least some aspects from data.
Perhaps the biggest challenge to scaling RSA is thus the *learning problem*: learning (aspects of) the lexicon, world knowledge, and perhaps linguistic system from (weakly supervised) data.


## Parameter learning 
Basically the same as BDA over RSA, though with a lot of params. Gradient optimization methods are typically used. We could just do stochastic gradient descent. Gradient-based monte carlo (e.g. HMC) also seems promising. Both of these require gradients of the `listener`, which is tricky to compute. One approach is to use a surrogate function with easy gradients (see below).

## Structure learning 
Hard but important. Maybe from structured language / interviews (ala whybot)?


# Some inference ideas

Scaling up inference to address the challenges outlined above will require exploring a number of different approaches. We will likely have to abandon exhaustive enumeration as state spaces get big. In it's place SMC seems like a fairly general approach that incorporates the best of MH (via rejuvenation), importance sampling (e.g. getting estimates of normalizing constant), and search.

A key observation is that nested models are a perfect amortized setting, where we need to do inference quickly on average across many instances of the sub-models.
Our current main method for inference in RSA is naive enumeration with caching of the sub-models. This is an exponential improvement over not caching, but doesn't help much with large or continuous state spaces.
Can we get faster/better inferences by generalizing from related (sub)queries that we've already computed?

## Surrogate caching

Instead of simply caching the computed value of a sub-query (e.g. `speaker`), we could use these values to estimate a surrogate likelihood function; the surrogate would give us predictions for nearby argument values, that we could use instead of computing the actual function return distribution. An appropriate and fairly tractable family of surrogates is Gaussian processes (GP). (Note: GP take quadratic time  in the number of data points. This could be problematic. There are a variety of techniques that help.)

If the inference algorithm used to estimate the likelihood of the sub-model is not exact, for instance if it is based on SMC samples, then the surrogate-estimator needs to take into account the noise---that is, we should not be fully confident in the value returned by any one call to the cached function. For an unbiased estimator with approximately normal error distribution, this will be handled by a standard squared-error GP kernel. However since we probably won't know the variance of the noise *a priori*, we'll have to estimate it.

An interesting and pleasant side-effect of using a surrogate likelihood function is that the surrogate gradient is likely to be easy to compute. This suggests doing gradient-based methods (HMC or SGD) using GP-caching to estimate the sub-models.

## Predictive inference

The fancy caching methods generalizes to nearby arguments, but only do so at the level of an entire sub-model. Can we get finer-grained re-use within a model? Based (loosely) on stochastic inverses, we have begun to explore several techniques for 'predictive' inference---algorithms that use previous similar runs to predict aspects of a completed execution from a partial execution. In general we are thinking of these in an SMC setting, where they help guide the early choices before factors happen, though these predictions can be useful more generally (e.g. for block MH proposals).

###Heuristic factors
Heuristic factors are canceling pairs of factors inserted into a program (not changing the distribution, but potentially influencing the algorithm). The ideal heuristic factor would (I think) be the expected score of completing the execution from the point here it occurs (since this up-weights executions that are likely to end up good, even if they aren't yet). One use of 'nearby' queries is to estimate this expected completion score. The algorithm is something like: find similar execution prefixes of the sub-model from similar arguments to the sub-model, use the final scores of these to estimate the expected final score of this execution. There are a number of details to explore in making this estimate good. It is also possible that something like a GP or neural net should be used to estimate the 'heuristic score' function.

###Sample predictors
An alternative is to predict the latent variables directly. That is, if you know the values that a variable tends to take on in the normalized (posterior) distributions, conditioned on the current prefix, you may as well set it to this value. We would train these sample predictors from nearby executions as for heuristic factors. In SMC the trained predictors could be used as the importance distribution from which to sample new variables.

## Other stuff
   
###Coarse-to-fine
How could coarse-to-fine inference help?

###Pseudo-marginal MH
This gives unbiased estimates for doubly-intractable models by using sampled estimates of the likelihood in MH. The likelihood estimates could come from SMC. While the unbiasedness is nice, it's not clear that this is the best use of computational resources.
  
### SOSMC
Do different orderings of the world, or parsing, models sometimes result in better samplers? If so, then SOSMC may help.
  
###Variational inference

###SMT solvers 
  particularly in literalListener


  