# METALEARN

Metalearning experiments. 

Goals:

- A model that can learn indefinitely from self-produces loss / reward signal, given only previous output and environmental observation (a la RLAIF)
- A model that can optimize it's self produced learning signal based on empirical performance of how much that loss signal subsequently improves performance (a la metalearning)
- A model that optimizes with respect to loss signals produced in semantic space rather than purely via CE loss (a la JEPA)

Plans:

1. Show a model that can learn indefinitely during pretraining by self-producing the tokens it should do CE loss against.
  - One might reasonably call this trivial, since in this context its much much or efficient to just do CE with the real tokens that are already available.
  - However, getting this working reliably is a useful exercise in figuring out what levers need to be pulled to make this setup work in non-trivial settings
  - Observations:
    - non-frozen critic (ie critic is the optimizee) reward hacks in ~5 steps
    - frozen critic (same as optimizee just frozen) reward hacks in ~750 steps
  - Thoughts:
    - It's interesting that the model still reward hacks the frozen critic eventually, it'd be interesting to see what's actually happening when the hack occurs
    - Increasing frozen critic robustness
      - adversarial training during meta training phase
      - just training the critic for longer
      - inference time detection of reward hacking -> go back to meta training phase
    - Increasing unfrozen critic robustness
      - adversarial training during meta training phase
      - EMA style updates for pseudo-frozen critic
2. Show a model that can learn indefinitely during pretraining by self-producing a "gist-token" based on it's own autoregressive output and calculating loss relative to self-produced gist-token on ground truth. Gist tokens are learned via separate attention-masking-bottleneck phase
  - Motivation: The gist tokens inherently must capture some compressed "semantic" representation of the dataset. By calculating loss relative to these gist tokens, the loss must focus more on semantic level features rather than "surface texture" implied by token level CE
  - Thoughts:
    - One of the claimed advantages of JEPA is the ability to focus learning on features that are more easily predictable, rather than less predictable surface level features. Does the setup I advocate for above also achieve these advantages? If not, why not?
    - Does learning in semantic space make reward hacking easier or harder?
      - Case for harder: semantic space is more compressed than token space which provides a regularization effect
      - Case for easier: semantic space is disconencted from token space, giving weaker guarantees that gains in semantic space translate to gains in token space
3. Show a model that can learn indefinitely during pretraining by calculating loss of gist token of autoregressive output versus gist token of autoregressive output + ground truth. The two gist tokens, rather than lear via a separate compressive bottleneck phase, are instead learning in-situ in the meta learning phase via 2nd order backpropogation, optimizing for minimizing validation cross entropy.
  - Thoughts:
    - Perhaps possible to handle learning of very long term dependencies by, rather than choosing the validation chunk to do 2nd order backprop with respect to randomly / just choosing the next chunk in the sequence, instead doing cosine similarity on the chunks within the rest of the dataset that have high semantic similarity with the current training just, and using the high scoring validation chunks as the 2nd order target. This might help ensure we optimize for the content learned during online learning is actually the kind of content that's actually useful downstream.