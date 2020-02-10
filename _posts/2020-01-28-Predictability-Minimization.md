---
layout: post
title: Predictability Minimization (Part I)
---

This will be a small blog post explaining the setup of Juergen Schmidhubers 1992 paper on 'Predictability Minimization'. The full title of the paper is 'Learning Factorial Codes by Predictability Minimization'. Recently, there has been a lot of literature focusing on 'disentangled representations' in variational autoencoders and other neural network architectures. In a follow-up post, I will give a simple Python implementation of the paper's central idea.

In the author's own words, in the abstract, the idea goes like this:

>I propose a novel general principle for unsupervised learning of distributed nonredundant internal representations of input patterns. The principle is based on two opposing forces. For each representational unit there is an adaptive predictor, which tries to predict the unit from the remaining units. In turn, each unit tries to react to the environment such that it minimizes its predictability. This encourages each unit to filter "abstract concepts" out of the environmental input such that these concepts are statistically independent of those on which the other units focus. 

Neural networks find internal representations of their training data during the gradient descent optimization procedure. The problem here is often that these representations are not intelligible by humans. One major component of this is that various dimensions of the representations can team up together, in pairs, groups, or all of them together, to capture a specific aspect of the training data. This makes it very hard to understand what a particular value of one of the representation dimensions represents. 

The neural network F is a function $$F:  X \to Z$$ where Z is lower-dimensional. Specific dimensions k of Z will be labeled $$ z_k $$.

A partial solution to this is enforcing statistical independence between the variables in the representations. In the paper, this means that the value of $$ z_1 $$ does not have any predictive value for the value of $$ z_2 $$. 

Schmidhuber suggests that we can enforce this statistical independence by using three neural network types together in an adversarial game.

The first, the encoding neural network, takes the input and produces some code $$ z_1 ...z_k $$ that represents the training sample. 

The second, the decoder neural network, takes this code and tries to reproduce the input sample. 

The third type of network that we need is a 'predictor' network, that takes as input all but 1 of the $$ z $$ code variables and tries to predict the missing one. If it is able to succeed in this, the latent code does not consist of statistically independent variables! Note that in this naive setup, we need K different predictor networks, one for each $$ z $$ variable.
 
 The example case that the paper uses the show the method's workings is the following:
 
 We have a set of 4 random sequences of real numbers. The goal is for the code units to contain a factorial code for these sequences, where the two code units activations are independent of each other.

Can we automatically learn a factorial code? The paper claims that the setup will encode each real number sequence into one of these four options: [0, 0], [0, 1], [1, 0], [1, 1]. We'll find out.
 
