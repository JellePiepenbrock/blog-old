---
layout: post
title: Predictability Minimization
---

This will be a small blog post explaining Juergen Schmidhubers 1992 paper on 'Predictability Minimization'. The full title of the paper is 'Learning Factorial Codes by Predictability Minimization'. Recently, there has been a lot of literature focusing on 'disentangled representations' in variational autoencoders and other neural network architectures.

Neural networks find internal representations of their training data during the gradient descent optimization procedure. The problem here is often that these representations are not intelligible by humans. One major component of this is that various dimensions of the representations can team up together, in pairs, groups, or all of them together, to capture a specific aspect of the training data. This makes it very hard to understand what a particular value of one of the representation dimensions represents. 

A partial solution to this is enforcing statistical independence between the variables in the representations. In the paper, this means that the value of $$ x_1 $$ ... 

zee
$$  x^{(n+1)} = x^{(n)} - \frac{cos(x)}{-sin(x)} $$
test
```python
number = 0
print(number)
asdf
```