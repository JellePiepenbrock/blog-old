---
layout: post
title: Predictability Minimization (Part II)
---

In the previous post I laid out the basic idea behind independent component analysis through predictability minimization. In this post, I will supply a basic PyTorch implementation of the idea. The main goal is to get a taste for how adversarial gradient descent works (or often, does not work). In a follow-up post, I will go into the convergence properties of these kinds of adversarial optimization systems.

For now though, image that you have 4 sequences containing uniformly random numbers, drawn from the interval (0, 1). This very small collection of data will stand in for a much larger dataset, that someone would like to analyse (say, for example, gene expression data with > 20.000 features per sample). In most cases, the amount of variables that are known places the datapoints in a higher-dimensional space that humans unfortunately have trouble with when trying to visualize it. Therefore, we want to summarize, for a suitable definition of 'summarize'. We want to find out what the big factors of variation in the data are, preferably finding a set of representations that are independent from each other.

There are widely established and well-understood methods such as principal component analysis (PCA). PCA however is a linear method and will not accurately represent non-linear patterns in the data. There are other-nonlinear, or locally linear methods such as UMAP, T-SNE, and so on. In independent component analysis with predictability minimization, we formulate the problem in an adversarial way. 

There is an encoder-decoder, which wants to encode the data and reconstruct it as best as possible. There is also a collection of predictors, that get the latent code of the encoder as their input, except for one code value, and these predictors have to predict the value of this missing unit.

The encoder is driven to put as much useful information in the code representation as possible (so that the encoder may use it to reconstruct the input). While doing this, the encoder also tries to maximize the loss of the collection of predictors: it encodes the information such that knowing the value of 1 code unit is of no use for predicting another. At the end of the process, what we are left with is an encoder that will perform independent component analysis on (data similar to) the training data.

Let us get to the code. First we define a Dataset that will give use the 4 sequences that we want to perform independent component analysis on. 
```python
class UniformDataset(Dataset):
    """Uniform dataset."""

    def __init__(self):
        self.uniformdata = np.random.uniform(size=(4, 4))
        print(self.uniformdata)

    def __len__(self):
        return len(self.uniformdata)

    def __getitem__(self, idx):
        return self.uniformdata[idx]
```

A typical set of 4 data points looks like this:
```python
>> [[0.08641456 0.19421044 0.88229338 0.0361495 ]
>>  [0.01163331 0.33199609 0.38802061 0.66070623]
>>  [0.5881097  0.97043498 0.98881562 0.11357908]
>>  [0.94657677 0.69272661 0.26470928 0.88500763]]
```

Now we define the Predictability Minimization class, which will host the neural networks that will be adversarially optimizing their losses. The point of this code is to be clear, but it can definitely be expanded to a case with a larger code and more predictor units. 

```python
class PM(nn.Module):

    def __init__(self):
        super(PM, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(4, 40),
            nn.ReLU(),
            nn.Linear(40, 2),
            nn.Sigmoid()
        )

        # decoder

        self.decoder = nn.Sequential(
            nn.Linear(2, 40),
            nn.ReLU(),
            nn.Linear(40, 4)
        )

        # predictor 1

        self.predictor1 = nn.Sequential(
            nn.Linear(1, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
            nn.Sigmoid()
        )

        # predictor 2

        self.predictor2 = nn.Sequential(
            nn.Linear(1, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        code = self.encoder(x)

        reconstruction = self.decoder(code)

        code1 = torch.tensor(code)[:, 1].reshape(4, 1)
        code0 = torch.tensor(code)[:, 0].reshape(4, 1)

        pred1 = self.predictor1(code1).to(device)
        pred2 = self.predictor2(code0).to(device)

        return code, pred1, pred2, reconstruction

```

On to the loss terms: we have a standarse mean square error loss for the autoencoder part:


$$ I = \frac{1}{n} \sum_{n=1}^N (Y_{code} - Y_{pred})^2 $$