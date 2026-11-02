# 📘 Day 3 Revision Sheet — Statistics & Naive Bayes (Corrected)

## PART 1 — Bayes' Theorem

### 1.1 The formula

\[ \boxed{P(A\mid B)=\frac{P(B\mid A)\cdot P(A)}{P(B)}} \]

Read it: "Probability of A given B equals the probability of B given A, multiplied by the probability of A, divided by the probability of B."

### 1.2 The four components

|Name|Formula|Meaning|
|---|---|---|
|Prior|\(P(A)\)|Belief about A before seeing evidence|
|Likelihood|\(P(B\mid A)\)|Probability of observing B if A is true|
|Evidence|\(P(B)\)|Overall probability of observing B|
|Posterior|\(P(A\mid B)\)|Updated probability of A after observing B|

Bayes is an update rule: start with a prior, observe evidence, and calculate a posterior.

### 1.3 Medical test example

- Disease prevalence: 1 in 10,000 → \(P(D)=0.0001\)
    
- Sensitivity: \(P(+\mid D)=0.99\)
    
- False-positive rate: \(P(+\mid \neg D)=0.01\)
    

We want:

\[ P(D\mid +)= \frac{P(+\mid D)P(D)}{P(+)} \]

First calculate the probability of a positive test:

\[ P(+)=P(+\mid D)P(D)+P(+\mid\neg D)P(\neg D) \]

\[ =(0.99)(0.0001)+(0.01)(0.9999) =0.010098 \]

Therefore:

\[ P(D\mid +)=\frac{0.000099}{0.010098} \approx 0.0098 \]

Approximately 0.98%

Why? The disease is extremely rare. In a population of 10,000 people, we'd expect approximately:

- 1 person with the disease, producing 0.99 expected true positives.
    
- 9,999 healthy people, producing about 100 expected false positives.
    

Lesson: A positive test does not necessarily mean a high probability of disease. The prior/base rate matters.

### 1.4 Classification version

\[ P(y\mid X)=\frac{P(X\mid y)P(y)}{P(X)} \]

For classification:

\[ \boxed{\hat y=\arg\max_c P(X\mid c)P(c)} \]

We can ignore \(P(X)\) because it is identical for every candidate class for the same input. It does not affect which class wins.

## PART 2 — The Naive Bayes Assumption

### 2.1 The problem it solves

We want to estimate:

\[ P(x_1,x_2,\ldots,x_d\mid c) \]

This is the joint probability of all features given a class.

For \(d=50\) binary features, there are:

\[ 2^{50}\approx 1.13\times10^{15} \]

possible feature combinations per class.

Estimating the full joint distribution would require an impractically large amount of data.

### 2.2 The naive assumption

\[ \boxed{ P(x_1,x_2,\ldots,x_d\mid c) = \prod_{i=1}^{d}P(x_i\mid c) } \]

Conditional independence: assume that once the class is known, the features are independent of one another.

Parameter comparison for binary features:

- Full joint distribution: \(2^d\) probability entries per class, or \(2^d-1\) independent parameters.
    
- Naive Bayes: \(d\) probability parameters per class.
    

For 50 binary features, that's 50 independent parameters instead of \(2^{50}-1\).

Exponential complexity becomes linear.

### 2.3 Why it's "naive" — and why it works anyway

Naive: real-world features are often correlated, even when the class is known. For example, `src_bytes` and `duration` in network traffic data.

Why it works: even when the estimated probabilities are imperfect, the class ranking can still be correct. Naive Bayes is also a low-variance model because it estimates relatively few parameters, which can help prevent overfitting.

Important nuance: errors do not always affect every class equally or cancel out. Naive Bayes can still make incorrect predictions.

## PART 3 — Gaussian Naive Bayes

### 3.1 When to use

Continuous numerical features, such as:

- Bytes transferred.
    
- Connection duration.
    
- Temperature.
    
- Pressure.
    

Assumption: each feature follows a Gaussian (normal) distribution within each class.

[Bell Curve Generator — Normal Distribution | SciFig](https://images.openai.com/static-rsc-4/ktoez2hj9DIv8jcdqSqFLTyAdoFVPHIlCSTNZwjYkDmrnqGs6MMaRYfTmbAMoBCOoq_dDJzaPSDIqAA0Hh33y9BwL3PykCR9JyXCcJfjZXNNuHRuqBabxKSNSobHdNhiVjyg1Wm_dlgjLtcp1Vw9D8CRRz9nMKUc7hHTubtbMQs?purpose=inline)

### 3.2 What `fit()` learns

For every class \(c\):

|Attribute|What it stores|Shape|
|---|---|---|
|`class_priors_[c]`|Fraction of training samples belonging to class c|Scalar|
|`class_means_[c]`|Mean of each feature for class c|`(n_features,)`|
|`class_vars_[c]`|Variance of each feature for class c|`(n_features,)`|

Also:

- `classes_`: unique class labels.
    
- `var_smoothing`: controls numerical stabilization.
    

Mean:

\[ \mu_{c,i}=\frac{1}{N_c}\sum_{j:y_j=c}x_{j,i} \]

Population variance:

\[ \sigma^2_{c,i}= \frac{1}{N_c}\sum_{j:y_j=c}(x_{j,i}-\mu_{c,i})^2 \]

Variance smoothing: prevents zero or extremely small variances from causing numerical problems.

One implementation detail: scikit-learn's GaussianNB does not simply add a fixed `1e-9` to every variance. It calculates:

\[ \epsilon=\text{var\_smoothing}\times \max_i \operatorname{Var}(X_i) \]

Then adds this epsilon to each class-feature variance.

### 3.3 The log-density formula

\[ \boxed{ \log N(x;\mu,\sigma^2)= -\frac12\left[ \log(2\pi\sigma^2)+ \frac{(x-\mu)^2}{\sigma^2} \right] } \]

Two terms:

- \(\log(2\pi\sigma^2)\): controls the Gaussian's normalization and peak height.
    
- \(\frac{(x-\mu)^2}{\sigma^2}\): squared distance from the mean, scaled by variance.
    

Intuition: a value close to the mean generally receives a higher density, but the variance also affects the result. The complete formula determines the score.

### 3.4 The prediction formula

\[ \boxed{ \hat y= \arg\max_c \left[ \log P(c)+ \sum_{i=1}^{d} \log N(x_i;\mu_{c,i},\sigma^2_{c,i}) \right] } \]

Vectorized NumPy implementation:

```
log_density = -0.5 * (    np.log(2 * np.pi * var)    + ((X - mean) ** 2) / var)score = log_density.sum(axis=1) + log_prior
```

Broadcasting:

- `X`: `(n_samples, n_features)`
    
- `mean`: `(n_features,)`
    
- `var`: `(n_features,)`
    
- `log_density`: `(n_samples, n_features)`
    
- `log_density.sum(axis=1)`: `(n_samples,)`
    

`axis=1` sums across features for each sample, leaving one score per sample.

### 3.5 `predict_proba()` = softmax

\[ \boxed{ P(c\mid X)= \frac{\exp(s_c-k)} {\sum_{c'}\exp(s_{c'}-k)} } \]

Where \(s_c\) is the score for class c and \(k\) is the maximum score for that sample.

Why subtract the maximum?

It prevents numerical overflow when exponentiating large scores. Softmax is shift-invariant, so subtracting the same number from every class score does not change the probabilities.

```
scores -= scores.max(axis=1, keepdims=True)exp_scores = np.exp(scores)probabilities = (    exp_scores /    exp_scores.sum(axis=1, keepdims=True))
```

`keepdims=True`: preserves the shape `(n_samples, 1)`, allowing NumPy to broadcast correctly when subtracting or dividing.

## PART 4 — Multinomial Naive Bayes

### 4.1 When to use

Non-negative count or frequency features, especially when each feature represents the number of occurrences of an event.

Examples:

- Bag-of-words text: number of times each word appears in a document.
    
- Event logs: number of occurrences of each event type.
    
- Clickstreams: number of clicks on each category.
    

Not the natural choice for raw NSL-KDD numerical measurements.

For example:

- `src_bytes = 491` is a measurement of bytes transferred, not 491 occurrences of a word or event in a vocabulary.
    
- `protocol_type = tcp` is categorical, not a count.
    
- `count = 2` represents a count of connections, but feeding it as a single Multinomial NB feature does not automatically make the entire dataset a suitable document-term matrix.
    

For NSL-KDD, Gaussian NB is a reasonable baseline for continuous features. Categorical features may need a categorical model or suitable encoding. The dataset is mixed, so don't assume one NB variant is automatically ideal for every column.

### 4.2 The document-term matrix

Each row represents a document, and each column represents a word.

|Document|viagra|meeting|urgent|Class|
|---|---|---|---|---|
|1|2|0|1|spam|
|2|1|0|0|spam|
|3|0|3|1|ham|
|4|0|2|1|ham|
|5|0|4|2|ham|

The input matrix has shape:

\[ X=(n_{\text{documents}},V) \]

Where \(V\) is the vocabulary size.

Here:

\[ X.shape=(5,3) \]

### 4.3 What `fit()` learns

For every class \(c\), Multinomial NB calculates:

1. The total count of each feature across documents in that class.
    
2. The total number of feature occurrences in that class.
    
3. The probability of each feature given that class, using smoothing.
    
4. The class prior probability.
    

Laplace/Lidstone smoothing formula:

\[ \boxed{ P(w_i\mid c)= \frac{N_{i,c}+\alpha} {\sum_{j=1}^{V}N_{j,c}+\alpha V} } \]

Where:

- \(N_{i,c}\): total count of word \(i\) in class \(c\).
    
- \(V\): vocabulary size.
    
- \(\alpha\): smoothing parameter, commonly 1.
    

Trace: spam

The spam documents contain:

|Word|Total count|
|---|---|
|viagra|3|
|meeting|0|
|urgent|1|
|Total|4|

With \(V=3\) and \(\alpha=1\):

\[ P(viagra\mid spam)=\frac{3+1}{4+3}=\frac47 \]

\[ P(meeting\mid spam)=\frac{0+1}{4+3}=\frac17 \]

\[ P(urgent\mid spam)=\frac{1+1}{4+3}=\frac27 \]

Check:

\[ \frac47+\frac17+\frac27=1 \]

Trace: ham

|Word|Total count|
|---|---|
|viagra|0|
|meeting|9|
|urgent|4|
|Total|13|

With the same smoothing:

\[ P(viagra\mid ham)=\frac1{16}=0.0625 \]

\[ P(meeting\mid ham)=\frac{10}{16}=0.625 \]

\[ P(urgent\mid ham)=\frac5{16}=0.3125 \]

Check:

\[ \frac1{16}+\frac{10}{16}+\frac5{16}=1 \]

What is stored on `self`?

```
self.classes_ = ["spam", "ham"]self.class_priors_ = {    "spam": 0.4,    "ham": 0.6}self.feature_log_prob_ = {    "spam": [        np.log(4 / 7),        np.log(1 / 7),        np.log(2 / 7)    ],    "ham": [        np.log(1 / 16),        np.log(10 / 16),        np.log(5 / 16)    ]}
```

`feature_log_prob_` has shape:

\[ (n_{\text{classes}},V) \]

In this example: `(2, 3)`.

Important: the class prior is based on the number of documents, while the word probabilities are based on the total word counts across documents in each class.

### 4.4 The prediction formula

For a new document with feature counts \(X=(x_1,\ldots,x_V)\):

\[ \boxed{ s_c=\log P(c)+ \sum_{i=1}^{V}x_i\log P(w_i\mid c) } \]

Then:

\[ \hat y=\arg\max_c s_c \]

Why does the count multiply the log probability?

Because the original likelihood is:

\[ P(X\mid c)= \prod_{i=1}^{V}P(w_i\mid c)^{x_i} \]

A word appearing twice contributes its probability twice:

\[ P(w_i\mid c)^2 \]

Taking logarithms converts the exponent into multiplication:

\[ \log(P(w_i\mid c)^{x_i}) =x_i\log P(w_i\mid c) \]

Worked example:

Test document:

\[ X=(1,1,0) \]

It contains:

- Viagra once.
    
- Meeting once.
    
- Urgent zero times.
    

Spam score:

\[ \begin{aligned} s_{spam} &=\log(0.4)+\log(4/7)+\log(1/7)\\ &\approx -0.916-0.560-1.946\\ &=\boxed{-3.422} \end{aligned} \]

Ham score:

\[ \begin{aligned} s_{ham} &=\log(0.6)+\log(1/16)+\log(10/16)\\ &\approx -0.511-2.773-0.470\\ &=\boxed{-3.754} \end{aligned} \]

Since:

\[ -3.422>-3.754 \]

Prediction: spam

The spam score wins because the strong viagra signal outweighs the meeting signal favoring ham.

### 4.5 Why store log-probabilities in `fit()`?

Three reasons:

1. Efficiency: calculate logarithms once during training rather than repeatedly during prediction.
    
2. Numerical stability: multiplying many small probabilities can cause floating-point underflow. Adding their logarithms avoids this.
    
3. Compatibility: scikit-learn's `MultinomialNB` also stores `feature_log_prob_`.
    

Implementation:

```
self.feature_log_prob_[c] = np.log(    (counts_c + alpha) /    (counts_c.sum() + alpha * V))
```

Where `counts_c` is the sum of all feature counts across training rows belonging to class `c`.

### 4.6 Laplace smoothing

Without smoothing:

If a word never appears in a class:

\[ P(viagra\mid ham)=\frac0{13}=0 \]

Therefore:

\[ \log(0)=-\infty \]

This can make the entire class score negative infinity.

With smoothing:

\[ P(viagra\mid ham)=\frac{0+1}{13+3} =\frac1{16} \]

Now the probability is small but nonzero.

- \(\alpha=1\): Laplace smoothing.
    
- \(0<\alpha<1\): commonly called Lidstone smoothing.
    
- \(\alpha\) is configurable; scikit-learn's default is 1.0.
    

What does vocabulary size \(V\) do?

The denominator contains \(\alpha V\). As vocabulary size increases, the total smoothing mass increases relative to the observed counts.

For a word with zero occurrences:

\[ P(unseen\ word\mid c)= \frac{\alpha}{N_c+\alpha V} \]

A larger vocabulary generally makes the probability assigned to each individual unseen vocabulary word smaller.

Important: smoothing handles words that are in the model's vocabulary but have zero count in a particular class. A completely new word outside the vocabulary needs preprocessing, such as mapping it to an `<UNK>` token, or ignoring it.

### 4.7 Why this is different from NSL-KDD

Multinomial NB expects each input feature to represent a count-like quantity.

For text, the columns represent word types, and their values represent how many times those words occur.

For NSL-KDD, raw measurements such as bytes and duration are not word-frequency features. Treating them as such gives the counts a different meaning from what they represent in the original data.

Practical conclusion:

- Gaussian NB: useful baseline for continuous numerical features.
    
- Multinomial NB: natural for document-term matrices and event-count vectors.
    
- Categorical NB: appropriate for categorical features.
    
- Mixed datasets: may require feature-specific preprocessing or a model that handles different feature types.
    

## PART 5 — The Complete Reference

|Aspect|Gaussian NB|Multinomial NB|
|---|---|---|
|Feature type|Continuous numerical values|Non-negative counts|
|Likelihood|Gaussian density|Multinomial feature probability|
|Learns during `fit()`|Mean and variance per class and feature|Feature probabilities per class|
|Stored probabilities|Means and variances|Log feature probabilities|
|Class prior|Fraction of samples per class|Fraction of documents per class|
|Prediction score|\(\log P(c)+\sum_i\log N(x_i;\mu_{c,i},\sigma^2_{c,i})\)|\(\log P(c)+\sum_i x_i\log P(w_i\mid c)\)|
|Smoothing|Variance smoothing|Laplace/Lidstone smoothing|
|Natural application|Continuous measurements|Text and event counts|

### What both algorithms have in common

1. Both calculate class priors during `fit()`.
    
2. Both learn class-specific feature statistics/probabilities.
    
3. Both use the conditional independence assumption.
    
4. Both calculate class scores in log-space.
    
5. Both use `argmax` to select the predicted class.
    
6. Both can calculate class probabilities using normalization (softmax/log-sum-exp).
    

## PART 6 — Blank Page Test


Write Bayes' theorem and explain its four components.

Calculate the posterior probability in the rare-disease example. Why is it below 1%?

Write the Naive Bayes conditional independence assumption.

Why is the assumption called naive? Why can the algorithm still work?

Why does Naive Bayes reduce exponential parameter growth to linear growth?

When should you use Gaussian NB? What does it calculate during fit?

Explain the Gaussian log-density formula and the role of variance smoothing.

Explain NumPy broadcasting for Gaussian NB prediction. What does axis=1 do?

What is softmax? Why subtract the maximum score before exponentiating?

When should you use Multinomial NB? What does it learn during fit?

Calculate Laplace-smoothed word probabilities for a class.

Why does the word count multiply the log probability during prediction?

Why store log probabilities? What problem does Laplace smoothing solve?

Explain why raw NSL-KDD measurements are not naturally suited to Multinomial NB.

