# 📈 AI Engineer Rebuild — Progress Tracker

> **This file records what has actually been learned, built, tested, and defended.**

Do not mark something complete because you watched a tutorial.

A task is complete when you can **recall it, implement it, debug it, and explain it.**

---

# 🏁 Overall Progress

## 42-Day Status

```text
Days Completed:      0 / 42
Days In Progress:    1
Days Remaining:      41

Projects Completed:  0 / 4

Overall Status:
[ ] Not Started
[x] In Progress
[ ] Halfway
[ ] Final Week
[ ] Completed
```

---

# 📊 Weekly Progress

| Week   | Focus                      | Status | Score |
| ------ | -------------------------- | ------ | ----: |
| Week 1 | ML Foundations             | 🟡     |   /10 |
| Week 2 | PyTorch + Deep Learning    | ⬜      |   /10 |
| Week 3 | Production CV              | ⬜      |   /10 |
| Week 4 | NLP + RAG                  | ⬜      |   /10 |
| Week 5 | MLOps + Cloud              | ⬜      |   /10 |
| Week 6 | System Design + Interviews | ⬜      |   /10 |

### Status Legend

```text
⬜ Not Started
🟡 In Progress
🟢 Complete
🔴 Blocked
```

---

# 🧠 Daily Progress Template

Copy this section for every day.

```text
### Day x — title (log)

Date: 28/10
Hours: 3

## Topics

* [ ] Recognition
* [ ] Recall
* [ ] Construction

## Coding

* [ ] Completed
* [ ] Tested
* [ ]  Debugged

## Knowledge

* [ ] Can explain
* [ ]  Can implement
* [ ]  Can debug
* [ ]  Can defend

## Project

* [ ]  Working
* [ ] Tested
* [ ]  Documented

## Interview

* [ ]  Cold explanation
* [ ] Technical questions

#### What I Actually Learned

-

#### What I Still Don't Understand

-

#### Bugs / Problems

-

#### What I Need to Review Tomorrow

-

#### Evidence

Code:
Commit:
Demo:
Notes:

## Day Score

Recall:      /10
Coding:      /10
Engineering: /10
Debugging:   /10
Interview:   /10
```

---

# 🗓️ 42-Day Checklist

## Week 1 — ML Foundations

### Day 1 — Python Engineering

* [x] OOP
* [x] Decorators
* [x] Generators
* [x] Typing
* [x] Exceptions
* [x] Logging
* [x] Clean code
* [x] Virtual environments
* [x] Project structure
* [x] Coding completed
* [x] Explained from memory

#### Day 1 — Python Engineering (log)

**Date:** 2026-10-25
**Hours:** 3

**Topics**

- [x] Recognition
- [x] Recall
- [x] Construction

**Coding**

- [x] Completed
- [x] Tested
- [x] Debugged

**Knowledge**

- [ ] Can explain
- [ ] Can implement
- [ ] Can debug
- [ ] Can defend

**Project**

- [x] Working
- [x] Tested
- [x] Documented

**Interview**

- [ ] Cold explanation
- [ ] Technical questions

**Recall check**

**1. Why is `Preprocessor` a class but `validate_schema` can be a plain function?**

`Preprocessor` is a class because it needs to maintain state, such as configuration and fitted parameters. `validate_schema` can be a plain function because it performs a specific task without needing to store state between calls.

**2. What does the trailing `_` in `self.mean_` signal?**

A trailing underscore is a Python naming convention commonly used in machine learning libraries such as scikit-learn to indicate an attribute that is learned or estimated during `fit()`.

**3. What does `@functools.wraps(fn)` do, and why is it inside every decorator?**

`@functools.wraps(fn)` preserves the original function's metadata, such as its name, docstring, and module, when a decorator replaces it with a wrapper function.

**4. `yield` vs `return` — one sentence on the practical difference.**

`yield` hands back data one item at a time (lazily), while `return` returns all of the data at once.

> 🔧 Correction: tighten this to — `return` ends the function and gives back one result; `yield` pauses the function and produces values one at a time, keeping memory O(1).

**5. Why do we define `LoadError(DataError)` instead of raising `Exception("...")`?**

> ❌ Original answer was unclear ("we raise the error as we expect it at the field of its usage but raise just prints"). Needs a redo.
>
> 🔧 Correct answer: a specific subclass lets callers catch exactly that failure (`except LoadError`) or the whole family (`except DataError`), without catching unrelated bugs. A bare `Exception` forces callers to catch everything or nothing. Also, `raise` doesn't print — it stops execution and propagates the error up the call stack.

**6. Write from memory: a list comprehension that keeps only even numbers from `nums`.**

```python
even_nums = [n for n in nums if n % 2 == 0]
```

**7. Why `logger.info("x=%s", x)` instead of `logger.info(f"x={x}")`?**

The `%s` form formats the message only if the log level is enabled, but the f-string is formatted on every call, even when the message is discarded.

**8. What's the `if not logger.handlers` guard in `get_logger` for?**

It checks whether the logger already has directly attached handlers, which prevents duplicate handlers (and duplicate log lines) when the setup function is called repeatedly.

**9. What single command produces `requirements.txt`?**

```bash
pip freeze > requirements.txt
```

**10. What does `__init__.py` do — the two jobs?**

1. Makes the folder be treated as a package.
2. Defines the package's public face (the API surface).

**Bugs / Problems**

- 12 bugs fixed while building the `data` package — see Day 1 Revision Sheet, Part 10.

**What I Need to Review Tomorrow**

- Day 1 revision sheet (redo recall questions 4 and 5)

**Evidence**

- Code: DONE
- Commit: `Day1 - data package`
- Demo: —
- Notes: —

**Day Score**

```text
Recall:       6/10
Coding:       7/10
Engineering:  7/10
Debugging:    8/10
Interview:    7/10
```

### Day 2 — NumPy + Pandas

* [ ] Vectorization
* [ ] Broadcasting
* [ ] GroupBy
* [ ] Merge
* [ ] Missing values
* [ ] Feature engineering
* [ ] Preprocessing pipeline
* [ ] Tested



### Day 2 — NumPy + Pandas (log)

Date: 28/10
Hours: 3

## Topics

* [ ] Recognition
* [ ] Recall
* [ ] Construction

## Coding

* [ ] Completed
* [ ] Tested
* [ ]  Debugged

## Knowledge

* [ ] Can explain
* [ ]  Can implement
* [ ]  Can debug
* [ ]  Can defend

## Project

* [ ]  Working
* [ ] Tested
* [ ]  Documented

## Interview

* [ ]  Cold explanation
* [ ] Technical questions

#### What I Actually Learned

-

#### What I Still Don't Understand

-

#### Bugs / Problems

-

#### What I Need to Review Tomorrow

-

#### Evidence

Code:
Commit:
Demo:
Notes:

## Day Score

Recall:      /10
Coding:      /10
Engineering: /10
Debugging:   /10
Interview:   /10


### Day 3 — Statistics

* [ ] Distributions
* [ ] Bayes
* [ ] Covariance
* [ ] Correlation
* [ ] Hypothesis testing
* [ ] Bias/variance
* [ ] Overfitting
* [ ] Leakage
* [ ] Naive Bayes from scratch
* [ ] Compared with sklearn

### Day 4 — Linear Algebra

* [ ] Matrix multiplication
* [ ] Eigenvalues/eigenvectors
* [ ] SVD
* [ ] Norms
* [ ] PCA from scratch
* [ ] Compared with sklearn

### Day 5 — Classical ML

* [ ] Linear Regression
* [ ] Logistic Regression
* [ ] Ridge
* [ ] Lasso
* [ ] KNN
* [ ] Decision Trees
* [ ] Random Forest
* [ ] Boosting
* [ ] XGBoost
* [ ] K-Means
* [ ] PCA
* [ ] Model comparison

### Day 6 — Evaluation

* [ ] Precision
* [ ] Recall
* [ ] F1
* [ ] ROC-AUC
* [ ] PR-AUC
* [ ] Confusion Matrix
* [ ] MAE
* [ ] MSE
* [ ] RMSE
* [ ] R²
* [ ] Cross-validation
* [ ] SQL warm-up
* [ ] Metrics dashboard

### Day 7 — Anomaly Detection

* [ ] Validation
* [ ] EDA
* [ ] Feature engineering
* [ ] Baseline
* [ ] Multiple models
* [ ] Evaluation
* [ ] Error analysis
* [ ] MLflow
* [ ] FastAPI
* [ ] README

---

## Week 2 — PyTorch + Deep Learning

### Day 8 — Neural Network Fundamentals

* [ ] Forward propagation
* [ ] Backpropagation
* [ ] Activations
* [ ] Loss functions
* [ ] Gradient descent
* [ ] NumPy neural network

### Day 9 — PyTorch Basics

* [ ] PyTorch tensors
* [ ] Dataset
* [ ] DataLoader
* [ ] nn.Module
* [ ] forward()
* [ ] Loss
* [ ] Optimizer
* [ ] backward()
* [ ] step()
* [ ] PyTorch implementation

### Day 10 — Training Practice

* [ ] Early stopping
* [ ] LR scheduling
* [ ] Weight decay
* [ ] Dropout
* [ ] BatchNorm
* [ ] Augmentation
* [ ] Checkpoints
* [ ] Experiment tracking

### Day 11 — CNNs

* [ ] Convolution
* [ ] Kernels
* [ ] Padding
* [ ] Stride
* [ ] Pooling
* [ ] Receptive fields
* [ ] Feature maps
* [ ] PyTorch CNN

### Day 12 — Transfer Learning

* [ ] ResNet
* [ ] EfficientNet
* [ ] MobileNet
* [ ] Feature extraction
* [ ] Fine-tuning
* [ ] Freezing
* [ ] Differential learning rates
* [ ] Fine-tuned model

### Day 13 — Vision Tasks

* [ ] Detection
* [ ] Classification
* [ ] Segmentation
* [ ] OCR
* [ ] IoU
* [ ] NMS
* [ ] mAP
* [ ] Confidence thresholds
* [ ] YOLO pipeline explanation

### Day 14 — Inference + Optimization

* [ ] ONNX
* [ ] Inference
* [ ] Latency
* [ ] Benchmarking
* [ ] LeetCode

---

## Week 3 — Production Computer Vision

### Day 15 — Object Detection

* [ ] Anchor-based detection
* [ ] Anchor-free detection
* [ ] IoU
* [ ] NMS
* [ ] mAP
* [ ] YOLO evolution
* [ ] R-CNN family

### Day 16 — Segmentation

* [ ] Semantic segmentation
* [ ] Instance segmentation
* [ ] U-Net
* [ ] Mask R-CNN
* [ ] Pose estimation
* [ ] Segmentation demo

### Day 17 — OCR

* [ ] Tesseract
* [ ] CRNN
* [ ] TrOCR
* [ ] PaddleOCR
* [ ] OCR comparison
* [ ] Failure analysis

### Day 18 — FastAPI

* [ ] FastAPI
* [ ] GET
* [ ] POST
* [ ] Validation
* [ ] Response models
* [ ] File uploads
* [ ] Exceptions
* [ ] `/detect`
* [ ] `/health`

### Day 19 — Docker

* [ ] Dockerfile
* [ ] Images
* [ ] Containers
* [ ] Volumes
* [ ] Networks
* [ ] Environment variables
* [ ] Docker Compose

### Day 20 — Testing

* [ ] pytest
* [ ] Unit tests
* [ ] Integration tests
* [ ] API tests
* [ ] Invalid input tests
* [ ] Error tests

### Day 21 — Deployment

* [ ] Deployment
* [ ] Latency measurement
* [ ] Throughput measurement
* [ ] Failure rate
* [ ] External API access

---

## Week 4 — NLP + RAG

### Day 22 — NLP Foundations

* [ ] Tokenization
* [ ] TF-IDF
* [ ] Word2Vec
* [ ] Sequence models
* [ ] Attention

### Day 23 — Transformers

* [ ] Embeddings
* [ ] Positional encoding
* [ ] Self-attention
* [ ] Multi-head attention
* [ ] Feed-forward
* [ ] LayerNorm
* [ ] Scaled dot-product attention

### Day 24 — Hugging Face

* [ ] Hugging Face tokenizers
* [ ] Datasets
* [ ] Pipelines
* [ ] Fine-tuning
* [ ] Model Hub
* [ ] BERT/DistilBERT project

### Day 25 — Semantic Search

* [ ] Cosine similarity
* [ ] Semantic search
* [ ] FAISS
* [ ] Chroma
* [ ] Chunking

### Day 26 — RAG Concepts

* [ ] RAG architecture
* [ ] Top-k
* [ ] Context windows
* [ ] Hallucinations
* [ ] Chunk overlap
* [ ] Reranking

### Days 27–28 — RAG Project

* [ ] Document loader
* [ ] Chunking
* [ ] Embeddings
* [ ] Vector DB
* [ ] Retriever
* [ ] LLM
* [ ] Source citations
* [ ] FastAPI
* [ ] Streamlit
* [ ] Docker

---

## Week 5 — MLOps + Cloud

### Day 29 — ML Lifecycle

* [ ] ML lifecycle
* [ ] Validation
* [ ] Training
* [ ] Evaluation
* [ ] Experiment tracking
* [ ] Registry
* [ ] Deployment
* [ ] Monitoring
* [ ] Retraining

### Day 30 — MLflow

* [ ] MLflow parameters
* [ ] Metrics
* [ ] Artifacts
* [ ] Experiments
* [ ] Model registry
* [ ] 3+ experiments

### Day 31 — Data Validation + Drift

* [ ] Schema validation
* [ ] Missing values
* [ ] Outliers
* [ ] Drift
* [ ] Pydantic
* [ ] Basic drift check

### Day 32 — CI

* [ ] GitHub Actions
* [ ] Automated tests
* [ ] Docker build
* [ ] CI pipeline

### Day 33 — Monitoring

* [ ] Latency
* [ ] Throughput
* [ ] Errors
* [ ] Prediction distribution
* [ ] Drift
* [ ] Prediction logging

### Day 34 — AWS

* [ ] S3
* [ ] EC2
* [ ] IAM
* [ ] ECR
* [ ] CloudWatch
* [ ] AWS deployment concepts

### Day 35 — End-to-End Deployment

* [ ] GitHub
* [ ] CI/CD
* [ ] Docker
* [ ] MLflow
* [ ] FastAPI
* [ ] AWS
* [ ] Monitoring
* [ ] Public deployment

---

## Week 6 — System Design + Interviews

### Day 36 — ML System Design

* [ ] Vehicle detection system design
* [ ] Content moderation design
* [ ] Data architecture
* [ ] Model architecture
* [ ] API design
* [ ] Latency
* [ ] CPU/GPU
* [ ] Scaling
* [ ] Caching
* [ ] Database
* [ ] Monitoring
* [ ] Retraining

### Day 37 — SQL

* [ ] JOIN
* [ ] GROUP BY
* [ ] CASE
* [ ] CTE
* [ ] Subqueries
* [ ] ROW_NUMBER
* [ ] RANK
* [ ] LAG
* [ ] LEAD
* [ ] Window functions
* [ ] 10–12 problems

### Day 38 — DSA

* [ ] Arrays
* [ ] Strings
* [ ] Hashmaps
* [ ] Stacks
* [ ] Queues
* [ ] Linked Lists
* [ ] Trees
* [ ] Graphs
* [ ] Binary Search
* [ ] Sorting
* [ ] Big-O
* [ ] 8–10 medium problems

### Day 39 — Interview Drills

* [ ] ML interview drill
* [ ] DL interview drill
* [ ] CV interview drill
* [ ] NLP interview drill
* [ ] MLOps interview drill

### Day 40 — Project Defense

* [ ] Project defense
* [ ] Model choice
* [ ] Metric choice
* [ ] Dataset split
* [ ] Leakage
* [ ] Distribution shift
* [ ] Latency
* [ ] Scaling
* [ ] Retraining
* [ ] Failure handling
* [ ] Monitoring

### Day 41 — Documentation + Portfolio

* [ ] Project READMEs
* [ ] Architecture diagrams
* [ ] Installation instructions
* [ ] Dataset documentation
* [ ] Training documentation
* [ ] Evaluation documentation
* [ ] Results
* [ ] API docs
* [ ] Example requests
* [ ] Limitations
* [ ] Future work
* [ ] Demo
* [ ] Resume update
* [ ] 5 STAR stories

### Day 42 — Final Review

* [ ] Final capstone
* [ ] Technical mock interview
* [ ] Behavioral mock interview
* [ ] Project defense recording
* [ ] Final knowledge-gap review

---

# 🏗️ Project Progress

## Project 1 — Network Traffic Anomaly Detection 2.0

Status:

```text
[ ] Not Started
[ ] In Progress
[ ] Complete
```

### Engineering

* [ ] Data validation
* [ ] EDA
* [ ] Feature engineering
* [ ] Train/validation/test split
* [ ] Baseline
* [ ] Multiple models
* [ ] Evaluation
* [ ] Error analysis
* [ ] MLflow
* [ ] FastAPI
* [ ] Tests
* [ ] README
* [ ] Deployment

### Defense

* [ ] Explain model choice
* [ ] Explain metrics
* [ ] Explain split
* [ ] Explain leakage prevention
* [ ] Explain failure cases
* [ ] Explain production monitoring

---

## 🚗 Project 2 — Intelligent Transportation AI

Status:

```text
[ ] Not Started
[ ] In Progress
[ ] Complete
```

### CV/OCR

* [ ] Object detection
* [ ] OCR
* [ ] Confidence handling
* [ ] Error handling
* [ ] Benchmarking

### Engineering

* [ ] FastAPI
* [ ] `/detect`
* [ ] `/health`
* [ ] Docker
* [ ] Tests
* [ ] CI/CD
* [ ] Deployment
* [ ] Monitoring

### Advanced

* [ ] PostgreSQL
* [ ] MLflow
* [ ] AWS

---

## 🤟 Project 3 — SignVision v2

Status:

```text
[ ] Not Started
[ ] In Progress
[ ] Complete
```

* [ ] PyTorch implementation
* [ ] Training pipeline
* [ ] Evaluation
* [ ] Real-time inference
* [ ] ONNX
* [ ] Optimization
* [ ] Latency benchmark
* [ ] Documentation
* [ ] Deployment/demo

---

## 🤖 Project 4 — AI Engineering Docs RAG Assistant

Status:

```text
[ ] Not Started
[ ] In Progress
[ ] Complete
```

* [ ] Document loader
* [ ] Chunking
* [ ] Embeddings
* [ ] Vector database
* [ ] Retriever
* [ ] LLM
* [ ] RAG pipeline
* [ ] Citations
* [ ] FastAPI
* [ ] Frontend
* [ ] Docker
* [ ] Evaluation
* [ ] Documentation

---

# 🧪 Weekly Recall Score

At the end of every week:

| Category    | Score |
| ----------- | ----: |
| Recall      |   /10 |
| Coding      |   /10 |
| Engineering |   /10 |
| Debugging   |   /10 |
| Interview   |   /10 |

### Interpretation

Do not use this score as a competition.

Use it to find weaknesses.

For example:

```text
Recall:       8/10
Coding:       9/10
Engineering:  5/10
Debugging:    4/10
Interview:    6/10
```

This means the problem is not necessarily knowledge.

The problem may be **engineering practice and debugging under pressure**.

---

# 🧠 Knowledge-Gap Log

Use this section whenever something repeatedly breaks.

| Topic                | Problem                              | Cause                  | Action                              | Status |
| -------------------- | ------------------------------------ | ---------------------- | ----------------------------------- | ------ |
| Example: PyTorch     | Forgot training loop                 | Weak recall            | Rebuild from memory                 | 🟡     |
| Python: `yield`      | One-sentence answer was imprecise    | Weak recall            | Re-read sheet §4, re-answer cold    | 🟡     |
| Python: exceptions   | Couldn't explain `LoadError(DataError)` | Weak recall         | Re-read sheet §5.1, re-answer cold  | 🟡     |
|                      |                                      |                        |                                     |        |

---

# 🐛 Bug Log

Keep a record of important bugs instead of deleting them from history.

| Date       | Project      | Bug                                   | Root Cause                    | Fix                                | Lesson                                  |
| ---------- | ------------ | ------------------------------------- | ----------------------------- | ---------------------------------- | --------------------------------------- |
| 2026-10-25 | `data` pkg   | `yield record` outside the `try`      | Yield not on the happy path   | Move `yield` inside `try`          | `yield` goes on the success branch      |
| 2026-10-25 | `data` pkg   | Decorator returned `None`             | Missing `return wrapper`      | Add `return wrapper`               | Check the last line of every decorator  |
|            |              |                                       |                               |                                    |                                         |

The goal is to build a personal **debugging knowledge base**. (Full list of Day 1 bugs: Day 1 Revision Sheet, Part 10.)

---

# 🎤 Interview Defense Log

Record questions you could not answer immediately.

| Question                                                   | Topic  | Could Answer? | Review Needed |
| ---------------------------------------------------------- | ------ | ------------- | ------------- |
| Why `LoadError(DataError)` instead of `Exception("...")`?  | Python | 🔴            | Yes           |
| `yield` vs `return` — practical difference                 | Python | 🟡            | Yes           |
|                                                            |        | ⬜            |               |

---

# 🚦 Current Bottleneck

At any point during the 42 days, answer:

```text
What is currently slowing me down?

→

Why?

→

What is the smallest action that fixes it?

→
```

Do not respond to falling behind by adding more resources.

Reduce scope first.

---

# 🏁 Definition of Done

## ML

* [ ] Build an ML pipeline from raw data
* [ ] Select appropriate metrics
* [ ] Detect leakage
* [ ] Compare models
* [ ] Explain bias/variance
* [ ] Perform error analysis

## Deep Learning

* [ ] Implement a neural network
* [ ] Train PyTorch models
* [ ] Debug training
* [ ] Fine-tune pretrained models
* [ ] Explain CNNs
* [ ] Export models

## Computer Vision

* [ ] Explain YOLO
* [ ] Explain IoU/NMS/mAP
* [ ] Build inference pipelines
* [ ] Handle OCR
* [ ] Benchmark inference

## NLP / LLM

* [ ] Explain attention
* [ ] Use Transformers
* [ ] Generate embeddings
* [ ] Build semantic search
* [ ] Build RAG
* [ ] Handle citations and retrieval quality

## Engineering

* [ ] Build FastAPI services
* [ ] Write tests
* [ ] Dockerize applications
* [ ] Use PostgreSQL
* [ ] Build CI/CD
* [ ] Track experiments with MLflow

## MLOps

* [ ] Deploy a model
* [ ] Monitor inference
* [ ] Understand drift
* [ ] Track model versions
* [ ] Understand production metrics

## Cloud

* [ ] Deploy on AWS
* [ ] Understand S3
* [ ] Understand EC2
* [ ] Understand IAM
* [ ] Understand ECR
* [ ] Understand CloudWatch

## Interviews

* [ ] Explain ML concepts cold
* [ ] Explain DL concepts cold
* [ ] Defend CV decisions
* [ ] Explain RAG architecture
* [ ] Discuss MLOps
* [ ] Design ML systems
* [ ] Solve SQL problems
* [ ] Solve DSA problems
* [ ] Defend all portfolio projects

---

# 🏆 Final 42-Day Review

## Technical

* [ ] ML knowledge recovered
* [ ] PyTorch comfortable
* [ ] CV engineering improved
* [ ] NLP/RAG implemented
* [ ] FastAPI comfortable
* [ ] Docker comfortable
* [ ] Testing implemented
* [ ] MLflow used
* [ ] CI/CD implemented
* [ ] Monitoring implemented
* [ ] AWS deployment completed

## Portfolio

* [ ] Network anomaly project
* [ ] Intelligent Transportation AI
* [ ] SignVision v2
* [ ] RAG Assistant

## Interview

* [ ] ML questions
* [ ] DL questions
* [ ] CV questions
* [ ] NLP questions
* [ ] MLOps questions
* [ ] System design
* [ ] SQL
* [ ] DSA
* [ ] Project defense

---

# 🔥 Final Self-Test

Close every notebook.

Close every tutorial.

Then answer:

> **"Give me a machine-learning problem. Can I reason about the data, build the model, evaluate it correctly, expose it through an API, test it, containerize it, deploy it, monitor it, debug it, and explain every engineering decision I made?"**

If the answer is **yes**, the rebuild achieved its purpose.

If the answer is **no**, the gaps identified here become the next iteration of the plan.

---

# 🚀 Final Status

```text
42 Days
    ↓
4 Portfolio Projects
    ↓
ML + DL + CV + NLP
    ↓
FastAPI + Docker + Testing
    ↓
MLflow + CI/CD + Monitoring
    ↓
AWS
    ↓
System Design
    ↓
Interview Defense
    ↓
AI / ML Engineer
```

**Build → Break → Debug → Explain → Ship.**
