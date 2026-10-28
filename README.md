# Prediction vs. decision: code and sources

Code, results and sources behind the article **"Your AI May Be Accurate and Still Make Bad Decisions"** by Philip Denkabe.

The article argues that companies point machine learning at the wrong question. They predict who will default, churn or respond, but a decision depends on what an action will *change*. This repository has everything needed to reproduce the article's numbers.

## What's here

| Folder | Contents |
|---|---|
| `simulation/` | A simulated 1-million-account credit card portfolio, used to compare ways of choosing 200,000 accounts for a $500 credit-line increase. Because it is a simulation, the true effect of the increase on every account is known. |
| `reanalysis/` | Reanalysis of two public randomized experiments: Kevin Hillstrom's e-mail test and Criteo's advertising incrementality tests. |
| `exhibits/` | Charts for Exhibits 1 and 2 of the article. |
| `SOURCES.md` | Every published source the article cites, with links. |

### Simulation

| File | What it does | Article |
|---|---|---|
| `sim.py` | Original simulation (first draft): random, lowest-risk, highest-usage and incremental-value policies; customer segments; risk-decile analysis. | First-draft figures |
| `sim2.py` | Revised simulation. Adds an expected-profit benchmark (no experiment), counts the pilot's cost and capacity, and runs 11 sensitivity scenarios × 4 random seeds. | Exhibits 1 and 2, "About the Simulation" |
| `consumer.py` | For the base case, counts the additional defaults each policy causes among the customers it chooses. Lender value and borrower outcomes are different measures. | "Customer Outcomes Are a Separate Ledger" |
| `constrained.py` | Tests the policy the article recommends: maximize predicted value subject to a cap on how much the increase may raise each customer's predicted probability of default (learned from the same pilot). Caps of 0.5 to 3 percentage points, four seeds. | "Customer Outcomes Are a Separate Ledger" |
| `constrained_results.json` | Output of `constrained.py`. | |
| `results_v1.json` | Output of `sim.py`. | |
| `results_v2.json` | Output of `sim2.py` (44 runs). | |

### Reanalysis of public experiments

| File | What it does |
|---|---|
| `hillstrom.py` | Compares targeting the top 25% by predicted response with targeting by predicted effect of the e-mail, over 50 random splits. |
| `criteo.py` | Same comparison on a random 10% sample of the Criteo uplift dataset (about 1.4 million users), at the top 10% and 25%, over 10 random splits. |

These are retrospective reanalyses of real randomized tests. Neither company deployed the targeting rules compared here. In the Criteo data, the treatment flag marks random *assignment* to the advertising group, not confirmed ad exposure, and Criteo subsampled the public data non-uniformly for privacy, so the rates describe this sample, not Criteo's business-wide effects.

## Key results

**Simulation, base case** (mean of four seeds; value created by 200,000 increases):

| Policy | Needs an experiment? | Value |
|---|---|---|
| Highest predicted usage | No | −$16.3M |
| At random | No | −$7.9M |
| Lowest predicted risk | No | $0.7M |
| Highest expected profit | No | $2.8M |
| Highest predicted incremental value | Yes (60,000-account pilot, one-time cost $1.2M) | $3.6M |
| Perfect hindsight (ceiling) | — | $8.0M |

The incremental-value policy beat the expected-profit rule in all 44 runs. Capping each customer's predicted increase in default probability at 3 percentage points still created $3.0M with about 1,800 additional defaults, against $2.8M and about 3,900 for the expected-profit rule (`constrained_results.json`). The gap is modest in normal conditions (+$0.9M) and large when the increase itself makes customers riskier or the economy weakens (+$2.4M to +$3.8M), because the expected-profit rule assumes an increase does not change default risk. Full results by scenario are in `results_v2.json`.

**Limits.** The simulation tests one decision, not a customer lifecycle. Its parameters are typical of a U.S. near-prime card portfolio and are not drawn from any company's data. It shows a mechanism, not the size of the gain in any real portfolio.

## How to reproduce

Requires Python 3.10 or later.

```bash
pip install -r requirements.txt

# Simulation
cd simulation
python sim.py          # first-draft results -> results.json (a few minutes)
python sim2.py         # revised results -> results_v2.json (about 25 minutes on 2 cores)
python consumer.py 7   # additional defaults by policy, seed 7
python constrained.py  # value vs. additional defaults under borrower-protection caps (about 2 minutes)

# Reanalysis (download the data first; see below)
cd ../reanalysis
python hillstrom.py path/to/hillstrom.csv "Womens E-Mail"
python hillstrom.py path/to/hillstrom.csv "Mens E-Mail"
python criteo.py path/to/criteo_sample.csv.gz
```

Exact numbers can differ slightly across versions of scikit-learn and NumPy.

## Data

The simulation generates its own data. The two public datasets are not included here; download them from their sources:

- **Hillstrom e-mail test** (64,000 customers): [MineThatData](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html). Also available through [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/hillstrom).
- **Criteo uplift dataset**: [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/), under Criteo's data terms of use. The article's analysis uses a random 10% sample of rows (about 1.4 million). To draw a similar sample: `pandas.read_csv(path).sample(frac=0.10, random_state=0)`.

## Disclosure

Generative AI was used to help write and run parts of this code and to and edit the article. The author checked the code, results and claims.

## License

Code is released under the MIT License (see `LICENSE`). The public datasets are subject to their own terms.
