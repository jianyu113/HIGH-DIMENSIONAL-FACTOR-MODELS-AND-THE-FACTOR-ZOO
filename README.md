# HIGH-DIMENSIONAL FACTOR MODELS AND THE FACTOR ZOO

This project is a small-scale empirical reproduction of the main idea in  
**"HIGH-DIMENSIONAL FACTOR MODELS AND THE FACTOR ZOO"** (Lettau, Pelger, and others).
http://www.nber.org/papers/w31719

The goal is to show how **tensor factor models (TFM)** can extract **low-dimensional, economically interpretable latent factors** from a high-dimensional panel of asset characteristics, and to compare this approach with a traditional **PCA baseline**.

---

## 1. Motivation

In modern asset pricing, hundreds of characteristics ("factor zoo") exist.  
However, many of them are highly correlated and driven by a much smaller number of **latent economic dimensions**.

Traditional approaches:
- Either sort assets on individual characteristics
- Or apply PCA to a flattened return/feature matrix

These approaches:
- Ignore the multi-dimensional structure of the data
- Often produce factors that are hard to interpret

This project follows the idea of the paper:
> Instead of reducing dimension in the return space, we reduce dimension **in the characteristic space** using a tensor factor model.

---

## 2. Methodology

We construct a 3D tensor with dimensions:

- t = time (monthly)
- n = ETF
- c = characteristic (12 financial features)

We then apply a **CP (PARAFAC) tensor decomposition**:


This produces:

- λ(t, k): time factors
- γ(n, k): asset loadings
- β(c, k): characteristic loadings

These latent factors correspond to **structural economic dimensions** such as:

- Risk / volatility
- Liquidity / trading activity
- Size / defensive style
- Illiquidity-driven rallies

We then apply a **CP (PARAFAC) tensor decomposition**:

X(t,n,c) ≈ ∑_{k=1}^K λ(t,k) · γ(n,k) · β(c,k)

---

## 3. Data

- Assets: 29 liquid ETFs (SPY, QQQ, IWM, sector ETFs, bonds, commodities, etc.)
- Frequency: Monthly
- Period: 2015–2025
- Source: Yahoo Finance (via yfinance)

Constructed characteristics (12 total):

- Past returns: 1m, 3m, 6m
- Volatility: 3m, 6m
- Turnover proxy
- Size proxy
- Amihud illiquidity
- Beta (6m)
- Sharpe (6m)
- Momentum (12m)
- Volume change (6m)

All characteristics are cross-sectionally z-scored each month.

---

## 4. Pipeline

1. Download ETF price and volume data
2. Compute 12 characteristics for each ETF and month
3. Build a balanced tensor of shape:
(T, N, C) = (time, ETF, characteristics)


4. Apply CP decomposition with rank = 4
5. Obtain:
   - Time factors (λ_t)
   - Asset exposures (γ_n)
   - Characteristic loadings (β_c)
6. As a baseline, flatten the tensor and apply PCA
7. Compare:
   - Time series behavior
   - Explained variance
   - Correlation structure

---

## 5. Main Results (Tensor Factors)

The tensor model extracts 4 highly interpretable latent factors:

- F1: High volatility / small-cap / high-beta risk factor
- F2: Trading activity / volume shock factor
- F3: Large-cap / low-beta defensive factor
- F4: Illiquidity-driven speculative factor

Each factor has a clear economic interpretation from the characteristic loadings.

---

## 6. Comparison with PCA (Baseline)

We flatten the tensor into:

X_pca(t) ∈ R^(N × C)


and apply PCA with 4 components.

Results:

- First 4 PCA components explain only ~36% of total variation
- PCA factors have no clear economic interpretation
- Correlation analysis shows:
  > Each PCA component mixes multiple tensor factors

This confirms the paper's main claim:

> PCA captures statistical variation, while tensor factor models recover structured, interpretable economic dimensions.

---

## 7. How to Run

```bash
pip install numpy pandas yfinance tensorly scikit-learn matplotlib seaborn
```
---

## 8. Project Structure
```text
.
├── get_data.py              # Download ETF data
├── get_factor_value.py      # Compute characteristics
├── main.py                  # Main pipeline: tensor + PCA + plots
├── README.md
