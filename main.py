from get_data import *
from get_factor_value import *
from tensorly.decomposition import parafac
import tensorly as tl
import matplotlib.pyplot as plt


etf_id = [
    "SPY","IVV","VOO","QQQ","IWM","DIA","EFA","EEM","EWJ","XLK","XLF","XLY","XLP","XLI","XLE","XLV","XLU","XLB","TLT","IEF","SHY","GLD","SLV","USO","MTUM","QUAL","SIZE","USMV","VLUE",
]

factor_names = [
    "past_1m_return","past_3m_return","past_6m_return",
    "volatility_3m","volatility_6m",
    "turnover_proxy",
    "size_factor",
    "amihud",
    "beta_6m",
    "sharpe_6m",
    "momentum_12m",
    "volume_6m_pct_change",
]

panel = get_etf_price_panel(etf_id, start="2015-01-01", end="2025-01-01", interval="1mo", auto_adjust=False,)
factor_df = panel[["Date", "etf_id"]].drop_duplicates()

#factor_past_1m_return = past_1m_return(panel)
#factor_df = factor_df.merge(factor_past_1m_return, on=["Date", "etf_id"], how="left")

# merge all factor value
def build_factor_df(panel):
    base = panel[["Date", "etf_id"]].drop_duplicates().copy()
    base = base.merge(past_1m_return(panel), on=["Date", "etf_id"], how="left")
    base = base.merge(past_3m_return(panel), on=["Date", "etf_id"], how="left")
    base = base.merge(past_6m_return(panel), on=["Date", "etf_id"], how="left")
    base = base.merge(volatility_3m(panel), on=["Date", "etf_id"], how="left")
    base = base.merge(volatility_6m(panel), on=["Date", "etf_id"], how="left")
    base = base.merge(turnover_proxy(panel), on=["Date", "etf_id"], how="left")
    base = base.merge(size_factor(panel), on=["Date", "etf_id"], how="left")
    base = base.merge(amihud(panel), on=["Date", "etf_id"], how="left")
    base = base.merge(beta_6m(panel), on=["Date", "etf_id"], how="left")
    base = base.merge(sharpe_6m(panel), on=["Date", "etf_id"], how="left")
    base = base.merge(momentum_12m(panel), on=["Date", "etf_id"], how="left")
    base = base.merge(volume_6m_pct_change(panel), on=["Date", "etf_id"], how="left")
    return base

def df_to_tensor(factor_df):
    X = (factor_df.pivot(index="Date", columns="etf_id"))
    X = X.to_numpy().reshape(factor_df["Date"].nunique(),factor_df["etf_id"].nunique(),-1)
    return X

factor_df = build_factor_df(panel)
tensor = df_to_tensor(factor_df)
#print(tensor.shape)

factor_cols = [c for c in factor_df.columns if c not in ["Date", "etf_id"]]
#print(factor_df[factor_cols].isna().mean())

factor_cols = [c for c in factor_df.columns if c not in ["Date", "etf_id"]]
# zscore
factor_df[factor_cols] = (factor_df.groupby("Date")[factor_cols].transform(lambda x: (x - x.mean()) / x.std()))

factor_df[factor_cols] = factor_df[factor_cols].fillna(0)
tensor = df_to_tensor(factor_df)  # shape (120, 29, 12)
print(np.isnan(tensor).sum())

tl.set_backend('numpy')

#cp = parafac(tensor, rank=5)
cp = parafac(tensor, rank=4)  # change rank to 4

weights, factor_mats = cp
time_factor = factor_mats[0]
asset_factor = factor_mats[1]
feature_factor = factor_mats[2]

rank = feature_factor.shape[1]
latent_names = [f"F{k+1}" for k in range(rank)]
beta_df = pd.DataFrame(feature_factor, index=factor_names, columns=latent_names)
print(beta_df.round(3))

# draw plt
dates = np.sort(factor_df["Date"].unique())
factor_names = ["F1", "F2", "F3", "F4"]
lambda_df = pd.DataFrame(
    time_factor,
    index=dates,
    columns=factor_names
)
plt.figure(figsize=(10, 8))

for i, f in enumerate(factor_names, 1):
    plt.subplot(4, 1, i)
    lambda_df[f].plot()
    plt.title(f"{f} over 10 years")
    plt.xlabel("")
    plt.tight_layout()

plt.show()

# PCA(baseline)
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# tensor: shape (T, N, C)
T, N, C = tensor.shape

# flatten pannel
X_pca = tensor.reshape(T, N * C)

# take same dimensions as TFM
pca = PCA(n_components=4)
pca_factors = pca.fit_transform(X_pca)   # shape (T, 4)

# plotting
pca_df = pd.DataFrame(
    pca_factors,
    index=dates,
    columns=["PC1", "PC2", "PC3", "PC4"]
)

print(pca.explained_variance_ratio_)
print(pca_df.head())
print("Explained variance:", pca.explained_variance_ratio_)
print("Cumulative:", pca.explained_variance_ratio_.cumsum())

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(4, 1, i+1)
    pca_df.iloc[:, i].plot()
    plt.title(f"PC{i+1}")
    plt.tight_layout()
plt.show()

# We now have two sets of time factors
# structural comparison to show they were not capturing the same underlying economic dimensions

# compute the correlation matrix
# align on dates
common_dates = lambda_df.index.intersection(pca_df.index)
F = lambda_df.loc[common_dates]
PC = pca_df.loc[common_dates]

# calaulate correlation matrix (Tensor factors vs PCA factors)
combined = pd.concat([F, PC], axis=1)
corr_full = combined.corr()
corr_matrix = corr_full.loc[F.columns, PC.columns]



print(corr_matrix.round(2))