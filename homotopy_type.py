import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def homotopy_path_equivalence(returns, macro_factor):
    """
    Compute homotopy path equivalence between two points in the market space.
    Paths are considered equivalent if they can be continuously deformed.
    """
    if len(returns) < 5:
        return 0
    # Split returns into two paths
    half = len(returns) // 2
    path1 = returns[:half]
    path2 = returns[half:]
    # Normalise paths
    path1 = (path1 - path1.mean()) / (path1.std() + 1e-8)
    path2 = (path2 - path2.mean()) / (path2.std() + 1e-8)
    # Compute path equivalence: how similar are the shapes?
    # Use dynamic time warping or correlation
    from scipy.spatial.distance import euclidean
    from scipy.signal import correlate
    if len(path1) < 2 or len(path2) < 2:
        return 0
    # Correlation of paths (shape equivalence)
    corr = np.correlate(path1, path2, mode='full')
    max_corr = np.max(corr)
    # Normalise by length
    equivalence = max_corr / len(path1)
    return equivalence

def higher_inductive_types(returns, macro_factor, levels=3):
    """
    Construct higher inductive types (HITs) from the return data.
    HITs capture higher-order structure beyond simple paths.
    """
    if len(returns) < 10:
        return 0
    # Use hierarchical clustering to find higher-order structure
    # HITs correspond to different levels of the hierarchy
    # Reshape returns for clustering
    X = returns.reshape(-1, 1)
    # Perform hierarchical clustering
    Z = linkage(X, method='ward')
    # Count clusters at different levels
    hit_structure = 0
    for i in range(1, levels + 1):
        n_clusters = len(np.unique(fcluster(Z, t=i, criterion='maxclust')))
        hit_structure += n_clusters * i
    return hit_structure

def univalence_axiom(returns, macro_factor, threshold=0.5):
    """
    Apply the univalence axiom: transport strategies across isomorphic market structures.
    Returns the transportability score.
    """
    if len(returns) < 10:
        return 0
    # Compute the "isomorphism" between different time periods
    half = len(returns) // 2
    period1 = returns[:half]
    period2 = returns[half:]
    # Isomorphism = similarity of distributions
    from scipy.stats import ks_2samp
    ks_stat, p_value = ks_2samp(period1, period2)
    # Transportability = 1 - KS statistic (higher = more transportable)
    transportability = 1 - ks_stat
    return transportability

def homotopy_type_score(returns, macro_df, hit_levels=3, univalence_threshold=0.5):
    """
    Compute per-ETF homotopy type score.
    Higher score = more structure / transportability.
    """
    if len(returns) < 15 or macro_df is None or len(macro_df) < 15:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 15:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)
    current_macro = macro_factor[-1]
    # 1. Path equivalence
    path_eq = homotopy_path_equivalence(returns, current_macro)
    # 2. Higher inductive types
    hit_structure = higher_inductive_types(returns, current_macro, hit_levels)
    # 3. Univalence axiom
    transport = univalence_axiom(returns, current_macro, univalence_threshold)
    # Combine scores
    score = path_eq * 0.3 + hit_structure * 0.4 + transport * 0.3
    return float(score)
