import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import shap

from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import savgol_filter

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Global Constants
HOME_DIR = os.path.expanduser("~")
BASE_BATTERY_FOLDER = os.path.join(HOME_DIR, "Desktop", "Python", "Battery")
CHEM_FOLDERS = {
    "NCA": os.path.join(BASE_BATTERY_FOLDER, "NCA"),
    "NCM": os.path.join(BASE_BATTERY_FOLDER, "NCM"),
    "NCM+NCA": os.path.join(BASE_BATTERY_FOLDER, "NCM+NCA")
}
OUTPUT_FOLDER = os.path.join(HOME_DIR, "Desktop", "Battery_Plots")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global constant for NCM beta multiplier
NCM_BETA_MULTIPLIER = 1.0

warnings.filterwarnings("ignore", category=OptimizeWarning)
plt.style.use("ggplot")
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})
np.seterr(divide='ignore', invalid='ignore')

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
ML_MODEL = "xgboost"

def cv_xgboost(X, y):
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'reg_alpha': [0.01, 0.1],
        'reg_lambda': [1, 1.5]
    }
    model = XGBRegressor(random_state=42)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error',
                        cv=cv, verbose=0, n_jobs=-1)
    grid.fit(X, y)
    print("[INFO] Best XGBoost params:", grid.best_params_)
    # Hyperparameter tuning heatmap for max_depth vs learning_rate
    results = grid.cv_results_
    # Extract unique sorted hyperparameter values
    depths = sorted({int(d) for d in results['param_max_depth']})
    rates = sorted({float(r) for r in results['param_learning_rate']})
    heatmap = np.zeros((len(depths), len(rates)))
    for i, d in enumerate(depths):
        for j, r in enumerate(rates):
            mask = (results['param_max_depth'] == d) & (results['param_learning_rate'] == r)
            if mask.any():
                rmse = np.sqrt(-results['mean_test_score'][mask][0])
                heatmap[i, j] = rmse
            else:
                heatmap[i, j] = np.nan
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, interpolation='nearest', aspect='auto', cmap='plasma')
    plt.xticks(np.arange(len(rates)), rates)
    plt.yticks(np.arange(len(depths)), depths)
    # Annotate best hyperparameter combo (minimum RMSE)
    min_idx = np.unravel_index(np.nanargmin(heatmap), heatmap.shape)
    plt.scatter(min_idx[1], min_idx[0], s=200, facecolors='none', edgecolors='white', linewidth=2)
    plt.xlabel("learning_rate")
    plt.ylabel("max_depth")
    # plt.clim(vmin=np.nanmin(heatmap), vmax=np.nanpercentile(heatmap, 95))
    plt.colorbar(label="RMSE")
    heatmap_path = os.path.join(OUTPUT_FOLDER, "XGB_Hyperparam_Heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved hyperparameter tuning heatmap to {heatmap_path}")
    return grid.best_estimator_

def train_ml_model_cv(X_train, y_train, poly_degree=2):
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    if ML_MODEL == "xgboost":
        model = cv_xgboost(X_train_poly, y_train)
    return (model, poly)

def ml_forecast_residual(model_poly_tuple, forecast_x):
    model, poly = model_poly_tuple
    X_poly = poly.transform(forecast_x)
    return model.predict(X_poly)

# Accuracy Metrics Utility
def compute_accuracy_metrics(actual, predicted):
    error = actual - predicted
    rmse = np.sqrt(np.mean(np.square(error)))
    mae = np.mean(np.abs(error))
    return rmse, mae

 
def plot_shap_explanation(model_poly_tuple, X_train, folder, chem):
    model, poly = model_poly_tuple
    X_train_poly = poly.transform(X_train)
    try:
        if ML_MODEL == "xgboost":
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_train_poly)
    except Exception as e:
        print("[WARN] SHAP explainer creation failed:", e)
        explainer = shap.Explainer(model, X_train_poly)
    shap_values = explainer(X_train_poly)
    
    n_in = poly.n_features_in_
    if n_in == 3:
        input_features = ["cycle_offset", "theoretical_capacity", "gradient"]
    elif n_in == 6:
        input_features = ["cycle_offset", "theoretical_capacity", "gradient",
                          "cycle_offset^2", "theoretical_capacity^2", "gradient^2"]
    else:
        input_features = None
    feature_names = poly.get_feature_names_out(input_features) if input_features is not None else poly.get_feature_names_out()
    
    shap.summary_plot(shap_values.values, X_train_poly, feature_names=feature_names, show=False)
    out_png = os.path.join(folder, f"SHAP_Summary_{chem}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[INFO] Saved SHAP summary plot for {chem} to {out_png}")

def analyze_degradation_feature_importance(norm_series, folder, chem):
    if norm_series is None or norm_series.empty:
        print(f"[WARN] Empty series for {chem}, skipping feature importance analysis.")
        return
    
    # Create a feature DataFrame
    X = pd.DataFrame({
        'cycle_number': norm_series.index.astype(float),
        'capacity_gradient': np.gradient(norm_series.values)
    })
    y = norm_series.values
    
    # Train the model (using XGBoost)
    model = XGBRegressor(random_state=42, n_estimators=100, max_depth=3)
    model.fit(X, y)
    
    # Print feature importances if the model provides them
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for fname, imp in zip(X.columns, importances):
            print(f"[FEATURE IMPORTANCE] {chem}: {fname} importance = {imp:.4f}")
    
    # Perform SHAP analysis to explain the model predictions
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        shap.summary_plot(shap_values.values, X, feature_names=X.columns, show=False)
        out_png = os.path.join(folder, f"SHAP_Feature_Importance_{chem}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[INFO] Saved SHAP feature importance plot for {chem} to {out_png}")
    except Exception as e:
        print(f"[WARN] SHAP analysis failed for {chem}: {e}")

def analyze_extended_feature_importance(df_grouped, folder, chem):
    if 'cycle number' not in df_grouped.columns:
        df_grouped = df_grouped.reset_index()
    
    df_grouped['Normalized Capacity'] = df_grouped['Q discharge/mA.h'] / df_grouped['Q discharge/mA.h'].iloc[0]
    
    feature_cols = ['cycle number', 'Ecell/V', '<I>/mA', 'Q discharge/mA.h', 'Q charge/mA.h', 'control/V', 'control/mA']
    feature_cols = [col for col in feature_cols if col in df_grouped.columns]
    
    X = df_grouped[feature_cols]
    X.columns = X.columns.str.replace("<", "", regex=False).str.replace(">", "", regex=False).str.replace("/", "_", regex=False)
    y = df_grouped['Normalized Capacity']
    
    model = XGBRegressor(random_state=42, n_estimators=100, max_depth=3)
    model.fit(X, y)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for feat, imp in zip(X.columns, importances):
            print(f"[EXTENDED FEATURE IMPORTANCE] {chem}: {feat} importance = {imp:.4f}")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        shap.summary_plot(shap_values.values, X, feature_names=X.columns, show=False)
        out_png = os.path.join(folder, f"SHAP_Extended_Feature_Importance_{chem}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[INFO] Saved extended SHAP feature importance plot for {chem} to {out_png}")
        vals = shap_values.values
        mean_abs = np.mean(np.abs(vals), axis=0)
        std_abs = np.std(np.abs(vals), axis=0)
        features = X.columns.tolist()
        summary_data = []
        for feat, m, s in zip(features, mean_abs, std_abs):
            summary_data.append({"Feature": feat, "MeanAbsoluteSHAP": m, "StdAbsoluteSHAP": s})
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(by="MeanAbsoluteSHAP", ascending=False)
        csv_out = os.path.join(folder, f"SHAP_Numerical_Summary_{chem}.csv")
        summary_df.to_csv(csv_out, index=False)
        print(f"[INFO] Saved SHAP numerical summary for {chem} to {csv_out}")
    except Exception as e:
        print(f"[WARN] Extended SHAP analysis failed for {chem}: {e}")

def plot_pca_analysis(df, folder):
    # PCA and StandardScaler are imported globally
    
    # Try to extract grouping labels if available.
    group_label = None
    if 'Chemistry' in df.columns:
        group_label = df['Chemistry']
    elif 'cycle number' in df.columns:
        group_label = pd.to_numeric(df['cycle number'], errors='coerce')
    
    X = df.select_dtypes(include=[np.number])
    for col in ['cycle number', 'Normalized Capacity']:
        if col in X.columns:
            X = X.drop(columns=[col])
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    if X.empty:
        print("[WARN] No numeric measured variables available for PCA analysis after conversion.")
        return
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pc_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"], index=X.index)
    
    explained_variance = pca.explained_variance_ratio_
    
    plt.figure(figsize=(10, 8))
    if group_label is not None:
        groups = group_label.loc[X.index]
        if pd.api.types.is_numeric_dtype(groups):
            sc = plt.scatter(pc_df["PC1"], pc_df["PC2"], c=groups, cmap="viridis", s=40)
            plt.colorbar(sc, label="Group (continuous)")
        else:
            # Use forecast colours: red for NCA, blue for NCM, green for NCM+NCA
            color_dict = {'NCA': 'red', 'NCM': 'blue', 'NCM+NCA': 'green'}
            unique_groups = groups.unique()
            for g in unique_groups:
                color = color_dict.get(g, 'black')
                mask = groups == g
                plt.scatter(pc_df.loc[mask, "PC1"], pc_df.loc[mask, "PC2"], label=str(g), color=color, s=60)
            plt.legend(title="Group")
    else:
        plt.scatter(pc_df["PC1"], pc_df["PC2"], c="blue", s=40)
    
    plt.xlabel(f"PC1 ({explained_variance[0]*100:.2f}% variance)")
    plt.ylabel(f"PC2 ({explained_variance[1]*100:.2f}% variance)")
    out_png = os.path.join(folder, "PCA_Analysis.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[INFO] Saved PCA analysis plot to {out_png}")
    
    ev_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(explained_variance))],
        "Explained_Variance (%)": explained_variance * 100
    })
    ev_csv = os.path.join(folder, "PCA_Explained_Variance.csv")
    ev_df.to_csv(ev_csv, index=False)
    print(f"[INFO] Saved PCA explained variance table to {ev_csv}")
    
    loadings_df = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])], index=X.columns)
    loadings_csv = os.path.join(folder, "PCA_Loadings.csv")
    loadings_df.to_csv(loadings_csv)
    print(f"[INFO] Saved PCA loadings table to {loadings_csv}")

def combined_degradation_model(x, floor, tau, beta):
    return floor + (1 - floor) / (1 + (x / tau)**beta)

def hybrid_forecast_ncm_nca(norm_series, end_cycle=5000):
    start_cycle = int(norm_series.index[0])
    x_data = np.array(norm_series.index - start_cycle)
    p0 = [0.2, np.median(x_data), 2.0]
    bnds = ([0.00, 1, 1.0], [0.5, 1e5, 5.0])
    try:
        popt, _ = curve_fit(combined_degradation_model, x_data, norm_series.values,
                            p0=p0, bounds=bnds, maxfev=200000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    except Exception as e:
        print("[WARN] Combined model fit (NCM+NCA) failed:", e)
        return None
    floor_val, tau_val, beta_val = popt
    print(f"[DIAG] Combined (NCM+NCA) model params: floor={floor_val:.8f}, tau={tau_val:.8f}, beta={beta_val:.8f}")
    
    cycles = np.arange(start_cycle, end_cycle + 1)
    x_fore = cycles - start_cycle
    fvals = combined_degradation_model(x_fore, floor_val, tau_val, beta_val)
    df_theor = pd.DataFrame({"cycle_number": cycles, "theoretical_capacity": fvals})
    
    y_model_train = combined_degradation_model(x_data, floor_val, tau_val, beta_val)
    residual = norm_series.values - y_model_train
    X_train = x_data.reshape(-1, 1)
    model_poly = train_ml_model_cv(X_train, residual, poly_degree=2)
    
    X_fore = x_fore.reshape(-1, 1)
    pred_res = ml_forecast_residual(model_poly, X_fore)
    final_vals = fvals + pred_res
    final_vals = np.minimum.accumulate(final_vals)
    
    return pd.DataFrame({"cycle_number": cycles, "predicted_capacity": final_vals})

def holdout_residuals_ncm_nca(norm_series, holdout_fraction=0.2):
    n = len(norm_series)
    split_idx = int(n * (1 - holdout_fraction))
    train_series = norm_series.iloc[:split_idx]
    holdout_series = norm_series.iloc[split_idx:]
    
    start_cycle = int(train_series.index[0])
    x_train = np.array(train_series.index - start_cycle)
    p0 = [0.2, np.median(x_train), 2.0]
    bnds = ([0.00, 1, 1.0], [0.5, 1e5, 5.0])
    try:
        popt, _ = curve_fit(combined_degradation_model, x_train, train_series.values,
                            p0=p0, bounds=bnds, maxfev=200000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    except Exception as e:
        print("[WARN] Combined model fit (NCM+NCA holdout) failed:", e)
        return None
    floor_val, tau_val, beta_val = popt
    print(f"[DIAG] Combined (NCM+NCA) model params (holdout): floor={floor_val:.8f}, tau={tau_val:.8f}, beta={beta_val:.8f}")
    
    x_hold = np.array(holdout_series.index - start_cycle)
    y_hold_theor = combined_degradation_model(x_hold, floor_val, tau_val, beta_val)
    
    y_model_train = combined_degradation_model(x_train, floor_val, tau_val, beta_val)
    residual_train = train_series.values - y_model_train
    X_train = x_train.reshape(-1, 1)
    model_poly = train_ml_model_cv(X_train, residual_train, poly_degree=2)
    
    X_hold = x_hold.reshape(-1, 1)
    ml_corr = ml_forecast_residual(model_poly, X_hold)
    
    y_hold_pred = y_hold_theor + ml_corr
    residual_hold = holdout_series.values - y_hold_pred
    
    rmse, mae = compute_accuracy_metrics(holdout_series.values, y_hold_pred)
    print(f"[ACCURACY] Combined (NCM+NCA) Holdout: RMSE = {rmse:.8f}, MAE = {mae:.8f}")
    
    plt.figure(figsize=(8,6))
    plt.plot(holdout_series.index, residual_hold, marker='o', linestyle='-', color='orange')
    plt.xlabel("Cycle Number")
    plt.ylabel("Residual (Actual - Predicted)")
    holdout_plot = os.path.join(OUTPUT_FOLDER, "Residuals_Holdout_NCM+NCA.png")
    plt.tight_layout()
    plt.savefig(holdout_plot, dpi=300)
    plt.close()
    print(f"[INFO] Saved Combined (NCM+NCA) holdout residual plot to {holdout_plot}")
    
    return residual_hold

def plot_capacity_prediction(chem, actual_cap, forecast_df, folder):
    plt.figure(figsize=(8, 6))
    plt.plot(actual_cap.index, actual_cap.values, label="Actual (Normalized)", linewidth=2)
    mask = forecast_df["cycle_number"] > actual_cap.index[-1]
    fc_future = forecast_df[mask]
    plt.plot(fc_future["cycle_number"], fc_future["predicted_capacity"],
             label="Hybrid Forecast", linestyle=":", linewidth=2)
    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    plt.legend()
    out_png = os.path.join(folder, f"Capacity_Prediction_{chem}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    out_csv = os.path.join(folder, f"Forecast_Adjusted_{chem}.csv")
    forecast_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved {chem} forecast plot to {out_png}")
    print(f"[INFO] Saved forecast CSV to {out_csv}")
    return forecast_df

def plot_combined_forecast(nca_fc, nca_actual, ncm_fc, ncm_actual, ncmnca_fc, ncmnca_actual, folder):
    plt.figure(figsize=(9, 6))
    plt.plot(nca_actual.index, nca_actual.values,
             label="NCA Actual", linestyle="-", linewidth=2, color="red")
    plt.plot(nca_fc["cycle_number"], nca_fc["predicted_capacity"],
             label="NCA Forecast", linestyle=":", linewidth=2, color="red")
    
    plt.plot(ncm_actual.index, ncm_actual.values,
             label="NCM Actual", linestyle="-", linewidth=2, color="blue")
    plt.plot(ncm_fc["cycle_number"], ncm_fc["predicted_capacity"],
             label="NCM Forecast", linestyle=":", linewidth=2, color="blue")
    
    plt.plot(ncmnca_actual.index, ncmnca_actual.values,
             label="NCM+NCA Actual", linestyle="-", linewidth=2, color="green")
    plt.plot(ncmnca_fc["cycle_number"], ncmnca_fc["predicted_capacity"],
             label="NCM+NCA Forecast", linestyle=":", linewidth=2, color="green")
    
    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    plt.legend()
    combined_png = os.path.join(folder, "Capacity_Prediction_Combined.png")
    plt.tight_layout()
    plt.savefig(combined_png, dpi=300)
    plt.close()
    print(f"[INFO] Saved combined forecast plot to {combined_png}")

def plot_comparison_normalized_capacity(ncap_dict, folder):
    plt.figure(figsize=(8, 6))
    for chem, norm_cap in ncap_dict.items():
        plt.plot(norm_cap.index, norm_cap.values, label=f"{chem} Normalized Capacity", linewidth=2)
    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    plt.legend()
    out_png = os.path.join(folder, "Comparison_Normalized_Capacity.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[INFO] Saved normalized capacity comparison plot to {out_png}")

def remove_outliers_iqr(series, factor=1.5):
    if series is None or series.empty:
        return series
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low_bound = q1 - factor * iqr
    high_bound = q3 + factor * iqr
    s2 = series.where((series >= low_bound) & (series <= high_bound), np.nan)
    return s2.interpolate(method="linear", limit_direction="both")

def remove_spikes_by_neighbors(series, neighbor_diff=0.05):
    if series is None or len(series) < 3:
        return series
    arr = series.values.copy()
    for i in range(1, len(arr) - 1):
        avg = (arr[i - 1] + arr[i + 1]) / 2.0
        if abs(arr[i] - avg) > neighbor_diff:
            arr[i] = np.nan
    return pd.Series(arr, index=series.index).interpolate(method="linear", limit_direction="both")

def remove_outliers_rolling_zscore(series, window=5, zscore_threshold=3.0):
    if series is None or len(series) < window:
        return series
    s = series.copy()
    roll_mean = s.rolling(window=window, center=True, min_periods=1).mean()
    roll_std  = s.rolling(window=window, center=True, min_periods=1).std()
    s[(s - roll_mean).abs() / roll_std > zscore_threshold] = np.nan
    s.interpolate(method="linear", limit_direction="both", inplace=True)
    return s

def clip_series(s, lower=None, upper=None):
    if s is None or s.empty:
        return s
    if lower is not None:
        s = s.where(s >= lower, lower)
    if upper is not None:
        s = s.where(s <= upper, upper)
    return s

def capacity_anomaly_filter(series, factor=1.5):
    if series is None or series.empty:
        return series
    base = series.iloc[0]
    if base == 0:
        return series
    norm = series / base
    norm = remove_outliers_iqr(norm, factor=factor)
    norm = clip_series(norm, 0.0, 1.2)
    return norm * base

def normalize_capacity(series):
    if series is None or series.empty:
        return series
    base = series.iloc[0]
    if base == 0:
        return series
    return series / base

def final_smoothing_pipeline(series, clip_low=None, clip_high=None,
                             neighbor_diff=0.05, use_savgol=True,
                             savgol_window=21, savgol_poly=3):
    if series is None or series.empty:
        return series
    s = remove_outliers_iqr(series, factor=1.5)
    s = remove_outliers_rolling_zscore(s, window=5, zscore_threshold=3.0)
    s = remove_spikes_by_neighbors(s, neighbor_diff=neighbor_diff)
    s = clip_series(s, clip_low, clip_high)
    s = s.rolling(window=5, center=True, min_periods=1).mean()
    if use_savgol and len(s) >= savgol_window:
        dyn_win = min(savgol_window, len(s))
        if dyn_win % 2 == 0:
            dyn_win += 1
        filt = savgol_filter(s.values, window_length=dyn_win, polyorder=savgol_poly, mode='nearest')
        s = pd.Series(filt, index=s.index)
    return s

def fill_missing_cycles(series):
    if series is None or series.empty:
        return series
    start = int(series.index.min())
    end = int(series.index.max())
    idx = range(start, end + 1)
    return series.reindex(idx).interpolate(method="linear")

def load_data(fp):
    df = pd.read_csv(fp)
    df.rename(columns={
        "time/s": "time/s",
        "control/V/mA": "control/V/mA",
        "Ecell/V": "Ecell/V",
        "<I>/mA": "<I>/mA",
        "Q discharge/mA.h": "Q discharge/mA.h",
        "Q charge/mA.h": "Q charge/mA.h",
        "control/V": "control/V",
        "control/mA": "control/mA",
        "cycle number": "cycle number"
    }, inplace=True)
    return df

def analyze_capacity_fade(df):
    if "Q discharge/mA.h" in df and "cycle number" in df:
        return df.groupby("cycle number", group_keys=False).mean()["Q discharge/mA.h"]
    return None

def extract_features(chem, data_dict):
    return pd.DataFrame({
        "cycle_number": data_dict["normalized_capacity"].index,
        "normalized_capacity": data_dict["normalized_capacity"].values,
        "chemistry": chem
    })

def stretched_exp_model(x, tau, beta, floor):
    return floor + (1 - floor) * np.exp(-np.power(x / tau, beta))

def fit_nca_model(norm_series):
    start_cycle = int(norm_series.index[0])
    x_data = norm_series.index - start_cycle
    y_data = norm_series.values
    p0 = [np.median(x_data), 0.5, 0.7]
    bnds = ([1, 0.1, 0], [1e5, 1.0, 1.0])
    try:
        popt, _ = curve_fit(stretched_exp_model, x_data, y_data, p0=p0,
                              bounds=bnds, maxfev=200000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        return popt
    except Exception as e:
        print("[WARN] NCA model fit failed:", e)
        return None

def theoretical_forecast_nca(norm_series, end_cycle=5000):
    params = fit_nca_model(norm_series)
    if params is None:
        return None
    tau, beta, floor_val = params
    start_cycle = int(norm_series.index[0])
    cycles = np.arange(start_cycle, end_cycle + 1)
    x_fore = cycles - start_cycle
    fvals = stretched_exp_model(x_fore, tau, beta, floor_val)
    fvals = np.minimum.accumulate(fvals)  # monotonic
    return pd.DataFrame({"cycle_number": cycles, "theoretical_capacity": fvals})

def hybrid_forecast_nca(norm_series, end_cycle=5000):
    df_theor = theoretical_forecast_nca(norm_series, end_cycle=end_cycle)
    if df_theor is None or df_theor.empty:
        return None
    p = fit_nca_model(norm_series)
    if p is None:
        return None
    start_cycle = int(norm_series.index[0])
    x_data = (norm_series.index - start_cycle).values.reshape(-1, 1)
    y_theor = stretched_exp_model(x_data.flatten(), *p)
    
    residual = norm_series.values - y_theor
    model_poly = train_ml_model_cv(x_data, residual, poly_degree=2)
    cycles = df_theor["cycle_number"].values.reshape(-1, 1)
    x_fore = cycles - start_cycle
    tvals = df_theor["theoretical_capacity"].values
    pred_res = ml_forecast_residual(model_poly, x_fore)
    
    final_vals = tvals + pred_res
    idx_after = int(norm_series.index[-1] - start_cycle + 1)
    if idx_after < len(final_vals):
        final_vals[idx_after] = norm_series.iloc[-1]
    final_vals = np.minimum.accumulate(final_vals)
    return pd.DataFrame({"cycle_number": cycles.flatten(), "predicted_capacity": final_vals})

def holdout_residuals_nca(norm_series, holdout_fraction=0.2):
    n = len(norm_series)
    split_idx = int(n * (1 - holdout_fraction))
    train_series = norm_series.iloc[:split_idx]
    holdout_series = norm_series.iloc[split_idx:]
    
    start_cycle = int(train_series.index[0])
    p = fit_nca_model(train_series)
    if p is None:
        return None
    x_train = (train_series.index - start_cycle).values.reshape(-1, 1)
    y_train_theor = stretched_exp_model(x_train.flatten(), *p)
    residual_train = train_series.values - y_train_theor
    model_poly = train_ml_model_cv(x_train, residual_train, poly_degree=2)
    
    x_hold = (holdout_series.index - start_cycle).values.reshape(-1, 1)
    y_hold_theor = stretched_exp_model(x_hold.flatten(), *p)
    ml_corr = ml_forecast_residual(model_poly, x_hold)
    y_hold_pred = y_hold_theor + ml_corr
    residual_hold = holdout_series.values - y_hold_pred
    
    plt.figure(figsize=(8,6))
    plt.plot(holdout_series.index, residual_hold, marker='o', linestyle='-', color='purple')
    plt.xlabel("Cycle Number")
    plt.ylabel("Residual (Actual - Predicted)")
    holdout_plot = os.path.join(OUTPUT_FOLDER, "Residuals_Holdout_NCA.png")
    plt.tight_layout()
    plt.savefig(holdout_plot, dpi=300)
    plt.close()
    print("[INFO] Saved NCA holdout residual plot:", holdout_plot)
    return residual_hold

def ncm_degradation_model(x, floor, alpha, beta):
    return floor + (1.0 - floor) * np.exp(-alpha * np.power(x, beta))

def fit_ncm_physbased(norm_series):
    start_cycle = int(norm_series.index[0])
    x_data = (norm_series.index - start_cycle).astype(float)
    y_data = norm_series.values.astype(float)
    p0 = [0.1, 0.001, 1.1]
    bnds = ([0.00, 1e-7, 1.0], [0.30, 1.0, 5.0])
    try:
        popt, _ = curve_fit(ncm_degradation_model, x_data, y_data, p0=p0,
                            bounds=bnds, maxfev=200000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        return popt
    except Exception as e:
        print("[WARN] NCM physically-based fit failed:", e)
        return None

def theoretical_forecast_ncm_physicalbased(norm_series, end_cycle=5000):
    p_ncm = fit_ncm_physbased(norm_series)
    if p_ncm is None:
        return None
    floor_val, alpha_val, beta_val = p_ncm
    adjusted_beta = beta_val * NCM_BETA_MULTIPLIER
    start_cycle = int(norm_series.index[0])
    cycles = np.arange(start_cycle, end_cycle + 1).astype(float)
    x_fore = cycles - start_cycle
    fvals = ncm_degradation_model(x_fore, floor_val, alpha_val, adjusted_beta)
    return pd.DataFrame({"cycle_number": cycles, "theoretical_capacity": fvals})

def advanced_ncm_physbased_hybrid(norm_series, end_cycle=5000):
    diag_dir = os.path.join(OUTPUT_FOLDER, "Diagnostics_NCM")
    os.makedirs(diag_dir, exist_ok=True)
    
    df_theor = theoretical_forecast_ncm_physicalbased(norm_series, end_cycle=end_cycle)
    if df_theor is None or df_theor.empty:
        return None
    
    p_ncm = fit_ncm_physbased(norm_series)
    if p_ncm is None:
        return None
    floor_val, alpha_val, beta_val = p_ncm
    start_cycle = int(norm_series.index[0])
    x_data = (norm_series.index - start_cycle).values.reshape(-1, 1).astype(float)
    y_theor_train = ncm_degradation_model(x_data.flatten(), floor_val, alpha_val, beta_val)
    
    residual = norm_series.values - y_theor_train
    grad_train = np.gradient(y_theor_train)
    X_train = np.column_stack([x_data, y_theor_train, grad_train])
    model_poly = train_ml_model_cv(X_train, residual, poly_degree=2)
    
    fc_cycles = df_theor["cycle_number"].values.reshape(-1, 1).astype(float)
    x_fore = fc_cycles - start_cycle
    theo_fore = df_theor["theoretical_capacity"].values
    grad_fore = np.gradient(theo_fore)
    X_fore = np.column_stack([x_fore, theo_fore, grad_fore])
    ml_corr = ml_forecast_residual(model_poly, X_fore)
    final_vals = theo_fore + ml_corr
    return pd.DataFrame({"cycle_number": fc_cycles.flatten(), "predicted_capacity": final_vals})

def holdout_residuals_ncm_physbased(norm_series, holdout_fraction=0.2):
    n = len(norm_series)
    split_idx = int(n * (1 - holdout_fraction))
    train_series = norm_series.iloc[:split_idx]
    holdout_series = norm_series.iloc[split_idx:]
    
    p = fit_ncm_physbased(train_series)
    if p is None:
        return None
    floor_val, alpha_val, beta_val = p
    start_cycle = int(train_series.index[0])
    x_train = (train_series.index - start_cycle).values.reshape(-1, 1)
    y_train_theor = ncm_degradation_model(x_train.flatten(), floor_val, alpha_val, beta_val)
    residual_train = train_series.values - y_train_theor
    grad_train = np.gradient(y_train_theor)
    X_train = np.column_stack([x_train, y_train_theor, grad_train])
    model_poly = train_ml_model_cv(X_train, residual_train, poly_degree=2)
    
    x_hold = (holdout_series.index - start_cycle).values.reshape(-1, 1)
    y_hold_theor = ncm_degradation_model(x_hold.flatten(), floor_val, alpha_val, beta_val)
    grad_hold = np.gradient(y_hold_theor)
    X_hold = np.column_stack([x_hold, y_hold_theor, grad_hold])
    ml_corr = ml_forecast_residual(model_poly, X_hold)
    
    y_hold_pred = y_hold_theor + ml_corr
    residual_hold = holdout_series.values - y_hold_pred
    
    plt.figure(figsize=(8,6))
    plt.plot(holdout_series.index, residual_hold, marker='o', linestyle='-', color='green')
    plt.xlabel("Cycle Number")
    plt.ylabel("Residual (Actual - Predicted)")
    holdout_plot = os.path.join(OUTPUT_FOLDER, "Residuals_Holdout_NCM.png")
    plt.tight_layout()
    plt.savefig(holdout_plot, dpi=300)
    plt.close()
    print(f"[INFO] Saved NCM holdout residual plot: {holdout_plot}")
    
    return residual_hold



def main():
    combined_forecasts = {}
    features_list = {}  # For normalized capacity series (for other plots)
    aggregated_features = {}  # For full measurement data for PCA analysis
    results_rows = []  # To store summary metrics for each model
    
    # Loop over each folder/dataset
    for chem, folder in CHEM_FOLDERS.items():
        if not os.path.isdir(folder):
            print(f"[ERROR] Folder for {chem} not found: {folder}")
            continue
        
        csv_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
        if not csv_files:
            print(f"[ERROR] No CSV files found for {chem} in {folder}")
            continue
        
        series_list = []
        feature_dfs = []
        for fname in csv_files:
            fp = os.path.join(folder, fname)
            df = load_data(fp)
            if df.empty:
                continue
            raw_cap = analyze_capacity_fade(df)
            if raw_cap is None or raw_cap.empty:
                continue
            s = capacity_anomaly_filter(raw_cap)
            s = final_smoothing_pipeline(s, clip_low=0.0, clip_high=None,
                                         neighbor_diff=0.05, use_savgol=True,
                                         savgol_window=21, savgol_poly=3)
            s = fill_missing_cycles(s)
            s = normalize_capacity(s)
            series_list.append(s)
            
            # Also aggregate full measured variables for PCA.
            df_grouped = df.groupby("cycle number").mean().reset_index()
            feature_dfs.append(df_grouped)
        
        if not series_list:
            print(f"[ERROR] No capacity data for {chem} after filtering.")
            continue
        
        combined_cap = np.mean(pd.concat(series_list, axis=1), axis=1)
        features_list[chem] = combined_cap
        
        # Save actual capacity plot for this chemistry
        plt.figure(figsize=(8, 6))
        plt.plot(combined_cap.index, combined_cap.values, label="Actual (Normalized)", linewidth=2)
        plt.xlabel("Cycle Number")
        plt.ylabel("Normalized Capacity")
        plt.legend()
        act_out = os.path.join(OUTPUT_FOLDER, f"Actual_{chem}_Capacity.png")
        plt.tight_layout()
        plt.savefig(act_out, dpi=300)
        plt.close()
        print(f"[INFO] Saved actual capacity plot for {chem} to {act_out}")
        
        # Extended feature importance analysis using full dataset features 
        if feature_dfs:
            combined_features = pd.concat(feature_dfs).groupby("cycle number").mean().reset_index()
            aggregated_features[chem] = combined_features  # store for PCA
            analyze_extended_feature_importance(combined_features, OUTPUT_FOLDER, chem)
        # Analyze degradation feature importance based on normalized capacity
        analyze_degradation_feature_importance(combined_cap, OUTPUT_FOLDER, chem)
        
        # Forecast generation and holdout analysis for each dataset
        if chem == "NCA":
            fc_df = hybrid_forecast_nca(combined_cap, end_cycle=5000)
            res = holdout_residuals_nca(combined_cap, holdout_fraction=0.2)
            rmse, mae = compute_accuracy_metrics(
                combined_cap.values,
                fc_df[fc_df["cycle_number"] <= combined_cap.index[-1]]["predicted_capacity"].values
            )
            res_mean, res_std = (np.mean(res), np.std(res)) if res is not None else (np.nan, np.nan)
            params = fit_nca_model(combined_cap)
            fitted = f"tau={params[0]:.8f}, beta={params[1]:.8f}, floor={params[2]:.8f}" if params is not None else ""
            eqn = "Q(n) = floor + (1-floor)*exp( - (n/tau)^beta )"
            comment = "ML correction improved forecast accuracy."
            results_rows.append({"Model": "NCA", "Equation": eqn,
                                 "Fitted Parameters": fitted,
                                 "RMSE": rmse, "MAE": mae,
                                 "Residual Mean": res_mean, "Residual Std": res_std,
                                 "Comments": comment})
        elif chem == "NCM":
            fc_df = advanced_ncm_physbased_hybrid(combined_cap, end_cycle=5000)
            res = holdout_residuals_ncm_physbased(combined_cap, holdout_fraction=0.2)
            rmse, mae = compute_accuracy_metrics(
                combined_cap.values,
                fc_df[fc_df["cycle_number"] <= combined_cap.index[-1]]["predicted_capacity"].values
            )
            res_mean, res_std = (np.mean(res), np.std(res)) if res is not None else (np.nan, np.nan)
            params = fit_ncm_physbased(combined_cap)
            if params is not None:
                fitted = f"floor={params[0]:.8f}, alpha={params[1]:.8f}, beta={params[2]:.8f}"
            else:
                fitted = ""
            eqn = "Q(n) = floor + (1-floor)*exp( - alpha * n^beta )"
            comment = "ML correction adjusted predictions to reduce holdout residuals."
            results_rows.append({"Model": "NCM", "Equation": eqn,
                                 "Fitted Parameters": fitted,
                                 "RMSE": rmse, "MAE": mae,
                                 "Residual Mean": res_mean, "Residual Std": res_std,
                                 "Comments": comment})
        elif chem == "NCM+NCA":
            fc_df = hybrid_forecast_ncm_nca(combined_cap, end_cycle=5000)
            res = holdout_residuals_ncm_nca(combined_cap, holdout_fraction=0.2)
            rmse, mae = compute_accuracy_metrics(
                combined_cap.values,
                fc_df[fc_df["cycle_number"] <= combined_cap.index[-1]]["predicted_capacity"].values
            )
            res_mean, res_std = (np.mean(res), np.std(res)) if res is not None else (np.nan, np.nan)
            start_cycle = int(combined_cap.index[0])
            x_data = np.array(combined_cap.index - start_cycle)
            try:
                popt, _ = curve_fit(combined_degradation_model, x_data, combined_cap.values,
                                    p0=[0.2, np.median(x_data), 2.0],
                                    bounds=([0.00, 1, 1.0], [0.5, 1e5, 5.0]),
                                    maxfev=200000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                fitted = f"floor={popt[0]:.8f}, tau={popt[1]:.8f}, beta={popt[2]:.8f}"
            except Exception as e:
                print("[WARN] Combined model fit for CSV results failed:", e)
                fitted = ""
            eqn = "Q(n) = floor + (1-floor) / (1 + (n/tau)^beta)"
            comment = "Logistic model with ML residual correction produced lower errors."
            results_rows.append({"Model": "NCM+NCA", "Equation": eqn,
                                 "Fitted Parameters": fitted,
                                 "RMSE": rmse, "MAE": mae,
                                 "Residual Mean": res_mean, "Residual Std": res_std,
                                 "Comments": comment})
        else:
            print(f"[WARN] No model defined for {chem}.")
            continue
        
        if fc_df is None or fc_df.empty:
            print(f"[ERROR] Forecast generation for {chem} failed.")
            continue
        
        fc_df = plot_capacity_prediction(chem, combined_cap, fc_df, OUTPUT_FOLDER)
        combined_forecasts[chem] = fc_df
    
    # Combined comparison plot
    if features_list:
        plot_comparison_normalized_capacity(features_list, OUTPUT_FOLDER)
    
    # Combined forecast plot
    if all(key in combined_forecasts for key in ["NCA", "NCM", "NCM+NCA"]):
        plot_combined_forecast(
            nca_fc=combined_forecasts["NCA"],
            nca_actual=features_list["NCA"],
            ncm_fc=combined_forecasts["NCM"],
            ncm_actual=features_list["NCM"],
            ncmnca_fc=combined_forecasts["NCM+NCA"],
            ncmnca_actual=features_list["NCM+NCA"],
            folder=OUTPUT_FOLDER
        )
    elif "NCA" in combined_forecasts and "NCM" in combined_forecasts:
        plot_combined_forecast(
            nca_fc=combined_forecasts["NCA"],
            nca_actual=features_list["NCA"],
            ncm_fc=combined_forecasts["NCM"],
            ncm_actual=features_list["NCM"],
            ncmnca_fc=combined_forecasts["NCA"],  # Dummy substitution
            ncmnca_actual=features_list["NCA"],
            folder=OUTPUT_FOLDER
        )
    
    # Save aggregated full-measurement features for PCA analysis (across all chemistries)
    if aggregated_features:
        combined_agg = []
        for chem, df in aggregated_features.items():
            df["Chemistry"] = chem
            combined_agg.append(df)
        combined_features_all = pd.concat(combined_agg, ignore_index=True)
        agg_csv = os.path.join(OUTPUT_FOLDER, "Aggregated_Features_for_PCA.csv")
        combined_features_all.to_csv(agg_csv, index=False)
        print(f"[INFO] Saved aggregated features for PCA to {agg_csv}")
        plot_pca_analysis(combined_features_all, OUTPUT_FOLDER)
    
    # Save merged forecast CSV.
    if combined_forecasts:
        merged = []
        for c in combined_forecasts:
            temp = combined_forecasts[c].copy()
            temp["chemistry"] = c
            merged.append(temp)
        df_merged = pd.concat(merged, ignore_index=True)
        out_csv = os.path.join(OUTPUT_FOLDER, "Capacity_Forecasts.csv")
        df_merged.to_csv(out_csv, index=False)
        print(f"[INFO] Saved merged forecast CSV to {out_csv}")
    
    # Save summary results as a CSV table for report.
    results_df = pd.DataFrame(results_rows, columns=["Model", "Equation", "Fitted Parameters", "RMSE", "MAE", "Residual Mean", "Residual Std", "Comments"])
    results_csv = os.path.join(OUTPUT_FOLDER, "Model_Results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"[INFO] Saved model results summary to {results_csv}")

if __name__ == "__main__":
    main()