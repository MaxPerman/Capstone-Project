import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import savgol_filter

from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

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


# Helper Functions for Preprocessing
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
        avg = (arr[i-1] + arr[i+1]) / 2.0
        if abs(arr[i] - avg) > neighbor_diff:
            arr[i] = np.nan
    return pd.Series(arr, index=series.index).interpolate(method="linear", limit_direction="both")

def remove_outliers_rolling_zscore(series, window=5, zscore_threshold=3.0):
    if series is None or len(series) < window:
        return series
    s = series.copy()
    roll_mean = s.rolling(window=window, center=True, min_periods=1).mean()
    roll_std = s.rolling(window=window, center=True, min_periods=1).std()
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
    idx = range(start, end+1)
    return series.reindex(idx).interpolate(method="linear")


# Data Loading

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


# Accuracy Metrics 

def compute_accuracy_metrics(actual, predicted):
    error = actual - predicted
    rmse = np.sqrt(np.mean(np.square(error)))
    mae = np.mean(np.abs(error))
    return rmse, mae

def compute_accuracy_metrics_traditional(actual, predicted):
    error = actual - predicted
    rmse = np.sqrt(np.mean(np.square(error)))
    mae = np.mean(np.abs(error))
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2

def print_accuracy_metrics(name, actual, predicted):
    rmse, mae = compute_accuracy_metrics(actual, predicted)
    print(f"[ACCURACY] {name}: RMSE = {rmse:.8f}, MAE = {mae:.8f}")


# Traditional Forecasting Functions

# NCA: Stretched Exponential Model.
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
    cycles = np.arange(start_cycle, end_cycle+1)
    x_fore = cycles - start_cycle
    fvals = stretched_exp_model(x_fore, tau, beta, floor_val)
    fvals = np.minimum.accumulate(fvals)
    return pd.DataFrame({"cycle_number": cycles, "theoretical_capacity": fvals})

# NCM: Physically Based Model.
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
    cycles = np.arange(start_cycle, end_cycle+1).astype(float)
    x_fore = cycles - start_cycle
    fvals = ncm_degradation_model(x_fore, floor_val, alpha_val, adjusted_beta)
    return pd.DataFrame({"cycle_number": cycles, "theoretical_capacity": fvals})

# NCM+NCA: Logistic Model.
def combined_degradation_model(x, floor, tau, beta):
    return floor + (1 - floor) / (1 + (x / tau)**beta)

def traditional_forecast_ncm_nca(norm_series, end_cycle=5000):
    start_cycle = int(norm_series.index[0])
    x_data = np.array(norm_series.index - start_cycle)
    p0 = [0.2, np.median(x_data), 2.0]
    bnds = ([0.00, 1, 1.0], [0.5, 1e5, 5.0])
    try:
        popt, _ = curve_fit(combined_degradation_model, x_data, norm_series.values,
                            p0=p0, bounds=bnds, maxfev=200000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    except Exception as e:
        print("[WARN] Combined model fit (traditional) for NCM+NCA failed:", e)
        return None
    floor_val, tau_val, beta_val = popt
    cycles = np.arange(start_cycle, end_cycle+1)
    x_fore = cycles - start_cycle
    fvals = combined_degradation_model(x_fore, floor_val, tau_val, beta_val)
    return pd.DataFrame({"cycle_number": cycles, "theoretical_capacity": fvals})


# Traditional Analysis Plot Function

def plot_traditional_capacity_prediction(chem, actual_cap, trad_df, folder):
    plt.figure(figsize=(8,6))
    plt.plot(actual_cap.index, actual_cap.values, label="Actual (Normalized)", linewidth=2)
    plt.plot(trad_df["cycle_number"], trad_df["theoretical_capacity"],
             label="Traditional Forecast", linestyle="-", linewidth=2)
    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    # Title removed
    plt.legend()
    out_png = os.path.join(folder, f"Traditional_Capacity_Prediction_{chem}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    out_csv = os.path.join(folder, f"Traditional_Forecast_Adjusted_{chem}.csv")
    trad_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved {chem} traditional forecast plot to {out_png}")
    print(f"[INFO] Saved traditional forecast CSV to {out_csv}")
    return trad_df


# Combined Plotting Functions

def plot_combined_forecast(*, nca_fc, nca_actual, ncm_fc, ncm_actual, ncmnca_fc, ncmnca_actual, folder):
    plt.figure(figsize=(9,6))
    plt.plot(nca_actual.index, nca_actual.values, label="NCA Actual", linestyle="-", linewidth=2, color="red")
    plt.plot(nca_fc["cycle_number"], nca_fc["theoretical_capacity"], label="NCA Traditional", linestyle=":", linewidth=2, color="red")
    
    plt.plot(ncm_actual.index, ncm_actual.values, label="NCM Actual", linestyle="-", linewidth=2, color="blue")
    plt.plot(ncm_fc["cycle_number"], ncm_fc["theoretical_capacity"], label="NCM Traditional", linestyle=":", linewidth=2, color="blue")
    
    plt.plot(ncmnca_actual.index, ncmnca_actual.values, label="NCM+NCA Actual", linestyle="-", linewidth=2, color="green")
    plt.plot(ncmnca_fc["cycle_number"], ncmnca_fc["theoretical_capacity"], label="NCM+NCA Traditional", linestyle=":", linewidth=2, color="green")
    
    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    # Title removed
    plt.legend()
    combined_png = os.path.join(folder, "Traditional_Capacity_Prediction_Combined.png")
    plt.tight_layout()
    plt.savefig(combined_png, dpi=300)
    plt.close()
    print(f"[INFO] Saved combined traditional forecast plot to {combined_png}")

def plot_comparison_normalized_capacity(ncap_dict, folder):
    plt.figure(figsize=(8,6))
    for chem, norm_cap in ncap_dict.items():
        plt.plot(norm_cap.index, norm_cap.values, label=f"{chem} Actual", linewidth=2)
    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    # Title removed
    plt.legend()
    out_png = os.path.join(folder, "Comparison_Normalized_Capacity.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[INFO] Saved normalized capacity comparison plot to {out_png}")

def plot_pca_features(features_df, folder):
    features_clean = features_df.dropna(subset=["normalized_capacity"])
    feat_cols = ["normalized_capacity"]
    if len(feat_cols) == 1:
        plt.figure(figsize=(8,6))
        for chem in features_clean["chemistry"].unique():
            chem_data = features_clean[features_clean["chemistry"] == chem]
            plt.scatter(chem_data["cycle_number"], chem_data["normalized_capacity"], label=chem, s=40)
        plt.xlabel("Cycle Number")
        plt.ylabel("Normalized Capacity")
        # Title removed (Normalized Capacity by Chemistry)
        plt.legend()
        out_png = os.path.join(folder, "PCA_Features.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[INFO] Only one feature available; plotted normalized capacity vs cycle number to {out_png}")
        return
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_clean[feat_cols])
    pc_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"], index=features_clean.index)
    pc_df["chemistry"] = features_clean["chemistry"]
    plt.figure(figsize=(8,6))
    for chem in pc_df["chemistry"].unique():
        mask = pc_df["chemistry"] == chem
        plt.scatter(pc_df.loc[mask, "PC1"], pc_df.loc[mask, "PC2"], label=chem, s=40)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    # Title removed (PCA on Normalized Capacity)
    plt.legend()
    out_png = os.path.join(folder, "PCA_Features.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[INFO] Saved PCA features plot to {out_png}")


# FOLDER SETUP

HOME_DIR = os.path.expanduser("~")
BASE_BATTERY_FOLDER = os.path.join(HOME_DIR, "Desktop", "Python", "Battery")
CHEM_FOLDERS = {
    "NCA": os.path.join(BASE_BATTERY_FOLDER, "NCA"),
    "NCM": os.path.join(BASE_BATTERY_FOLDER, "NCM"),
    "NCM+NCA": os.path.join(BASE_BATTERY_FOLDER, "NCM+NCA")
}
OUTPUT_FOLDER = os.path.join(HOME_DIR, "Desktop", "Battery_Plots")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# MAIN FUNCTION

def main():
    combined_traditional = {}  # Traditional forecasts (theoretical model only)
    features_list = {}
    results_rows = []  # Summary metrics for each model

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

        if not series_list:
            print(f"[ERROR] No capacity data for {chem} after filtering.")
            continue

        combined_cap = np.mean(pd.concat(series_list, axis=1), axis=1)

        # Plot actual capacity
        plt.figure(figsize=(8,6))
        plt.plot(combined_cap.index, combined_cap.values, label="Actual (Normalized)", linewidth=2)
        plt.xlabel("Cycle Number")
        plt.ylabel("Normalized Capacity")
        # Title removed for actual capacity
        plt.legend()
        act_out = os.path.join(OUTPUT_FOLDER, f"Actual_{chem}_Capacity.png")
        plt.savefig(act_out, dpi=300)
        plt.close()
        print(f"[INFO] Saved actual capacity plot for {chem} to {act_out}")

        # Generate traditional forecasts for each chemistry
        if chem == "NCA":
            trad_df = theoretical_forecast_nca(combined_cap, end_cycle=5000)
            eqn = "Q(n) = floor + (1-floor)*exp( - (n/tau)^beta )"
            params = fit_nca_model(combined_cap)
            fitted = f"tau={params[0]:.8f}, beta={params[1]:.8f}, floor={params[2]:.8f}" if params is not None else ""
            comment = "Traditional model based solely on the stretched exponential equation."
        elif chem == "NCM":
            trad_df = theoretical_forecast_ncm_physicalbased(combined_cap, end_cycle=5000)
            eqn = "Q(n) = floor + (1-floor)*exp( - alpha * n^beta )"
            params = fit_ncm_physbased(combined_cap)
            if params is not None:
                fitted = f"floor={params[0]:.8f}, alpha={params[1]:.8f}, beta={params[2]:.8f}"
            else:
                fitted = ""
            comment = "Traditional model using the physically based exponential decay equation."
        elif chem == "NCM+NCA":
            trad_df = traditional_forecast_ncm_nca(combined_cap, end_cycle=5000)
            eqn = "Q(n) = floor + (1-floor) / (1 + (n/tau)^beta)"
            start_cycle = int(combined_cap.index[0])
            x_data = np.array(combined_cap.index - start_cycle)
            try:
                popt, _ = curve_fit(combined_degradation_model, x_data, combined_cap.values,
                                    p0=[0.2, np.median(x_data), 2.0],
                                    bounds=([0.00, 1, 1.0], [0.5, 1e5, 5.0]),
                                    maxfev=200000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                fitted = f"floor={popt[0]:.8f}, tau={popt[1]:.8f}, beta={popt[2]:.8f}"
            except Exception as e:
                print("[WARN] Combined model fit (traditional) for NCM+NCA failed:", e)
                fitted = ""
            comment = "Traditional model using a logistic (sigmoid) degradation equation."
        else:
            print(f"[WARN] No traditional model defined for {chem}.")
            continue

        # Compute accuracy metrics for traditional forecast over overlapping period.
        last_actual_cycle = combined_cap.index[-1]
        trad_overlap = trad_df[trad_df["cycle_number"] <= last_actual_cycle]["theoretical_capacity"].values
        actual_overlap = combined_cap.loc[combined_cap.index <= last_actual_cycle].values

        trad_rmse, trad_mae, trad_r2 = compute_accuracy_metrics_traditional(actual_overlap, trad_overlap)

        results_rows.append({
            "Model": chem,
            "Equation": eqn,
            "Fitted Parameters": fitted,
            "Traditional RMSE": trad_rmse,
            "Traditional MAE": trad_mae,
            "Traditional R2": trad_r2,
            "Comments": comment
        })

        # Plot traditional forecast for this chemistry.
        if trad_df is None or trad_df.empty:
            print(f"[ERROR] Traditional forecast generation for {chem} failed.")
            continue
        trad_df = plot_traditional_capacity_prediction(chem, combined_cap, trad_df, OUTPUT_FOLDER)
        combined_traditional[chem] = trad_df
        features_list[chem] = combined_cap

    # Plot combined comparison of actual capacities.
    if features_list:
        plot_comparison_normalized_capacity(features_list, OUTPUT_FOLDER)

    # Plot combined forecast (traditional) if all available.
    if all(key in combined_traditional for key in ["NCA", "NCM", "NCM+NCA"]):
        plot_combined_forecast(
            nca_fc=combined_traditional["NCA"],
            nca_actual=features_list["NCA"],
            ncm_fc=combined_traditional["NCM"],
            ncm_actual=features_list["NCM"],
            ncmnca_fc=combined_traditional["NCM+NCA"],
            ncmnca_actual=features_list["NCM+NCA"],
            folder=OUTPUT_FOLDER
        )
    elif "NCA" in combined_traditional and "NCM" in combined_traditional:
        plot_combined_forecast(
            nca_fc=combined_traditional["NCA"],
            nca_actual=features_list["NCA"],
            ncm_fc=combined_traditional["NCM"],
            ncm_actual=features_list["NCM"],
            ncmnca_fc=combined_traditional["NCA"],  # Dummy substitution
            ncmnca_actual=features_list["NCA"],
            folder=OUTPUT_FOLDER
        )

    # Save extracted features and PCA plot.
    if features_list:
        all_feats = []
        for c, series in features_list.items():
            df_tmp = extract_features(c, {"normalized_capacity": series})
            all_feats.append(df_tmp)
        if all_feats:
            features_all = pd.concat(all_feats, ignore_index=True)
            features_csv = os.path.join(OUTPUT_FOLDER, "Extracted_Features.csv")
            features_all.to_csv(features_csv, index=False)
            print(f"[INFO] Saved extracted features to {features_csv}")
            plot_pca_features(features_all, OUTPUT_FOLDER)

    # Save merged traditional forecast CSV.
    if combined_traditional:
        merged = []
        for c in combined_traditional:
            temp = combined_traditional[c].copy()
            temp["chemistry"] = c
            merged.append(temp)
        df_merged = pd.concat(merged, ignore_index=True)
        out_csv = os.path.join(OUTPUT_FOLDER, "Traditional_Capacity_Forecasts.csv")
        df_merged.to_csv(out_csv, index=False)
        print(f"[INFO] Saved merged traditional forecast CSV to {out_csv}")

    # Save summary results as a CSV table for your report.
    results_df = pd.DataFrame(results_rows, columns=["Model", "Equation", "Fitted Parameters",
                                                       "Traditional RMSE", "Traditional MAE", "Traditional R2", "Comments"])
    results_csv = os.path.join(OUTPUT_FOLDER, "Model_Results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"[INFO] Saved model results summary to {results_csv}")

if __name__ == "__main__":
    main()