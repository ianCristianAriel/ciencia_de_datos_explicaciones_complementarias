def regression_report(y_true, y_pred, n_features=None):
    """
    Prints a regression metrics summary report.
    
    Handles NaNs automatically and computes RMSLE only on non-negative values.
    """
    import numpy as np
    from sklearn.metrics import (
        mean_absolute_error,
        median_absolute_error,
        max_error,
        mean_squared_error,
        r2_score,
        explained_variance_score,
        mean_absolute_percentage_error
    )

    # ==========================
    # Filtrar NaNs para todas las métricas
    # ==========================
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    n = len(y_true_clean)

    # ==========================
    # Métricas de error absoluto
    # ==========================
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    medae = median_absolute_error(y_true_clean, y_pred_clean)
    max_err = max_error(y_true_clean, y_pred_clean)

    # MASE: normalizado por la media del cambio absoluto
    if n > 1:
        mase = mae / np.mean(np.abs(np.diff(y_true_clean)))
    else:
        mase = np.nan

    # ==========================
    # Métricas de error cuadrático
    # ==========================
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)

    # RMSLE solo para valores no negativos
    mask_rmsle = (y_true_clean >= 0) & (y_pred_clean >= 0)
    if np.any(mask_rmsle):
        rmsle = np.sqrt(mean_squared_error(
            np.log1p(y_true_clean[mask_rmsle]),
            np.log1p(y_pred_clean[mask_rmsle])
        ))
    else:
        rmsle = np.nan

    # ==========================
    # Métricas de error porcentual
    # ==========================
    try:
        mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
    except:
        mape = np.nan

    smape = np.mean(
        2 * np.abs(y_pred_clean - y_true_clean) /
        (np.abs(y_true_clean) + np.abs(y_pred_clean) + 1e-8)  # evitar división por 0
    )

    wape = np.sum(np.abs(y_true_clean - y_pred_clean)) / (np.sum(np.abs(y_true_clean)) + 1e-8)

    # ==========================
    # Goodness of fit
    # ==========================
    r2 = r2_score(y_true_clean, y_pred_clean)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n_features else None
    explained_var = explained_variance_score(y_true_clean, y_pred_clean)

    # ==========================
    # Impresión del reporte
    # ==========================
    print("=" * 60)
    print(" " * 20 + "REGRESSION REPORT")
    print("=" * 60)
    print(f"\nObservations: {n}")
    if n_features:
        print(f"Features:     {n_features}")
    print("-" * 60)

    print("\nERROR METRICS (Absolute Scale)")
    print(f"MAE:                     {mae:.4f}")
    print(f"Median Absolute Error:   {medae:.4f}")
    print(f"Max Error:               {max_err:.4f}")
    print(f"MASE:                    {mase:.4f}")

    print("\nSQUARED ERROR METRICS")
    print(f"MSE:                     {mse:.4f}")
    print(f"RMSE:                    {rmse:.4f}")
    print(f"RMSLE:                   {rmsle:.4f}")

    print("\nPERCENTAGE ERROR METRICS")
    print(f"MAPE:                    {mape:.4%}")
    print(f"SMAPE:                   {smape:.4%}")
    print(f"WAPE:                    {wape:.4%}")

    print("\nGOODNESS OF FIT")
    print(f"R²:                      {r2:.4f}")
    if adj_r2:
        print(f"Adjusted R²:             {adj_r2:.4f}")
    print(f"Explained Variance:       {explained_var:.4f}")
    print("=" * 60)