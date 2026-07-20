import os
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from model import load

# Must match the target_map ordering in model.py (Dropout=0, Enrolled=1, Graduate=2)
CLASS_NAMES = ['Dropout', 'Enrolled', 'Graduate']

# All generated figures are written here (regenerable, so kept out of git).
PLOTS_DIR = 'plots'


def load_artifacts():
    """Load the models and scaler produced by model.py."""
    logistic_model = joblib.load('logistic_model.pkl')
    scaler = joblib.load('scaler.pkl')
    decision_tree_model = joblib.load('decision_tree_model.pkl')
    return logistic_model, scaler, decision_tree_model


def save_confusion_heatmap(cm, title, out_file, values_format=None):
    """Render a single confusion matrix as a labelled heatmap and save it."""
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format=values_format)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap -> {out_file}")


def confusion_matrix_analysis(name, y_true, preds):
    """Print the classification report and save raw + normalized matrices."""
    print(f"\n=== {name}: Confusion Matrix Analysis ===")
    print(classification_report(y_true, preds, target_names=CLASS_NAMES))
    slug = name.lower().replace(' ', '_')

    # Raw counts: rows = true class, columns = predicted class.
    cm = confusion_matrix(y_true, preds)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(cm)
    save_confusion_heatmap(
        cm,
        f"{name} - Confusion Matrix",
        os.path.join(PLOTS_DIR, f"confusion_matrix_{slug}.png"),
    )

    # Normalized by true class: each row sums to 1, so the diagonal = per-class recall.
    cm_norm = confusion_matrix(y_true, preds, normalize='true')
    print("Normalized confusion matrix (row-wise, diagonal = recall):")
    print(cm_norm.round(2))
    save_confusion_heatmap(
        cm_norm,
        f"{name} - Normalized Confusion Matrix (recall)",
        os.path.join(PLOTS_DIR, f"confusion_matrix_{slug}_normalized.png"),
        values_format='.2f',
    )


def _as_class_list(shap_values):
    """Normalize SHAP output to a list of (n_samples, n_features) arrays, one per class.

    Different SHAP versions return either a list-per-class or a single
    (n_samples, n_features, n_classes) array, so we handle both.
    """
    if isinstance(shap_values, list):
        return shap_values
    if hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        return [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    return shap_values


def shap_summary(name, explainer, X, feature_names):
    """Compute SHAP values and save a global feature-importance bar plot."""
    print(f"\n=== {name}: SHAP Feature Importance ===")

    shap_values = _as_class_list(explainer.shap_values(X))

    fig = plt.figure()
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        plot_type='bar',
        show=False,
    )
    plt.title(f"{name} - SHAP Feature Importance")
    plt.tight_layout()

    slug = name.lower().replace(' ', '_')
    out_file = os.path.join(PLOTS_DIR, f"shap_importance_{slug}.png")
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved SHAP summary -> {out_file}")


def shap_analysis(logistic_model, decision_tree_model, scaler, X_train, X_test, X_test_scaled):
    """Global SHAP feature importance for both saved models."""
    feature_names = list(X_train.columns)

    # Linear model: explained on scaled features, with training data as the background.
    X_train_scaled = scaler.transform(X_train)
    lr_explainer = shap.LinearExplainer(logistic_model, X_train_scaled)
    shap_summary("Logistic Regression", lr_explainer,
                 X_test_scaled, feature_names)

    # Tree model: exact TreeExplainer on the raw (unscaled) features.
    dt_explainer = shap.TreeExplainer(decision_tree_model)
    shap_summary("Decision Tree", dt_explainer, X_test, feature_names)


def early_warning_analysis(X_train, X_test, y_train, y_test):
    """Train a Logistic Regression on first-semester-only features.

    Drops every '2nd sem' column so the model only sees information a school
    actually has partway through year one -- a true 'early warning' test.
    """
    print("\n########## EARLY-WARNING (1st-semester-only) ##########")

    early_cols = [c for c in X_train.columns if '2nd sem' not in c]
    dropped = [c for c in X_train.columns if '2nd sem' in c]
    print(
        f"Using {len(early_cols)} features; dropped {len(dropped)} second-semester features.")

    X_train_e = X_train[early_cols]
    X_test_e = X_test[early_cols]

    # Same recipe as the deployed Logistic Regression: scale, then fit.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_e)
    X_test_scaled = scaler.transform(X_test_e)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, preds)
    print(f"Early-warning accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES))

    # Normalized confusion matrix (diagonal = per-class recall).
    cm_norm = confusion_matrix(y_test, preds, normalize='true')
    print("Normalized confusion matrix (row-wise, diagonal = recall):")
    print(cm_norm.round(2))
    save_confusion_heatmap(
        cm_norm,
        "Early Warning (1st sem) - Normalized Confusion Matrix",
        os.path.join(
            PLOTS_DIR, "confusion_matrix_early_warning_normalized.png"),
        values_format='.2f',
    )

    # Which early features drive the prediction?
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_summary("Early Warning", explainer, X_test_scaled, early_cols)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Reuse the exact same split as training (random_state=42, stratified).
    X_train, X_test, y_train, y_test = load()
    logistic_model, scaler, decision_tree_model = load_artifacts()
    print(f"Loaded models and test set ({X_test.shape[0]} students).")

    # Each model expects features the same way it was trained:
    # Logistic Regression -> scaled, Decision Tree -> raw.
    X_test_scaled = scaler.transform(X_test)
    predictions = {
        "Logistic Regression": logistic_model.predict(X_test_scaled),
        "Decision Tree": decision_tree_model.predict(X_test),
    }

    for name, preds in predictions.items():
        confusion_matrix_analysis(name, y_test, preds)

    shap_analysis(logistic_model, decision_tree_model,
                  scaler, X_train, X_test, X_test_scaled)

    # True early-warning test: retrain on first-semester-only features.
    early_warning_analysis(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
