"""
IronICL in a skrub pipeline, evaluated with skore
=================================================

``IronICLClassifier`` is a scikit-learn classifier, so it slots into the same
tooling as any other estimator. Here :func:`skrub.tabular_pipeline` turns a
raw table with strings and missing values into numeric features, and
:class:`skore.EstimatorReport` evaluates the result.

Running this example needs ``pip install ironicl skrub skore`` and downloads
the TabICLv2 checkpoint on first use.
"""

# %%
# Load a small mixed-type classification table.
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

credit = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
X, y = credit.data, credit.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# %%
# ``tabular_pipeline`` vectorises the table, imputes missing values and scales
# numeric columns before IronICL sees them. The engine needs finite numeric
# input, which is exactly what the pipeline produces.
from skrub import tabular_pipeline

from ironicl import IronICLClassifier

pipeline = tabular_pipeline(IronICLClassifier(n_estimators=4))
print(pipeline)

# %%
# skore's report fits the pipeline and summarises its test metrics.
from skore import EstimatorReport

report = EstimatorReport(
    pipeline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)
print(f"accuracy: {report.metrics.accuracy():.3f}")
print(f"ROC AUC:  {report.metrics.roc_auc():.3f}")

# %%
# The fitted estimator keeps the training context on disk and in memory, so
# repeated predictions only pay for the query rows.
print(report.estimator_[-1].context_path_)
