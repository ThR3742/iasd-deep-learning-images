import tempfile
import glob
import shutil
import tensorflow as tf
import mlflow
from dli.mlflow_helpers import log_metrics_and_params, tag_model


def test_log_metrics_and_params():
    mlflow_root = tempfile.mkdtemp()
    mlflow.set_tracking_uri(mlflow_root)
    mlflow.create_experiment("xp")

    with mlflow.start_run(nested=False) as active_run:
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
        model.compile(optimizer="sgd", loss=tf.keras.losses.mean_squared_error)
        history = model.fit(x=[1, 1, 1], y=[2, 2, 2], verbose=1)

        log_metrics_and_params(
            history=history, config=None, mlflow_active_run=active_run
        )

        tag_model(model)

    print(glob.glob(f"{mlflow_root}/*"))

    shutil.rmtree(mlflow_root)
