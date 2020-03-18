import os
import tempfile
import typing
import mlflow
import io
import tensorflow as tf
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
import mlflow.keras


def log_metrics_and_params(
    history: tf.keras.callbacks.History,
    config: typing.Optional[typing.NamedTuple],
    mlflow_active_run: mlflow.ActiveRun,
):
    """ log metrics from history returned by keras.model.fit
    use it with
    with mlflow.start_run() as active_run:

    See https://medium.com/@ij_82957/how-to-reduce-mlflow-logging-overhead-by-using-log-batch-b61301cc540f
    """
    mlflow_client = MlflowClient()
    all_metrics = []
    for metric_name in history.history:
        for i in history.epoch:
            metric = Metric(
                key=metric_name,
                value=history.history[metric_name][i],
                timestamp=i,
                step=i,
            )
            all_metrics.append(metric)
    all_params = list()
    if config is not None:
        for k, v in config._asdict().items():
            param = Param(key=k, value=str(v))
            all_params.append(param)

    mlflow_client.log_batch(
        run_id=mlflow_active_run.info.run_id, metrics=all_metrics, params=all_params
    )


def tag_model(model):
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.writelines(x + " \n"))
    mlflow.set_tag("model_summary", buffer.getvalue())

    with tempfile.TemporaryDirectory() as tempdir:
        summary_file = os.path.join(tempdir, "model_summary.txt")
        with open(summary_file, "w") as f:
            f.write(buffer.getvalue())
        mlflow.log_artifact(local_path=summary_file)

    mlflow.keras.log_model(model, artifact_path="model")

    for attribute, value in model.optimizer.get_config().items():
        mlflow.log_param("optim_" + attribute, value)
