from clearml import Task
from clearml.automation.optuna import OptimizerOptuna

# ✅  "Pipeline step 3 train model" Task ID
BASE_TRAIN_TASK_ID = "replace_with_your_training_task_id"

optimizer = OptimizerOptuna(
    base_task_id=BASE_TRAIN_TASK_ID,
    hyper_parameters=[
        {
            "name": "learning_rate",
            "type": "float",
            "min": 0.0001,
            "max": 0.01,
            "log": True
        },
        {
            "name": "weight_decay",
            "type": "float",
            "min": 1e-6,
            "max": 1e-3,
            "log": True
        },
        {
            "name": "batch_size",
            "type": "int",
            "min": 16,
            "max": 64,
            "step": 16
        }
    ],
    objective_metric='validation_accuracy',   #
    objective_metric_goal='maximize',
    num_concurrent_workers=2,
    max_iteration_per_job=1,
    total_max_jobs=10,
    project_name="AI_Studio_Demo",
    task_name="HPO: Train Model"
)

optimizer.set_time_limit(in_minutes=60)

optimizer.start()