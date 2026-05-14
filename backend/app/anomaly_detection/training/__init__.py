from app.anomaly_detection.training.datasets.raw import RawWindowDataset, RawWindowMosaicDataset
from app.anomaly_detection.training.datasets.stat import (
    StatWindowDataset,
    StatWindowMosaicDataset,
    write_sdf_to_mds,
)
from app.anomaly_detection.training.single_stage_trainer import SingleStageTrainer
from app.anomaly_detection.training.flow_matching_trainer import (
    FlowMatchingTrainer,
    Stage1Trainer,
    Stage2Trainer,
    Evaluator,
    EvalResult,
)

# Backward-compat aliases
RobotMosaicDataset = StatWindowMosaicDataset
RobotTorchDataset = RawWindowDataset
V6Trainer = SingleStageTrainer
GLCFlowTrainer = FlowMatchingTrainer
