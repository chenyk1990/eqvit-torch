"""EQViT-torch: PyTorch P picking and rapid magnitude estimation for research/EEW prototyping."""
from .models import EQViTPicker, EQViTMagnitude, build_picker, build_magnitude
from .io import TXEDDataset, load_txed_record, split_ids_by_event
from .inference import pick_p, estimate_magnitude, mc_magnitude, EarlyWarningPipeline
from .weights import load_checkpoint, load_legacy_torch_model
from .metrics import magnitude_metrics, picking_metrics
__version__='0.2.0'
