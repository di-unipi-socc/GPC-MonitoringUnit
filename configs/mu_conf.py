from types import SimpleNamespace as namespace
from local.config_local import inputs_local, gate_id, crowd_thresh, neural_processor

inputs = inputs_local

t_conf = namespace(
    gate_id=gate_id,
    setup_grid=False,
    crowd_thresh=crowd_thresh,
    config='configs/person.py',
    device=neural_processor,  # MYRIAD, CPU
    input=[],
    gate_rects=[],
    out_thresholds=[],
    loop=False,
    m_detector='intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml',
    m_reid='intel/person-reidentification-retail-0031/FP16/person-reidentification-retail-0031.xml',
    no_show=True,
    t_detector=0.6,

    cpu_extension=None,
    m_segmentation=None,
    detections=None,
    utilization_monitors=''
)

for video_src in inputs:
    t_conf.input.append(video_src[0])
    t_conf.gate_rects.append(video_src[1])
    t_conf.out_thresholds.append(video_src[2])

assert len(t_conf.input) == len(t_conf.gate_rects) == len(t_conf.out_thresholds)
