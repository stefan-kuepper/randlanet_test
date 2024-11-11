# %%
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import dataset
from logging import getLogger, DEBUG
import numpy as np


def pred_custom_data(pc_names: list[str], pcs: list[dict], pipeline):
    vis_points = []
    for i, data in enumerate(pcs):
        print(f"Processing {pc_names[i]}")
        name = pc_names[i]

        results = pipeline.run_inference(data)
        pred_label = (results["predict_labels"] + 1).astype(np.int32)
        # Fill "unlabeled" value because predictions have no 0 values.
        pred_label[0] = 0

        label = data["label"]
        pts = data["point"]
        colors = data["feat"]
        vis_d = {
            "name": name,
            "points": pts,
            "colors": colors,
        }
        vis_points.append(vis_d)

        vis_d = {
            "name": name + "_randlanet",
            "points": pts,
            "labels": pred_label,
        }
        vis_points.append(vis_d)

    return vis_points


logger = getLogger(__name__)
logger.setLevel(DEBUG)
# %%
cfg_file = "randlanet_dorsten3d.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.RandLANet(**cfg.model)
dataset = dataset.Dorsten3D(**cfg.dataset)
pipeline = ml3d.pipelines.SemanticSegmentation(
    model, dataset=dataset, device="cpu", **cfg.pipeline
)
# %%
# download the weights.
# ckpt_folder = "./logs/"
# os.makedirs(ckpt_folder, exist_ok=True)
# ckpt_path = ckpt_folder + "randlanet_semantickitti_202201071330utc.pth"
# randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth"
# if not os.path.exists(ckpt_path):
#     cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
#     os.system(cmd)

# load the parameters.
pipeline.load_ckpt(ckpt_path=cfg.model.ckpt_path)

test_split = dataset.get_split("test")
print(f"processing {len(test_split)} files")
data = [test_split.get_data(i) for i in range(len(test_split))]
names = [test_split.get_attr(i)["name"] for i in range(len(test_split))]

# visualize
vis_points = pred_custom_data(names, data, pipeline)

kitti_labels = ml3d.datasets.Toronto3D.get_label_to_names()
v = ml3d.vis.Visualizer()
lut = ml3d.vis.LabelLUT()
for val in sorted(kitti_labels.keys()):
    lut.add_label(kitti_labels[val], val)
v.set_lut("labels", lut)
v.set_lut("pred", lut)

v.visualize(vis_points)

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
# result = pipeline.run_inference(data)

# evaluate performance on the test set; this will write logs to './logs'.
# pipeline.run_test()
