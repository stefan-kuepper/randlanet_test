# %%
import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

# %%
cfg_file = "randlanet_toronto3d.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.RandLANet(**cfg.model)
dataset = ml3d.datasets.Toronto3D(**cfg.dataset)
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
data = test_split.get_data(0)

# run inference on a single example.
# returns dict with 'predict_labels' and 'predict_scores'.
result = pipeline.run_inference(data)

# evaluate performance on the test set; this will write logs to './logs'.
pipeline.run_test()
