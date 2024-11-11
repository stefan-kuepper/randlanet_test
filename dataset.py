import numpy as np
from pathlib import Path
from os.path import join, exists, dirname, abspath
from pathlib import Path
from sklearn.neighbors import KDTree
import logging
import open3d as o3d
import laspy
import geopandas as gpd
from open3d._ml3d.datasets.base_dataset import BaseDataset, BaseDatasetSplit
from open3d._ml3d.utils import make_dir, DATASET

log = logging.getLogger(__name__)


class Dorsten3D(BaseDataset):
    """Dorsten3D dataset, used in visualizer, training, or test."""

    def __init__(
        self,
        dataset_path,
        name="Dorsten3D",
        cache_dir="./logs/cache",
        use_cache=False,
        num_points=65536,
        test_result_folder="./test",
        **kwargs,
    ):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Semantic3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            num_points: The maximum number of points to use when splitting the dataset.
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.

        Returns:
            class: The corresponding class.
        """
        super().__init__(
            dataset_path=dataset_path,
            name=name,
            cache_dir=cache_dir,
            use_cache=use_cache,
            num_points=num_points,
            test_result_folder=test_result_folder,
            **kwargs,
        )
        cfg = self.cfg

        self.dataset_path = Path(cfg.dataset_path)

        self.num_classes = 1

        las_df = gpd.read_file(self.dataset_path / "dataset.gpkg", layer="las")
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(len(las_df))

        train_indices = indices[: int(len(las_df) * 0.8)]
        val_indices = indices[int(len(las_df) * 0.8) : int(len(las_df) * 0.9)]
        test_indices = indices[int(len(las_df) * 0.9) :]

        self.train_files = self.dataset_path / las_df.iloc[train_indices].paths.values
        self.val_files = self.dataset_path / las_df.iloc[val_indices].paths.values
        self.test_files = self.dataset_path / las_df.iloc[test_indices].paths.values

    def get_split(self, split):  # pyright: ignore
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return Dorsten3DSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of files belonging to the dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A list of file paths.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        if split == "test":
            return self.test_files
        elif split == "train":
            return self.train_files
        elif split == "val":
            return self.val_files
        elif split == "all":
            return self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(self.split))

    def is_tested(self, attr):  # pyright: ignore
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr["name"]
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + ".npy")
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr["name"].split(".")[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results["predict_labels"]
        pred = np.array(pred)

        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        store_path = join(path, self.name, name + ".npy")
        make_dir(Path(store_path).parent)
        np.save(store_path, pred)
        log.info("Saved {} in {}.".format(name, store_path))

    @staticmethod
    def get_label_to_names():
        pass


class Dorsten3DSplit(BaseDatasetSplit):
    def __init__(self, dataset, split="training"):
        super().__init__(dataset, split=split)

        log.info("Found {} pointclouds for {}".format(len(self.path_list), split))
        self.UTM_OFFSET = [627285, 4841948, 0]

    def __len__(self):  # pyright: ignore
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        with laspy.open(pc_path) as f:
            las = f.read()
            scales = las.header.scales
            points = np.column_stack(
                (
                    np.array(las.X * scales[0]),
                    np.array(las.Y * scales[1]),
                    np.array(las.Z * scales[2]),
                )
            )
            # feat: np.ndarray = np.array(las.intensity)
            feat = np.column_stack(
                (
                    np.float32(np.array(las.red)) / 2**16,
                    np.float32(np.array(las.green)) / 2**16,
                    np.float32(np.array(las.blue)) / 2**16,
                )
            )

        points = np.float32(points)

        labels = np.zeros(points.shape[0], dtype=np.int32)  # TODO: create labels

        data = {"point": points, "feat": feat, "label": labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace(".txt", "")

        pc_path = str(pc_path)
        split = self.split
        attr = {"idx": idx, "name": name, "path": pc_path, "split": split}

        return attr


DATASET._register_module(Dorsten3D)
