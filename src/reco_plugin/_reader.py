import h5py
import numpy as np

def napari_get_reader(path):

    if isinstance(path, list):
        path = path[0]

    if path.endswith('.nxs') or path.endswith('.tdf'):  
        return read_nxs

def find_datasets_with_dim_3(file, group=None, path="", results=None):
    """
    Find all datasets with 3 dimensions in a HDF5 file.       
    """
    if results is None:
        results = []

    if group is None:
        group = file

    for key in group:
        item = group[key]
        current_path = f"{path}/{key}"
        if isinstance(item, h5py.Group):
            find_datasets_with_dim_3(
                file, group=item, path=current_path, results=results
            )
        elif isinstance(item, h5py.Dataset):
            if len(item.shape) == 3:
                results.append(
                    (current_path, item.shape)
                ) 
    return results

def read_nxs(paths):
    """
    Reads full 3D volumes from HDF5/NXS files and organizes layers for each dataset.

    Parameters
    ----------
    paths : list[str] | str
        Paths to files to be processed.

    Returns
    -------
    list
        A list containing tuples for each dataset.
    """

    if isinstance(paths, str):
        paths = [paths]

    dataset_layers = {} 
    for path in paths:
        with h5py.File(path, "r") as h5file:
            print(f"Processing file: {path}")
            datasets_3d = find_datasets_with_dim_3(h5file)

            if not datasets_3d:
                print(f"No 3D datasets found in {path}.")
                continue

            for dataset_key, shape in datasets_3d:
                print(f"Loading full volume: {dataset_key}")

                data = np.array(h5file[dataset_key], dtype=np.float32)

                if dataset_key not in dataset_layers:
                    dataset_layers[dataset_key] = []

                dataset_layers[dataset_key].append(data)

    layers = []
    for dataset_key, volumes in dataset_layers.items():
        try:
            if len(volumes) > 1:
                stacked_volumes = np.stack(volumes, axis=0)
            else:
                stacked_volumes = volumes[0]  

        except ValueError as e:
            print(f"Error stacking volumes for dataset {dataset_key}: {e}")
            continue

        name_image = dataset_key.strip("/").split("/")[-1]
        metadata = {
            "paths": paths,
            "dataset_key": dataset_key,
            "multiscale": False,
        }

        layers.append(
            (
                stacked_volumes,
                {"name": name_image, "metadata": metadata},
                "image",
            )
        )

    return layers

                
