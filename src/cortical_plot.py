import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import datasets, plotting

def cortical_plot(data, hemisphere="left", cols=None,view="lateral",cmap="gist_ncar"):
    """Plot the destrieux cortical map

    Args:
        data (a pandas dataframe): has a 'label' column, and one or more columns with the scores, one for which the cortical plot will be made
    """
    SCORE_COLS = data.columns.tolist()
    SCORE_COLS.remove("label")
    if cols is not None:
        SCORE_COLS = list(set(SCORE_COLS) & set(cols))
    LABEL_COL = "label"

    MESH_RES = "fsaverage5"                    # 'fsaverage5' (fast) or 'fsaverage7' (prettier)
    CMAP = cmap                          
    # Views to render for each hemisphere (choose from: 'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior')
    # --- Load meshes (fsaverage) ---
    fsavg = datasets.fetch_surf_fsaverage(mesh=MESH_RES)

    # --- Load Destrieux surface atlas (labels live on fsaverage) ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        destrieux = datasets.fetch_atlas_surf_destrieux(verbose=0)

    if hemisphere == "left":
        hem_labels = destrieux["map_left"]
    else:
        hem_labels = destrieux["map_right"]
    label_names = list(destrieux["labels"])  # index i names the parcel with integer i in the label maps
    name_to_idx = {name: i for i, name in enumerate(label_names)}


    for col in SCORE_COLS:
        data.loc[:, col] = data[col].astype(float)
        # Make vertex-wise arrays filled with NaN so missing parcels don't bias color scaling
        hem_tex = np.full(hem_labels.shape, np.nan, dtype=float)

        # Build a dict: parcel_name -> score
        scores = dict(zip(data[LABEL_COL], data[col]))

        # Paint scores into vertices for each parcel index
        filled_count = 0
        for name, idx in name_to_idx.items():
            if name in scores and np.isfinite(scores[name]):
                val = float(scores[name])
                hem_tex[hem_labels == idx] = val
                filled_count += 1

        if filled_count == 0:
            warnings.warn(f"No parcels matched for column '{col}'. Skipping.")
            continue

        if hemisphere == "left":
            surf_mesh = fsavg.infl_left
            bg_map = fsavg.sulc_left
        else:
            surf_mesh = fsavg.infl_right
            bg_map = fsavg.sulc_right

        plotting.plot_surf_stat_map(
            surf_mesh=surf_mesh,
            stat_map=hem_tex,
            bg_map=bg_map,
            hemi=hemisphere,
            view=view,
            cmap=CMAP,
            colorbar=True,
            darkness=None,
            threshold=None,
            title=f"{col}",
        )
        plt.show()
