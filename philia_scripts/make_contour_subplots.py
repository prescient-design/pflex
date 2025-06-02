
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import corner
import scienceplots

plt.style.use('default')
plt.style.use(['nature', "no-latex"])
# plt.style.use('ggplot')
plt.rcParams['axes.facecolor']= "white"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["xtick.labelsize"] = 3
plt.rcParams["ytick.labelsize"] = 3

plt.rcParams["axes.labelsize"] = 6

df = pd.read_csv("distances.csv", index_col=None)
df.rename(columns={"w2": "$W_2$"}, inplace=True)

import sys
sys.path.append("scripts")
from pdbids import ids_dict

df["Peptide length (sequence-based split)"] = "unassigned"
df["Peptide length (structure-based split)"] = "unassigned"

seq_keys = [f"pdbids_{l}mer" for l in range(8, 14)]
structure_keys = [f"structure_{l}mer" for l in range(8, 11)] + [f"pdbids_{l}mer" for l in range(11, 14)]
seq_ids_dict = {k: ids_dict[k] for k in seq_keys}
structure_ids_dict = {k: ids_dict[k] for k in structure_keys}

for key, ids in seq_ids_dict.items():
    pep_len = int(key.split("_")[1].replace("mer", ""))
    df.loc[df["pdbid"].isin(ids), "Peptide length (sequence-based split)"] = pep_len

for key, ids in structure_ids_dict.items():
    pep_len = int(key.split("_")[1].replace("mer", ""))
    df.loc[df["pdbid"].isin(ids), "Peptide length (structure-based split)"] = pep_len

plt.close("all")

for pep_len in [8, 9, 10, 11, 12, 13]:
    for split in ["sequence", "structure"]:
        fig, ax = plt.subplots(ncols=pep_len, nrows=2, figsize=(7, 7/4*(8/pep_len))) # , sharex=True, sharey=True)


        highest_w2_pdbids = df.loc[df.groupby(f"Peptide length ({split}-based split)")["$W_2$"].idxmax(), [f"Peptide length ({split}-based split)", "pdbid", "$W_2$"]]
        lowest_w2_pdbids = df.loc[df.groupby(f"Peptide length ({split}-based split)")["$W_2$"].idxmin(), [f"Peptide length ({split}-based split)", "pdbid", "$W_2$"]]

        # Get pdbids
        best_pdbid = lowest_w2_pdbids.loc[lowest_w2_pdbids[f"Peptide length ({split}-based split)"] == pep_len, "pdbid"].values[0]
        worst_pdbid = highest_w2_pdbids.loc[highest_w2_pdbids[f"Peptide length ({split}-based split)"] == pep_len, "pdbid"].values[0]

        # Load dihedral angles
        best_fpd_samples = np.load(f"/scratch/site/u/parj2/pmhc_tensors/{best_pdbid}/10.0/fpd_tensor.npy")  # [P, L-1, 2]
        best_md_samples = np.load(f"/scratch/site/u/parj2/pmhc_tensors/{best_pdbid}/10.0/md_tensor.npy")  # [Q, L-1, 2]

        best_fpd_samples = (best_fpd_samples + 180) % 360 - 180
        best_md_samples = (best_md_samples + 180) % 360 - 180

        worst_fpd_samples = np.load(f"/scratch/site/u/parj2/pmhc_tensors/{worst_pdbid}/10.0/fpd_tensor.npy")  # [P, L-1, 2]
        worst_md_samples = np.load(f"/scratch/site/u/parj2/pmhc_tensors/{worst_pdbid}/10.0/md_tensor.npy")  # [Q, L-1, 2]

        worst_fpd_samples = (worst_fpd_samples + 180) % 360 - 180
        worst_md_samples = (worst_md_samples + 180) % 360 - 180

        levels = [0.9995]
        palette = {"FPD": "#66C2A5", "MD": "tab:purple"}
        for col_idx, pep_pos in enumerate(range(1, pep_len+1)):

            for row_idx in range(2):
                if row_idx == 0:
                    fpd_samples = best_fpd_samples
                    md_samples = best_md_samples
                else:
                    fpd_samples = worst_fpd_samples
                    md_samples = worst_md_samples
                if pep_pos == 1:
                    # There's no phi
                    psi_fpd = fpd_samples[:, col_idx, 0]  # [P]
                    psi_md = md_samples[:, col_idx, 0]  # [Q]
                    samples_label = ["FPD"] * len(psi_fpd) + ["MD"] * len(psi_md)
                    # df = pd.DataFrame({"MD": psi_md, "FPD": psi_fpd}, hue="")
                    df_plot = pd.DataFrame({"$\psi$ / deg": np.concatenate([psi_fpd, psi_md]), "Method": samples_label})
                    g = sns.kdeplot(data=df_plot, y="$\psi$ / deg", hue="Method", ax=ax[row_idx, col_idx], legend=False, common_norm=False, palette=palette)  #, levels=levels)
                    ax[row_idx, col_idx].text(0.05, 0.95, f"PDB ID: {best_pdbid if row_idx == 0 else worst_pdbid}",
                                                                    transform=ax[row_idx, col_idx].transAxes,
                                                                    fontsize=4, color='black', ha='left', va='top',
                                                                    bbox=dict(facecolor='none', alpha=0.5, edgecolor='none'))
                    ax[row_idx, col_idx].set_xlabel("")
                    ax[row_idx, col_idx].set_ylim([-180, 180])
                    ax[row_idx, col_idx].set_yticks([-150, -100, -50, 0, 50, 100, 150])
                    ax[row_idx, col_idx].tick_params(width=0.25, length=2)

                elif pep_pos == pep_len:
                    # There's no psi
                    phi_fpd = fpd_samples[:, -1, 1]  # [P]
                    phi_md = md_samples[:, -1, 1]
                    samples_label = ["FPD"] * len(psi_fpd) + ["MD"] * len(psi_md)
                    df_plot = pd.DataFrame({"$\phi$ / deg": np.concatenate([psi_fpd, psi_md]), "Method": samples_label})
                    g = sns.kdeplot(data=df_plot, x="$\phi$ / deg", hue="Method", ax=ax[row_idx, col_idx], legend=False, common_norm=False, palette=palette)  #, levels=levels)
                    # if row_idx == 0:
                    #     g.legend_.set_title(None)
                    #     g.legend_.prop.set_size(4)
                    #     for text in g.legend_.get_texts():
                    #         text.set_fontsize(4)
                    #     g.legend_.get_frame().set_linewidth(0.0)
                    #     g.legend_.get_frame().set_alpha(0.0)

                    ax[row_idx, col_idx].set_yticklabels([])
                    ax[row_idx, col_idx].set_ylabel("")
                    ax[row_idx, col_idx].set_xlim([-180, 180])
                    ax[row_idx, col_idx].set_xticks([-150, -100, -50, 0, 50, 100, 150])
                    ax[row_idx, col_idx].tick_params(width=0.25, length=2)
                else:
                    # Middle positions
                    phi_fpd = fpd_samples[:, col_idx, 0]  # [P,]
                    psi_fpd = fpd_samples[:, col_idx-1, 1]  # [P,]
                    angles_fpd = np.stack([phi_fpd, psi_fpd], axis=1)  # [P, 2]
                    phi_md = md_samples[:, col_idx, 0]  # [Q,]
                    psi_md = md_samples[:, col_idx-1, 1]  # [Q,]
                    angles_md = np.stack([phi_md, psi_md], axis=1)  # [Q, 2]
                    samples_label = ["FPD"] * len(phi_fpd) + ["MD"] * len(phi_md)
                    # angles = np.concatenate([angles_fpd, angles_md], axis=0)
                    # df_plot = pd.DataFrame({"$\phi$ / deg": angles[:, 0], "$\psi$ / deg": angles[:, 1], "Method": samples_label})
                    # sns.kdeplot(data=df_plot, x="$\phi$ / deg", y="$\psi$ / deg", hue="Method", ax=ax[row_idx, col_idx], legend=False, levels=levels, common_norm=False, palette=palette)
                    buf = 20.0
                    _ = corner.hist2d(angles_fpd[:1000, 0], angles_fpd[:1000, 1], ax=ax[row_idx, col_idx], color=palette["FPD"], plot_datapoints=False, no_fill_contours=True, levels=levels, range=[[-180 - buf, 180 + buf], [-180 - buf, 180 + buf]], plot_density=False)
                    _ = corner.hist2d(angles_md[-300:, 0], angles_md[-300:, 1], ax=ax[row_idx, col_idx], color=palette["MD"], plot_datapoints=False, no_fill_contours=True, levels=levels, range=[[-180 - buf, 180 + buf], [-180 - buf, 180 + buf]], plot_density=False)

                    ax[row_idx, col_idx].set_ylabel("")
                    ax[row_idx, col_idx].set_xlabel("$\phi$ / deg")
                    ax[row_idx, col_idx].set_yticklabels([])
                    ax[row_idx, col_idx].set_ylim([-180, 180])
                    ax[row_idx, col_idx].set_xlim([-180, 180])

                    ax[row_idx, col_idx].set_xticks([-150, -100, -50, 0, 50, 100, 150])
                    ax[row_idx, col_idx].set_yticks([-150, -100, -50, 0, 50, 100, 150])
                    ax[row_idx, col_idx].tick_params(width=0.25, length=2)

                if row_idx == 0:
                    ax[row_idx, col_idx].set_title(f"Pos {pep_pos}", fontsize=6)
                    ax[row_idx, col_idx].set_xticklabels([])


        plt.subplots_adjust(wspace=0, hspace=0)

        fig.savefig(f"md_fpd_figures/{pep_len}_best_worst_{split}.png", dpi=200, transparent=False, bbox_inches="tight")


