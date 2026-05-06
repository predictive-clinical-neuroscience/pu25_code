# protocol_update_2025_code

This repository contains code examples corresponding from the paper [Protocol Update: The Normative Modelling Paradigm for Computational Psychiatry](https://www.biorxiv.org/content/10.64898/2026.02.17.706268.full).

The notebooks include the main normative modelling workflow, longitudinal modelling, model comparison, data synthesis, data harmonization, federated learning workflows (model transfer, model extension, model merging) and a real-life scenario where we apply model transfer using pre-trained models from [Rutherford et al. (2022)](https://elifesciences.org/articles/72904)

Note that you need to run `1_main_workflow.ipynb` first to get the trained model used by the other notebooks. The exceptions are `2.1_longitudinal_modelling.ipynb` and `4.1_model_transfer_lifespan_reference.ipynb`, which use their own data.
