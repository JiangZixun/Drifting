#!/usr/bin/env bash

bash scripts/cloudseg/train_drifting_unet_all_class.sh --wandb
bash scripts/cloudseg/train_drifting_unet_all_{Ci_Ac_Cu_St}.sh --wandb