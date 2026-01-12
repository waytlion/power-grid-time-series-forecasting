# Datakit Cheat Sheet

## Data Validation
(.venv) PS F:\studium\MA-code> python -m gridfm_datakit.cli validate ./data_out/phase1_baseline/case14_ieee/raw --mode "pf"

## Data gen
(.venv) PS F:\studium\MA-code> python -m gridfm_datakit.cli generate phase1_generation/configs/phase1_config.yaml
(.venv) PS F:\studium\Thesis_Repo> gridfm_datakit generate generation/configs/phase1_baseline_config.yaml 

## Stats 
python -m gridfm_datakit.cli stats ./data_out/phase1_baseline/case14_ieee/raw --n-partitions 100