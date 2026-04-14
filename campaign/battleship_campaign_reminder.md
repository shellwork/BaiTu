
run this two command in two windows

```
streamlit run battleship_campaign_dashboard.py

python battleship_campaign.py \
  --max_cycles 20 \
  --query_size 8 \
  --replicates 3 \
  --validation_policy_count 16 \
  --validation_replicates 4 \
  --ensemble_members 10 \
  --out_dir battleship_campaign_results1
```