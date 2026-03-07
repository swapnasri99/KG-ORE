#for converting the entitiy linking results into sql db
python entity_store_disk.py build entity_linking_results/msmarco_passage_with_id.jsonl passage_entities.db

#to run the experiment

python run_experiment_unified.py \
  --dl 19 --budget 50 \
  --ce 4 --s1 10 --s2 15 --s 10 \
  --gamma 1.0 --alpha 0.0 --beta 0.0 \
  --passage_el_db passage_entities.db \
  --freebase_dir freebase/ \
  --mode overwrite




CUDA_VISIBLE_DEVICES="" python run_experiment_unified_mmead.py \
    --dl 19 --budget 50 --ce 4 --s1 10 --s2 15 --s 10 \
    --alpha 0.0 --beta 0.5 --gamma 0.5 \
    --use_mmead --mmead_cache mmead_cache.db \
    --mode overwrite --verbose 


CUDA_VISIBLE_DEVICES="" python run_experiment_unified_patched.py \
  --dl 19 --budget 50 \
  --ce 4 --s1 10 --s2 15 --s 10 \
  --alpha 0.0 --beta 0.0 --gamma 1.0 \
  --kg_mode minmax \
  --passage_el_db passage_entities.db \
  --query_el query_dev_full_test_with_id.jsnol\
  --freebase_dir freebase/ \
  --verbose \
 

python diagnose_kg_neighbors.py \
  --dl 19 --budget 50\
  --passage_el_db passage_entities.db \
  --freebase_dir freebase/ \
  --query_el query_dev_full_test_with_id.jsonl \
  --alpha 0.0 --beta 0.0 --gamma 1.0 \
  --output exp1_kg_only.csv


CUDA_VISIBLE_DEVICES=""  python run_experiment_unified_v2.py \
  --dl 19 --budget 50 --ce 4 --s 10 --s2 15 \
  --alpha 0.0 --beta 0.0 --gamma 1.0 \
  --passage_el_db passage_entities.db \
  --freebase_dir freebase/ \
  --query_el query_dev_full_test_with_id.jsonl \
  --verbose

