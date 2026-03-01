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


### 
Thesis Name :  Adaptive Re-Ranking with Knowledge Graph Derived Entity Embeddings


