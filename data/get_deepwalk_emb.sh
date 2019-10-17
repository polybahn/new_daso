source activate deepwalk
deepwalk --format edgelist --input /home/yao/Desktop/new_daso/data/ciao/new_social.csv --max-memory-data-size 0 --number-walks 80 --representation-size 50 --walk-length 40 --window-size 10 --workers 6 --output /home/yao/Desktop/new_daso/data/ciao/new_social_emb_deepwalk_50.emb
deepwalk --format edgelist --input /home/yao/Desktop/new_daso/data/ciao/new_rating.csv --max-memory-data-size 0 --number-walks 80 --representation-size 50 --walk-length 40 --window-size 10 --workers 6 --output /home/yao/Desktop/new_daso/data/ciao/new_rating_emb_deepwalk_50.emb
