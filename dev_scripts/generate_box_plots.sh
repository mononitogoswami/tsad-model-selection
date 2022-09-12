# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
python generate_box_plot_smd.py --split 'train' --metric 'influence' --aggregation_type 'borda' --overwrite True 
python generate_box_plot_smd.py --split 'train' --metric 'influence' --aggregation_type 'kemeny' --overwrite True 
python generate_box_plot_smd.py --split 'train' --metric 'pagerank' --overwrite True 
python generate_box_plot_smd.py --split 'train' --metric 'averagedistance' --overwrite True 

python generate_box_plot_smd.py --split 'test' --metric 'influence' --aggregation_type 'borda' --overwrite True 
python generate_box_plot_smd.py --split 'test' --metric 'influence' --aggregation_type 'kemeny' --overwrite True 
python generate_box_plot_smd.py --split 'test' --metric 'pagerank' --overwrite True
python generate_box_plot_smd.py --split 'test' --metric 'averagedistance' --overwrite True 

python generate_box_plot_anomaly_archive.py --split 'train' --metric 'influence' --aggregation_type 'borda' --overwrite True 
python generate_box_plot_anomaly_archive.py --split 'train' --metric 'influence' --aggregation_type 'kemeny' --overwrite True 
python generate_box_plot_anomaly_archive.py --split 'train' --metric 'pagerank' --overwrite True
python generate_box_plot_anomaly_archive.py --split 'train' --metric 'averagedistance' --overwrite True 

python generate_box_plot_anomaly_archive.py --split 'test' --metric 'influence' --aggregation_type 'borda' --overwrite True 
python generate_box_plot_anomaly_archive.py --split 'test' --metric 'influence' --aggregation_type 'kemeny' --overwrite True 
python generate_box_plot_anomaly_archive.py --split 'test' --metric 'pagerank' --overwrite True
python generate_box_plot_anomaly_archive.py --split 'test' --metric 'averagedistance' --overwrite True 