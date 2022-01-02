#Train
lr=0.0001
wd=0
bs=1024
shuffle=True
epoch=300
negative_slope=0.2
n_neg=1
# Model
model='LGCN'
pretrain=False
aggr='add'
emb_size=64
pool='mean'
# Save and device
data_path='/local/DEEPLEARNING/graph_datasets/preprocess_gowalla'
gpu=0

python3 main.py --model $model --learning_rate $lr --n_neg $n_neg --negative_slope $negative_slope --wd $wd --batch_size $bs --epoch $epoch --aggr $aggr --pool $pool --shuffle $shuffle --emb_size $emb_size --pretrain $pretrain --data_path $data_path --gpu $gpu 