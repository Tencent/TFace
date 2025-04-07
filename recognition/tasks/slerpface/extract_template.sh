### get SlerpFace model features ###
python3 -um tasks.slerpface.extract_template.verification \
    --model_path=./tasks/slerpface/ckpt/Backbone_Epoch_24_checkpoint.pth \
    --data_root=/remote-home/share/yxmi/datasets/val_data \
    --backbone=IR_50 \
    --batch_size=512 \
    --output_dir=./tasks/slerpface/templates \
    --gpu_ids=0