DATA_PATH_BASE='/digital'
DATASET='jpeg_compression'
LEVEL='5'
RESUME_MODEL='/checkpoints/mae_pretrain_vit_large_full.pth'
RESUME_FINETUNE='/checkpoints/prob_lr1e-3_wd.2_blk12_ep20.pth'
OUTPUT_DIR_BASE='output'
python test_without_adaptation.py \
        --data_path "$DATA_PATH_BASE/$DATASET/$LEVEL" \
        --model mae_vit_large_patch16 \
        --input_size 224 \
        --resume_model ${RESUME_MODEL} \
        --resume_finetune ${RESUME_FINETUNE} \
        --output_dir "$OUTPUT_DIR_BASE/$DATASET/baseline" \
        --classifier_depth 12 \
        --head_type "vit_head" 
