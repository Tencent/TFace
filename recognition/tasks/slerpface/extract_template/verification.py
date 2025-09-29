import os
import sys
import argparse
import numpy as np
import torch
from tasks.slerpface.extract_template.utils import get_templates
from tasks.slerpface.modules.model import SlerpFace
from test.utils import get_val_data_from_bin


def parse_args():
    """Parse command line arguments for template extraction"""
    parser = argparse.ArgumentParser(description='Template extraction tool for SlerpFace')
    parser.add_argument('--model_path', default=None, required=True, help='Path to the model checkpoint')
    parser.add_argument('--backbone', default='IR_18', help='Backbone architecture type')
    parser.add_argument('--gpu_ids', default='0', help='GPU device IDs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for template extraction')
    parser.add_argument('--data_root', default='', required=True, help='Root directory of validation data')
    parser.add_argument('--output_dir', default='./templates', help='Directory to save extracted templates')
    return parser.parse_args()


def main():
    """Extract templates from face verification datasets (LFW, CFP-FP, AgeDB, CALFW, CPLFW)
    and save them for later use in face verification tasks.
    """
    args = parse_args()

    # Initialize SlerpFace model
    slerp_model = SlerpFace(
        input_size=[112, 112],
        num_layers=int(args.backbone.split("_")[-1]),
        group_size=16
    ).cuda()
    slerp_model.load_state_dict(torch.load(args.model_path, weights_only=True))

    # Load validation datasets
    lfw, cfp, agedb, cplfw, calfw, \
    _, _, _, _, _ = get_val_data_from_bin(args.data_root)

    # Move model to specified GPU
    slerp_model = slerp_model.to(torch.device(f'cuda:{args.gpu_ids}'))

    print("Extracting templates from LFW, cfp, AgeDB, CPLFW...")

    # Define dataset configurations
    dataset_configs = {
        'LFW': lfw,
        'CFP': cfp,
        'AGEDB': agedb,
        'CALFW': calfw,
        'CPLFW': cplfw
    }
    
    # Initialize template storage
    extracted_templates = {
        'gallery': {},
        'query': {}
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract and save templates for each dataset
    for dataset_name, dataset_data in dataset_configs.items():
        gallery_templates, query_templates = get_templates(
            512,
            args.batch_size,
            slerp_model,
            dataset_data,
            name=dataset_name,
            gpu_ids=args.gpu_ids
        )
        
        print(f"{dataset_name} gallery templates shape: {gallery_templates['group_feature'].shape}")
        
        # Save extracted templates and attention maps
        output_prefix = os.path.join(args.output_dir, dataset_name)
        np.save(f'{output_prefix}_gallery_templates.npy', gallery_templates['group_feature'])
        np.save(f'{output_prefix}_query_templates.npy', query_templates['group_feature'])
        np.save(f'{output_prefix}_gallery_attention_map.npy', gallery_templates['attention_map'])
        np.save(f'{output_prefix}_query_attention_map.npy', query_templates['attention_map'])


if __name__ == "__main__":
    main()
