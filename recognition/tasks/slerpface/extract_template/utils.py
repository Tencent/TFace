import numpy as np
import torch


def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def process_batch(batch, backbone, device):
    """Process a single batch of data
    Args:
        batch: Input data batch
        backbone: Model
        device: GPU device
    Returns:
        tuple: (group_features, attention_maps, embeddings)
    """
    # Move data to GPU
    batch = batch.to(device)
    flipped = torch.flip(batch, dims=[3])
    
    # Generate group features
    group_features = backbone.gen_group_feature(batch, flip=True)
    attention_maps = backbone.gen_weight_map(group_features)
    
    # Generate vector features
    vector_features = (
        backbone.gen_vector_feature(batch).cpu() + 
        backbone.gen_vector_feature(flipped).cpu()
    )
    embeddings = l2_norm(vector_features)
    
    return (
        group_features.cpu().detach().numpy(),
        attention_maps.cpu().detach().numpy(),
        embeddings
    )

def get_templates(
    embedding_size,
    batch_size,
    backbone,
    carray,
    group_width=7,
    group_size=16,
    name='LFW',
    gpu_ids=0
):
    """Extract registration and matching templates for SlerpFace model
    Args:
        embedding_size: Feature dimension
        batch_size: Batch size
        backbone: Model
        carray: Input data
        group_width: Width of group
        group_size: Size of group
        name: Dataset name
        gpu_ids: GPU ID
    Returns:
        tuple: (gallery_templates_dict, query_templates_dict)
    """
    backbone.eval()
    device = torch.device(f'cuda:{gpu_ids}')
    
    # Initialize storage arrays
    embeddings = np.zeros([len(carray), embedding_size])
    attention_maps = np.zeros([len(carray), group_width, group_width, 1])
    group_features = np.zeros([len(carray), group_size, group_width, group_width])
    
    # Process complete batches
    with torch.no_grad():
        for idx in range(0, len(carray) - batch_size + 1, batch_size):
            print("S", end='')
            batch = torch.tensor(carray[idx:idx + batch_size])
            
            # Process current batch
            g_feat, a_maps, emb = process_batch(batch, backbone, device)
            
            # Store results
            group_features[idx:idx + batch_size] = g_feat
            attention_maps[idx:idx + batch_size] = a_maps
            embeddings[idx:idx + batch_size] = emb
            
            print('=', end='')
        
        # Process remaining data
        if idx + batch_size < len(carray):
            print("S", end='')
            batch = torch.tensor(carray[idx + batch_size:])
            
            # Process remaining batch
            g_feat, a_maps, emb = process_batch(batch, backbone, device)
            
            # Store results
            group_features[idx + batch_size:] = g_feat
            attention_maps[idx + batch_size:] = a_maps
            embeddings[idx + batch_size:] = emb
            print('=')
    
    # Separate gallery and query templates
    gallery_templates = {
        'vector_feature': embeddings[0::2],
        'group_feature': group_features[0::2],
        'attention_map': attention_maps[0::2]
    }
    
    query_templates = {
        'vector_feature': embeddings[1::2],
        'group_feature': group_features[1::2],
        'attention_map': attention_maps[1::2]
    }
    
    return gallery_templates, query_templates
