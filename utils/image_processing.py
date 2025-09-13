import torch

# Takes in a tensor representation of an image with dimension (C,H,W)
# Returns a tensor representing patches of size PxP of the input image
def patchify(img, patch_size):
    # simple assertions that we must satisfy to get clean non-overlapping patches
    # may want to handle this differently, such as adding padding
    assert img.shape[1] % patch_size == 0
    assert img.shape[2] % patch_size == 0

    m = int(img.shape[1] / patch_size) # number of vertical jumps
    n = int(img.shape[2] / patch_size) # number of horizontal jumps

    patches = []
    for i in range(n):
        for j in range(m):
            channel_seperated_patch = img[:, patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1)]

            # Unsure if we should permute the entire image tensor before loop, consider performance here
            flattened_patch = channel_seperated_patch.permute(1,2,0).flatten() 

            patches.append(flattened_patch)

    return torch.stack(patches)

# TODO: should move this function as it is not really part of image preprocessing
def position_encoder(num_pos, pos_embedding_size):
    pos_enc = torch.zeros((num_pos, pos_embedding_size))
    for i in range(num_pos):
        for j in range(pos_embedding_size):
            if j % 2 == 0:
                pos_enc[i][j] = torch.sin(torch.tensor(i / (10000**(j // pos_embedding_size))))
            else:
                pos_enc[i][j] = torch.cos(torch.tensor(i / (10000**((j+1) // pos_embedding_size))))
    return pos_enc