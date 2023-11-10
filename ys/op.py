import cv2
import torch
import numpy as np

# uv padding from threestudio
def uv_padding(image, mask, padding=2, backend='knn'):
    # image: [H, W, 3] torch.tensor or np.ndarray in [0, 1]
    # mask: [H, W] torch.tensor or np.ndarray, bool, regions with texture.
    # padding: size to pad into mask

    if torch.is_tensor(image):
        image_input = image.detach().cpu().numpy()
    else:
        image_input = image

    if torch.is_tensor(mask):
        mask_input = mask.detach().cpu().numpy()
    else:
        mask_input = mask
    
    # padding backend
    if backend == 'knn':

        from sklearn.neighbors import NearestNeighbors
        from scipy.ndimage import binary_dilation, binary_erosion

        inpaint_region = binary_dilation(mask_input, iterations=padding)
        inpaint_region[mask_input] = 0

        search_region = mask_input.copy()
        not_search_region = binary_erosion(search_region, iterations=2)
        search_region[not_search_region] = 0

        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
        _, indices = knn.kneighbors(inpaint_coords)

        inpaint_image = image_input.copy()
        inpaint_image[tuple(inpaint_coords.T)] = inpaint_image[tuple(search_coords[indices[:, 0]].T)]

    elif backend == 'cv2':
        # kind of slow
        inpaint_image = cv2.inpaint(
            (image_input * 255).astype(np.uint8),
            (~mask_input * 255).astype(np.uint8),
            padding,
            cv2.INPAINT_TELEA,
        ).astype(np.float32) / 255

    if torch.is_tensor(image):
        inpaint_image = torch.from_numpy(inpaint_image).to(image)
    
    return inpaint_image
