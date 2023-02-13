import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import SmoothL1Loss as SL1L
from sklearn.utils.extmath import cartesian

from logger import get_logger
logger = get_logger("Losses logger")

import pdb

class MSELoss(nn.MSELoss):
    """
    Module for implementing an L2 loss on the location predictions.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        
        self.loss_fn = nn.MSELoss()
        self.device = device
        
    def forward(self, pred_maps, gt_locations_list):
        # Obtain the number of GT locations
        n_gt_locations = torch.tensor(
            [gt_locations.shape[0] for gt_locations in gt_locations_list], 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Initialize the GT heatmaps
        gt_maps = torch.zeros(size=pred_maps.shape, dtype=torch.float32, device=self.device)
        
        # Add the GT points to each heatmap
        for i in range(len(gt_locations_list)):
            if n_gt_locations[i] > 0:
                gt_locations = gt_locations_list[i]
                # NOTE: gt_locations have the following format: (x, y).
                # Thus, the 1st term is responsible for the column index, while the 2nd term is responsible
                # for the row index. Hence, gt_locations[:, 1], gt_locations[:, 0] instead of 
                # gt_locations[:, 0], gt_locations[:, 1]
                gt_maps[i][gt_locations[:, 1], gt_locations[:, 0]] = 1.
        
        # Compute the weighted loss
        diff = gt_maps - pred_maps
        loss = torch.norm(diff, p=2) + torch.norm(diff * gt_maps * 1000, p=2)
        
        return loss

class SmoothL1Loss(nn.SmoothL1Loss):
    """
    Module for implementing the smooth L1 loss on the predicted number of objects.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        
        self.loss_fn = SL1L()
        self.device = device
        
    def forward(self, pred_count, gt_count):
        return self.loss_fn(pred_count, gt_count)
        
class WeightedHausdorffDistance(nn.Module):
    def __init__(self,
                 resized_height, resized_width,
                 p=-9,
                 return_2_terms=False,
                 device=torch.device('cpu')):
        """
        :param resized_height: Number of rows in the image.
        :param resized_width: Number of columns in the image.
        :param p: Exponent in the generalized mean. -inf makes it the minimum.
        :param return_2_terms: Whether to return the 2 terms
                               of the WHD instead of their sum.
                               Default: False.
        :param device: Device where all Tensors will reside.
        """
        super().__init__()

        # Prepare all possible (row, col) locations in the image
        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)
        self.max_dist = math.sqrt(resized_height**2 + resized_width**2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))
        # Convert to appropiate type
        self.all_img_locations = self.all_img_locations.to(device=device,
                                                           dtype=torch.get_default_dtype())

        self.return_2_terms = return_2_terms
        self.p = p
        self.device = device

    def forward(self, prob_map, gt, orig_sizes):
        """
        Compute the Weighted Hausdorff Distance function
        between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.
        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :param orig_sizes: Bx2 Tensor containing the size
                           of the original images.
                           B is batch size.
                           The size must be in (height, width) format.
        :param orig_widths: List of the original widths for each image
                            in the batch.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                 If self.return_2_terms=True, then return a tuple containing
                 the two terms of the Weighted Hausdorff Distance.
        """

        self._assert_no_grad(gt)

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s'\
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)

        terms_1 = []
        terms_2 = []
        for b in range(batch_size):

            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b]
            orig_size_b = orig_sizes[b, :]
            norm_factor = (orig_size_b/self.resized_size).unsqueeze(0)
            n_gt_pts = gt_b.size()[0]

            # Corner case: no GT points
            # if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0:
            if n_gt_pts == 0:
                terms_1.append(torch.tensor(0,
                                            dtype=torch.get_default_dtype(),
                                           device=self.device))
                terms_2.append(torch.tensor(self.max_dist,
                                            dtype=torch.get_default_dtype(),
                                           device=self.device))
                continue

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1) *\
                self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1)*gt_b
            d_matrix = self.cdist(normalized_x, normalized_y)

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + 1e-6)) * \
                torch.sum(p * torch.min(d_matrix, 1)[0])
            weighted_d_matrix = (1 - p_replicated)*self.max_dist + p_replicated*d_matrix
            minn = self.generaliz_mean(weighted_d_matrix,
                                  p=self.p,
                                  dim=0, keepdim=False)
            term_2 = torch.mean(minn)

            # terms_1[b] = term_1
            # terms_2[b] = term_2
            terms_1.append(term_1)
            terms_2.append(term_2)
        
        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        if self.return_2_terms:
            res = terms_1.mean(), terms_2.mean()
        else:
            res = terms_1.mean() + terms_2.mean()

        return res
    
    def _assert_no_grad(self, variables):
        for var in variables:
            assert not var.requires_grad, \
                "nn criterions don't compute the gradient w.r.t. targets - please " \
                "mark these variables as volatile or not requiring gradients"
    
    def cdist(self, x, y):
        """
        Compute distance between each pair of the two collections of inputs.
        :param x: Nxd Tensor
        :param y: Mxd Tensor
        :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
              i.e. dist[i,j] = ||x[i,:]-y[j,:]||
        """
        differences = x.unsqueeze(1) - y.unsqueeze(0)
        distances = torch.sum(differences**2, -1).sqrt()
        return distances
    
    def generaliz_mean(self, tensor, dim, p=-9, keepdim=False):
        # """
        # Computes the softmin along some axes.
        # Softmin is the same as -softmax(-x), i.e,
        # softmin(x) = -log(sum_i(exp(-x_i)))

        # The smoothness of the operator is controlled with k:
        # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

        # :param input: Tensor of any dimension.
        # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
        # :param keepdim: (bool) Whether the output tensor has dim retained or not.
        # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
        # """
        # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
        """
        The generalized mean. It corresponds to the minimum when p = -inf.
        https://en.wikipedia.org/wiki/Generalized_mean
        :param tensor: Tensor of any dimension.
        :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
        :param keepdim: (bool) Whether the output tensor has dim retained or not.
        :param p: (float<0).
        """
        assert p < 0
        res= torch.mean((tensor + 1e-6)**p, dim, keepdim=keepdim)**(1./p)
        return res