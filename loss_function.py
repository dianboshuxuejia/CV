import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import Config

class LossFun(nn.Module):
    def __init__(self):
        super(LossFun,self).__init__()
    def forward(self, prediction,targets,priors_boxes):
        loc_data , conf_data = prediction
        loc_data = torch.cat([o.view(o.size(0),-1,4) for o in loc_data] ,1)
        conf_data = torch.cat([o.view(o.size(0),-1,Config.class_num) for o in conf_data],1)
        priors_boxes = torch.cat([o.view(-1,4) for o in priors_boxes],0)
        if Config.use_cuda:
            loc_data = loc_data.cuda()
            conf_data = conf_data.cuda()
            priors_boxes = priors_boxes.cuda()
        # batch_size
        batch_num = loc_data.size(0)
        # default_box number
        box_num = loc_data.size(1)
        # store targets according to each prior_box date after transformation
        target_loc = torch.Tensor(batch_num,box_num,4)
        target_loc.requires_grad_(requires_grad=False)
        # store each type of prediction of default_box
        target_conf = torch.LongTensor(batch_num,box_num)
        target_conf.requires_grad_(requires_grad=False)
        if Config.use_cuda:
            target_loc = target_loc.cuda()
            target_conf = target_conf.cuda()
        # Since there may be multiple graphs in a batch, each loop computes the loc and conf of one box in the graph, i.e. 8732 boxes, which are stored in target_loc and target_conf
        for batch_id in range(batch_num):
            target_truths = targets[batch_id][:,:-1].data
            target_labels = targets[batch_id][:,-1].data
            if Config.use_cuda:
                target_truths = target_truths.cuda()
                target_labels = target_labels.cuda()
            # Calculate the box function, i.e. the formula for the loc loss function in Eq.
            utils.match(0.5,target_truths,priors_boxes,target_labels,target_loc,target_conf,batch_id)
        pos = target_conf > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # Equivalent to the operation of multiplying xij by the L1 loss function in the paper
        pre_loc_xij = loc_data[pos_idx].view(-1,4)
        tar_loc_xij = target_loc[pos_idx].view(-1,4)
        # Smooth_li loss function by taking the calculated loc and prediction
        loss_loc = F.smooth_l1_loss(pre_loc_xij,tar_loc_xij,size_average=False)

        batch_conf = conf_data.view(-1,Config.class_num)

        # Referring to the conf calculation in the paper, find the ci
        loss_c = utils.log_sum_exp(batch_conf) - batch_conf.gather(1, target_conf.view(-1, 1))

        loss_c = loss_c.view(batch_num, -1)
        # Set positive sample to 0
        loss_c[pos] = 0

        # Sort the remaining negative samples and select the target number of negative samples
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3*num_pos, max=pos.size(1)-1)

        # Extraction of positive and negative samples
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, Config.class_num)
        targets_weighted = target_conf[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        N = num_pos.data.sum().double()
        loss_l = loss_loc.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
