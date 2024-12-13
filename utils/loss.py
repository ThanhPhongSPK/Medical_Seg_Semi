import torch 
import torch.nn as nn 

class DiceLoss(nn.Module): 
    def __init__(self, n_classes): 
        super(DiceLoss, self).__init__() 
        self.n_classes = n_classes
    
    def _one_hot_encoder(self, input_tensor): # torch.nn.functional.one_hot()
        """
        Apply one-hot encoder for input_tensor 
        Parameters: 
            - input_tensor.shape = (batchsize,1, H, W), the target image
        """
        tensor_list = [] 
        for i in range(self.n_classes): 
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim= 1)
        return output_tensor.float() 
    
    def _dice_loss(self, score, target): 
        target = target.float() 
        smooth = 1e-10 
        
        intersection = torch.sum(score * target)
        union = torch.sum(score* score) + torch.sum(target*target)
        dice = ( 2*intersection + smooth) / (union + smooth)
        loss = 1 - dice 
        return loss 
    
    def _dice_mask_loss(self, score, target, mask): 
        target = target.float() 
        mask = mask.float() 
        smooth = 1e-10 

        intersection = torch.sum(score * target * mask)
        union = torch.sum(score * score * mask ) + torch.sum(target * target * mask)
        dice = (2*intersection + smooth) / (union + smooth)
        loss = 1 - dice 
        return loss 

    def forward(self, inputs, target, mask= None, weight= None, softmax= False): 
        if softmax: 
            inputs = torch.softmax(inputs, dim= 1) 
        
        target = self._one_hot_encoder(target)

        # weight 
        if weight is  None: 
            weight = [1] * self.n_classes
        
        assert inputs.size() == target.size(), 'predict and target shape do not match'
        class_wise_dice = [] 
        loss = 0.0 
        if mask is not None: 
            mask = mask.repeat(1, self.n_classes, 1, 1).type(torch.float32)
            for i in range(0, self.n_classes): 
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append( 1.0 - dice.item())
                loss += dice * weight[i]

        else: 
            for i in range(0, self.n_classes): 
                dice = self._dice_loss(inputs[:, i], target[:, i]) 
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i] 
        
        return loss / self.n_classes