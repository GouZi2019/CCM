import SimpleITK as sitk
import numpy as np

from . import framework as methods

class Assessment(object):

    def __init__(self, registration: methods.Registration = None, 
                 fixed_img: sitk.Image=None, fixed_seg_list=[], fixed_points=[],
                 moving_img: sitk.Image=None, moving_seg_list=[], moving_points=[],
                 mask: sitk.Image=None, gt_field:sitk.Image=None, default: float=0.0) -> None:
        
        if registration is not None:
            self.fixed_img = registration.fixed
            self.fixed_seg_list = fixed_seg_list
            self.fixed_points = fixed_points
            
            self.moving_img = registration.moving
            self.moving_seg_list = moving_seg_list
            self.moving_points = moving_points
            
            self.gt_field = gt_field
            self.default = default
            self.mask = self.__CheckMaskValid(mask)
            self.SetField(registration.field)
            
            self.time = registration.time
        else:      
            self.fixed_img = fixed_img
            self.fixed_seg_list = fixed_seg_list
            self.fixed_points = fixed_points
            
            self.moving_img = moving_img
            self.moving_seg_list = moving_seg_list
            self.moving_points = moving_points
            
            self.gt_field = gt_field
            self.default = default
            self.mask = self.__CheckMaskValid(mask)
            
            self.time = []
            
        
    def SetField(self, field: sitk.Image=None) -> None:
        
        self.field = sitk.Cast(field, sitk.sitkVectorFloat64)
        self.trans = sitk.DisplacementFieldTransform(sitk.Image(self.field))
    
        # warp moving image
        self.warped_img = sitk.Resample(self.moving_img, self.fixed_img, self.trans, sitk.sitkBSplineResamplerOrder1, self.default)
        
        # warp moving segmentations
        self.warped_seg_list = []
        for seg in self.moving_seg_list:
            warped_seg = sitk.Resample(seg, self.fixed_img, self.trans, sitk.sitkNearestNeighbor)
            self.warped_seg_list.append(warped_seg)
            
        # transform points
        self.transformed_fixed_points = []
        for pf in self.fixed_points:
            transformed_pf = self.trans.TransformPoint(np.array(pf, dtype=np.float64))
            self.transformed_fixed_points.append(transformed_pf)
        
        # measure jacobian
        self.jac = sitk.DisplacementFieldJacobianDeterminant(self.field)
        
            
    
    def AssessRegistration(self, field: sitk.Image=None, output_prefix=None) -> dict:
        
        if field is not None:            
            self.SetField(field)
            
        mse_mean, mse_std, mse_num = self.GetMSE()
        jac_mean, jac_std, jac_num = self.GetJac()
        tre_mean, tre_std, tre_num = self.GetTRE()
        err_mean, err_std, err_num = self.GetError()
        SDlogJ = self.GetSDlogJ()
        dice_list = self.GetDICE()
        hdd_list = self.GetHDD()
        
        if output_prefix is not None:
            sitk.WriteImage(self.warped_img, output_prefix+'img.mha', True)
            sitk.WriteImage(self.jac, output_prefix+'jac.mha', True)
            for i in range(len(self.warped_seg_list)):
                sitk.WriteImage(self.warped_seg_list[i], output_prefix+f'seg{i}.mha', True)
            if self.transformed_fixed_points:
                np.savetxt(output_prefix+'points.txt', np.array(self.transformed_fixed_points))
        
        results = {'mse': [mse_mean, mse_std, mse_num],
                   'jac': [jac_mean, jac_std, jac_num],
                   'tre': [tre_mean, tre_std, tre_num],
                   'err': [err_mean, err_std, err_num],
                   'SDlogJ': SDlogJ,
                   'dice': dice_list,
                   'hdd': hdd_list,
                   'time': self.time
                   }
        return results
        
    def __CheckMaskValid(self, mask: sitk.Image=None):
        
        if mask is None:
            mask = sitk.Image(self.fixed_img.GetSize(), sitk.sitkUInt16)
            mask.CopyInformation(self.fixed_img)
            mask = sitk.Add(mask, 1)
        else:
            mask = mask > 0

        return mask
    
        
    def __GetLabelStatisticsFromImage(self, img):
        
        filter = sitk.LabelStatisticsImageFilter()
        filter.Execute(img, self.mask)
        
        return filter.GetMean(1), filter.GetSigma(1), filter.GetCount(1)        

    def GetSDlogJ(self):
        jac_array = sitk.GetArrayFromImage(self.jac).clip(0.000000001, 1000000000)
        logJ_array = np.log(jac_array)
        SDlogJ = np.ma.MaskedArray(logJ_array, 1-sitk.GetArrayFromImage(self.mask)).std()
        return SDlogJ
        
    def GetMSE(self):
        diff_img = sitk.Square(self.fixed_img - self.warped_img) 
        return self.__GetLabelStatisticsFromImage(diff_img)
    
    def GetJac(self):
        return self.__GetLabelStatisticsFromImage(self.jac)
        
    def GetError(self):
        if self.gt_field is None:
            return [-1,-1,-1]
        
        diff_field = sitk.VectorMagnitude(self.gt_field - self.field)
        return self.__GetLabelStatisticsFromImage(diff_field)
    
    def GetTRE(self):
        if len(self.fixed_points) and len(self.moving_points):

            pm = np.array(self.moving_points, dtype=np.float64)
            transformed_pf = np.array(self.transformed_fixed_points, dtype=np.float64)
            diff_points = pm - transformed_pf
            diff_norm = np.linalg.norm(diff_points, axis=1)
            
            self.tre_array = diff_norm
            return np.mean(diff_norm), np.std(diff_norm), len(diff_norm)
        else:
            return [-1,-1,-1]
        
    def GetDICE(self, index=None):

        if index:
            diceFilter = sitk.LabelOverlapMeasuresImageFilter()
            diceFilter.Execute(self.fixed_seg_list[index], self.warped_seg_list[index])
            return diceFilter.GetDiceCoefficient()
        else:
            DICE_list = []
            for f_seg, m_seg in zip(self.fixed_seg_list, self.warped_seg_list):
                diceFilter = sitk.LabelOverlapMeasuresImageFilter()
                diceFilter.Execute(f_seg, m_seg)
                DICE_list.append(diceFilter.GetDiceCoefficient())
            return DICE_list


    def GetHDD(self, index=None):

        if index:
            hausdorffFilter = sitk.HausdorffDistanceImageFilter()
            hausdorffFilter.Execute(self.fixed_seg_list[index], self.warped_seg_list[index])
            return hausdorffFilter.GetHausdorffDistance()
        else:
            HDD_list = []
            for f_seg, m_seg in zip(self.fixed_seg_list, self.warped_seg_list):
                hausdorffFilter = sitk.HausdorffDistanceImageFilter()
                hausdorffFilter.Execute(f_seg, m_seg)
                HDD_list.append(hausdorffFilter.GetHausdorffDistance())
            return HDD_list
