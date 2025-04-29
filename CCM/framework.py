import SimpleITK as sitk
import numpy as np
import os
import copy
from markupsafe import string
import matplotlib.pyplot as plt
import time 
import matlab.engine
import cv2

from .util import random_warper as warp
from . import magnetic as mag
from . import gpu_tps

class Registration(object):

    detectorSIFT = 'SIFT'
    detectorSURF = 'SURF'
    detectorFAST = 'FAST'
    detectorBRISK = 'BRISK'
    detectorMSER = 'MSER'
    detectorORB = 'ORB'
    detectorHarris = 'Harris'
    detectorCANNY = 'CANNY'
    
    descriptorSIFT = 'SIFT'
    descriptorSURF = 'SURF'
    descriptorBRISK = 'BRISK'
    descriptorORB = 'ORB'

    def __init__(self, warper: warp.ImageWarper=None, 
                 fixed: sitk.Image=None, moving: sitk.Image=None, initial_trans:sitk.CompositeTransform=None, 
                 default_value=0, debug=False, threads=12, matlab_eng=None,
                 fixed_mask: sitk.Image=None, moving_mask: sitk.Image=None, create_mask=False):
        """测试一下提示"""

        if warper:
            self.fixed = warper.img0
            self.moving = warper.img
            self.seg_list = warper.warped_seg_list
            self.warper = copy.copy(warper)
        else:
            self.fixed = fixed
            self.moving = moving
            self.seg_list = []
            self.warper = None

        if self.fixed.GetNumberOfComponentsPerPixel()==3:
            self.fixed = self.__rgb2gray(self.fixed)

        if self.moving.GetNumberOfComponentsPerPixel()==3:
            self.moving = self.__rgb2gray(self.moving)

        self.dimension = self.fixed.GetDimension()

        # feature point list
        self.pfs = []
        self.pms = []

        # background value, default is 0
        self.default = default_value

        # debug or not
        self.debug = debug

        # multi threads setting
        self.threads = threads
        
        # time record
        self.time = []

        # matlab eng (really slow to start)
        self.eng = matlab_eng

        # create mask or not
        self.create_mask = create_mask
        
        # route merge pfs
        self.route_merge_pfs = []
        
        if fixed_mask is not None:
            self.fixed_mask = fixed_mask>0
        elif create_mask:
            self.fixed_mask = self.__CreateFeatureMask(self.fixed)
        else:
            self.fixed_mask = sitk.Image(self.fixed.GetSize(), sitk.sitkUInt8) + 1
            self.fixed_mask.CopyInformation(self.fixed)

        if moving_mask is not None:
            self.moving_mask = moving_mask>0
        elif create_mask:
            self.moving_mask = self.__CreateFeatureMask(self.moving)
        else:
            self.moving_mask = sitk.Image(self.moving.GetSize(), sitk.sitkUInt8) + 1
            self.moving_mask.CopyInformation(self.moving)
        
        self.trans = sitk.CompositeTransform(self.dimension)
        self.field = sitk.TransformToDisplacementField(self.trans, size=self.fixed.GetSize(), outputOrigin=self.fixed.GetOrigin(), outputSpacing=self.fixed.GetSpacing(), outputDirection=self.fixed.GetDirection())

        if initial_trans:
            self.__WelcomeNewTransformation(initial_trans)
        else:
            self.warped = self.moving
            self.warped_mask = self.moving_mask
            self.warped_seg_list = self.seg_list

    def __rgb2gray(self, image):
        # Convert RGB image to gray scale and rescale results to [0,255]
        channels = [sitk.VectorIndexSelectionCast(image,i, sitk.sitkFloat32) for i in range(image.GetNumberOfComponentsPerPixel())]
        #linear mapping
        I = 0.2126*channels[0] + 0.7152*channels[1] + 0.0722*channels[2]
        return sitk.Cast(I, sitk.sitkFloat32)

    def __CreateFeatureMask(self, img):

        threshold = sitk.TriangleThreshold(img)
        mask = sitk.BinaryFillhole(sitk.Not(threshold))
        mask.CopyInformation(img)
        
        return sitk.Cast(mask, sitk.sitkUInt8)

    def __WelcomeNewTransformation(self, transform):
        self.trans.AddTransform(transform)
        self.field = sitk.TransformToDisplacementField(self.trans, size=self.fixed.GetSize(), outputOrigin=self.fixed.GetOrigin(), outputSpacing=self.fixed.GetSpacing(), outputDirection=self.fixed.GetDirection())

        self.warped = sitk.Resample(self.moving, self.fixed, self.trans, defaultPixelValue=self.default)
        self.warped_mask = sitk.Resample(self.moving_mask, self.fixed, self.trans, interpolator=sitk.sitkNearestNeighbor)

        warped_seg_list = []
        for seg in self.seg_list:
            warped_seg = sitk.Resample(seg, self.fixed, self.trans, sitk.sitkNearestNeighbor)
            warped_seg_list.append(warped_seg)
        self.warped_seg_list = warped_seg_list

    def __MultiResolutionRegistration(self, methods=[], **params):
        
        t_start = time.time()
        
        if self.debug:
            os.makedirs('temp', exist_ok=True)
                
        for _level in range(len(params['nlevel'])):
            level = params['nlevel'][_level]
            shrink_factor = level
            smooth_factor = 0

            _fixed = self.resize_smooth_image(self.fixed, shrink_factor, smooth_factor)
            _moving = self.resize_smooth_image(self.warped, shrink_factor, smooth_factor)
            
            if self.fixed_mask is not None:
                _fixed_mask = self.resize_smooth_image(self.fixed_mask, shrink_factor, 0, sitk.sitkNearestNeighbor)
            if self.warped_mask is not None:
                _moving_mask = self.resize_smooth_image(self.warped_mask, shrink_factor, 0, sitk.sitkNearestNeighbor)
            
            if self.debug:
                sitk.WriteImage(_fixed, f'temp/{level}_fixed.mha', True)
                sitk.WriteImage(_moving, f'temp/{level}_moving.mha', True)
                if self.fixed_mask is not None:
                    sitk.WriteImage(_fixed_mask, f'temp/{level}_fixed_mask.mha', True)
                if self.warped_mask is not None:
                    sitk.WriteImage(_moving_mask, f'temp/{level}_moving_mask.mha', True)   
            
            if self.debug:
                print('Level: ', level)
            
            # Perform functions
            for method in methods:
                if method == 'ccm':
                    _field = self.__CCMRegistrationFunction(_level, _fixed, _moving, _fixed_mask, _moving_mask, params)
                if method in ['rigid', 'affine', 'bspline']:
                    _field = self.__ElastixRegistrationFunction(method, _level, _fixed, _moving, _fixed_mask, _moving_mask, params)
                    
                _trans = sitk.DisplacementFieldTransform(sitk.Image(_field))
                self.__WelcomeNewTransformation(_trans)
                    
                # warping moving image
                _moving = self.resize_smooth_image(self.warped, shrink_factor, smooth_factor)
                
                if self.debug:
                    sitk.WriteImage(sitk.Cast(_field, sitk.sitkVectorFloat32), f'temp/{level}_{method}_field.mha', True)
                    sitk.WriteImage(sitk.DisplacementFieldJacobianDeterminant(_field), f'temp/{level}_{method}_jac.mha', True)
                    sitk.WriteImage(_moving, f'temp/{level}_{method}_warped.mha', True)

        t_end = time.time()
        self.time.append(t_end - t_start)
        
        if self.debug:
            print('Use time：', t_end - t_start, 's')

    def GetRegistrationResult(self):
        return [self.field, self.time]
    
    def OutputRegistrationResult(self, output_prefix):
        sitk.WriteImage(self.warped, output_prefix+'image.mha', True)
        sitk.WriteImage(sitk.Cast(self.field, sitk.sitkVectorFloat32), output_prefix+'field.mha', True)
        
        time_list = [str(x) for x in self.time]
        with open(f'{output_prefix}time.txt', 'w') as file:
            file.write(' '.join(time_list))

    def resize_smooth_image(self, image, shrink_factors, smoothing_sigmas, interpolator=sitk.sitkLinear):
        """
        Args:
            image: The image we want to resample.
            shrink_factor(s): Number(s) greater than one, such that the new image's size is original_size/shrink_factor.
            smoothing_sigma(s): Sigma(s) for Gaussian smoothing, this is in physical units, not pixels.
        Return:
            Image which is a result of smoothing the input and then resampling it using the given sigma(s) and shrink factor(s).
        """
        if np.isscalar(shrink_factors):
            shrink_factors = [shrink_factors]*image.GetDimension()

        if smoothing_sigmas == 0:
            smoothed_image = image
        else:
            smoothing_sigmas = [smoothing_sigmas]*image.GetDimension()
            smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigmas)
        
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [int(sz/float(sf) + 0.5) for sf,sz in zip(shrink_factors,original_size)]
        new_spacing = [((original_sz-1)*original_spc)/(new_sz-1) 
                    for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
        return sitk.Resample(smoothed_image, new_size, sitk.Transform(), 
                            interpolator, image.GetOrigin(),
                            new_spacing, image.GetDirection(), self.default, 
                            image.GetPixelID())

    def __add_colorbar(self, mappable):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        last_axes = plt.gca()
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        plt.sca(last_axes)
        return cbar

    def ShowResults(self, axis=2, slice=0, field_sub_ratio=20):

        __, ax = plt.subplots(1,5, figsize=(30,60))
        
        if self.dimension == 2:
            show_moving = sitk.GetArrayFromImage(self.moving)
            show_fixed = sitk.GetArrayFromImage(self.fixed)
            show_warped = sitk.GetArrayFromImage(self.warped)
            show_field = sitk.GetArrayFromImage(self.field)[::field_sub_ratio, ::field_sub_ratio, :]
            show_jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(self.field))
        elif self.dimension == 3:
            spacing_ratio = int(field_sub_ratio / (self.fixed.GetSpacing()[2]/self.fixed.GetSpacing()[0]))
            show_axis = 2 - axis
            show_moving = np.take(sitk.GetArrayFromImage(self.moving), axis=show_axis, indices=slice)
            show_fixed = np.take(sitk.GetArrayFromImage(self.fixed), axis=show_axis, indices=slice)
            show_warped = np.take(sitk.GetArrayFromImage(self.warped), axis=show_axis, indices=slice)
            show_field = np.take(sitk.GetArrayFromImage(self.field), axis=show_axis, indices=slice)[::field_sub_ratio, ::field_sub_ratio, :]
            show_jac = np.take(sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(self.field)), axis=show_axis, indices=slice)        

        ax[0].set_title('moving')
        ax[0].imshow(show_moving, cmap='gray')
        ax[0].set_axis_off()

        ax[1].set_title('fixed')
        ax[1].imshow(show_fixed, cmap='gray')
        ax[1].set_axis_off()

        ax[2].set_title('warped')
        ax[2].imshow(show_warped, cmap='gray')
        ax[2].set_axis_off()

        ax[3].set_title('field')
        ax[3].set_aspect('equal')
        if self.field.GetNumberOfComponentsPerPixel() == 2:
            ax[3].quiver(show_field[..., 0], show_field[..., 1], angles='xy')
        else:
            if axis ==0:
                ax[3].quiver(show_field[..., 1], show_field[..., 2])
            elif axis ==1:
                ax[3].quiver(show_field[..., 0], show_field[..., 2])
            elif axis ==2:
                ax[3].quiver(show_field[..., 1], show_field[..., 0])
        ax[3].invert_yaxis()
        ax[3].set_axis_off()

        ax[4].set_title('jac')
        jac = ax[4].imshow(show_jac, vmin=0, vmax=2, cmap='jet')
        ax[4].set_axis_off()
        self.__add_colorbar(jac)
        plt.show()

    ## rigid, affine, bspline
    def PerformRigidRegistration(self, nlevel=[4,2,1], niter=[200,100,50], step=1, sigma=16):
        self.__MultiResolutionRegistration(methods=['rigid'], nlevel=nlevel, niter=niter, step=step, sigma=sigma)
        return self

    def PerformAffineRegistration(self, nlevel=[4,2,1], niter=[200,100,50], step=1, sigma=16):
        self.__MultiResolutionRegistration(methods=['affine'], nlevel=nlevel, niter=niter, step=step, sigma=sigma)
        return self

    def PerformBSplineRegistration(self, nlevel=[4,2,1], niter=[200,100,50], step=1, sigma=16):
        self.__MultiResolutionRegistration(methods=['bspline'], nlevel=nlevel, niter=niter, step=step, sigma=sigma)
        return self

    # our method
    def PerformCCMRegistration(self, nlevel=[4,2,1], nmigr=[5,5,5], niter=[200,100,50], step=1, sigma=16,
            feature_detector=detectorFAST, feature_detector_size=7, feature_detector_angle=90, feature_detector_ratio=[3,2,1], feature_detector_num=[1000,1000,1000],
            feature_descriptor=descriptorBRISK, feature_descriptor_radius=[0,3,6,9], feature_descriptor_samplenum=[1,8,14,20], feature_descriptor_dmax=6, feature_descriptor_dmin=9,
            feature_match_ratio=1, feature_match_crosscheck=True,
            route_merge=True, route_check=True, route_stay=0.1, mask_valid=True, just_tps=False, tps_lambda=0.0,
            ccm_k=[0.1,0.1,0.1], ccm_z=[0.5,0.5,0.5], ccm_t=[0.5,0.5,0.5], ccm_e=[0.5,0.5,0.5], ccm_j=[0.5,0.5,0.5]):
        
        if self.eng is None:
            self.__CreateMatlabEngine()
            
        self.__MultiResolutionRegistration(methods=['ccm'], nlevel=nlevel, nmigr=nmigr, niter=niter, step=step, sigma=sigma,
                feature_detector=feature_detector, feature_detector_size=feature_detector_size, feature_detector_angle=feature_detector_angle, feature_detector_ratio=feature_detector_ratio, feature_detector_num=feature_detector_num,
                
                feature_descriptor=feature_descriptor, feature_descriptor_radius=feature_descriptor_radius, feature_descriptor_samplenum=feature_descriptor_samplenum, feature_descriptor_dmax=feature_descriptor_dmax, feature_descriptor_dmin=feature_descriptor_dmin,
                
                feature_match_ratio=feature_match_ratio, feature_match_crosscheck=feature_match_crosscheck, 
                
                route_merge=route_merge, route_check=route_check, route_stay=route_stay, mask_valid=mask_valid, just_tps=just_tps, tps_lambda=tps_lambda,
                ccm_k=ccm_k, ccm_z=ccm_z, ccm_t=ccm_t, ccm_e=ccm_e, ccm_j=ccm_j)
        return self

    def __CCMRegistrationFunction(self, level, fixed, moving, fixed_mask, moving_mask, params) -> sitk.Image:
        
        # unpack params: level
        schedule=params['nlevel'][level]
        
        # unpack params: feature detector, descriptor, matcher
        feature_detector=params['feature_detector']; feature_detector_size=params['feature_detector_size']; feature_detector_angle=params['feature_detector_angle']; feature_detector_ratio=params['feature_detector_ratio'][level]; feature_detector_num=params['feature_detector_num'][level]
        feature_descriptor=params['feature_descriptor']; feature_descriptor_radius=params['feature_descriptor_radius']; feature_descriptor_samplenum=params['feature_descriptor_samplenum']; feature_descriptor_dmax=params['feature_descriptor_dmax']; feature_descriptor_dmin=params['feature_descriptor_dmin']
        feature_match_ratio=params['feature_match_ratio']; feature_match_crosscheck=params['feature_match_crosscheck']
                        
        # unpack params: ccm
        n_migration=params['nmigr'][level];route_merge=params['route_merge']; route_check=params['route_check']; route_stay=params['route_stay']; mask_valid=params['mask_valid']
        ccm_k=params['ccm_k'][level]; ccm_z=params['ccm_z'][level]; ccm_t=params['ccm_t'][level]; ccm_e=params['ccm_e'][level]; ccm_j=params['ccm_j'][level]
        just_tps=params['just_tps']; tps_lambda=params['tps_lambda']

        # unpack params: bspline
        iter=params['niter'][level]
        
        _moving = moving; _moving_mask = moving_mask
        _trans = sitk.CompositeTransform(self.dimension)
        for i in range(n_migration):
            if self.debug:
                print(f'\n\nCCM: migration={i+1}')
                print('CCM Flight begin...')
                
            # TODO: 这里会反复提取固定图像的特征点，可以考虑提取一次，然后传递给下一次迭代
            _mp = mag.MagneticPoint(fixed=fixed, moving=_moving, debug=self.debug, fixed_mask=fixed_mask, moving_mask=_moving_mask, create_mask=self.create_mask, debug_prefix=f'{schedule}_{i+1}')
            _pf, _pm = _mp.GetCorrespondencePoints(
                feature_detector=feature_detector, feature_detector_size=feature_detector_size, feature_detector_angle=feature_detector_angle, feature_detector_ratio=feature_detector_ratio, feature_detector_num=feature_detector_num,
                
                feature_descriptor=feature_descriptor, feature_descriptor_radius=feature_descriptor_radius, feature_descriptor_samplenum=feature_descriptor_samplenum, feature_descriptor_dmax=feature_descriptor_dmax, feature_descriptor_dmin=feature_descriptor_dmin,
                
                feature_match_ratio=feature_match_ratio, feature_match_crosscheck=feature_match_crosscheck
                )

            if self.debug:
                if self.dimension==2:
                    _mp.ShowResults(_pf, _pm, number=range(50))
                print(f'Find {len(_pf)} correspondences.')
            
            if route_merge:
                _field, _route_stay_pfs, self.route_merge_pfs = _mp.GenerateFlightField(ccm_k, ccm_z, ccm_t, ccm_e, ccm_j, self.eng, route_check=route_check, route_merge_pfs=self.route_merge_pfs, mask_valid=mask_valid, just_tps=just_tps, tps_lambda=tps_lambda)
            else:
                _field, _route_stay_pfs, _ = _mp.GenerateFlightField(ccm_k, ccm_z, ccm_t, ccm_e, ccm_j, self.eng, route_check=route_check, mask_valid=mask_valid, just_tps=just_tps, tps_lambda=tps_lambda)
            
            # if non pfs in route stay, break; i>0 to guarantee at least one spread
            if i>0 and len(_route_stay_pfs) == 0:
                break
            
            _trans.AddTransform(sitk.DisplacementFieldTransform(_field))
            _moving = sitk.Resample(moving, fixed, _trans, defaultPixelValue=self.default)
            _moving_mask = sitk.Resample(moving_mask, fixed, _trans, interpolator=sitk.sitkNearestNeighbor)

            if self.debug:
                sitk.WriteImage(_moving, f'temp/{schedule}_{i+1}_ccm_warped.mha')
                sitk.WriteImage(_moving_mask, f'temp/{schedule}_{i+1}_ccm_warped_mask.mha')
                if route_merge:
                    print('Route merge points:', len(self.route_merge_pfs))
                if route_stay:
                    print('Route stay points:', len(_route_stay_pfs))
            
            if iter>0:
                if self.debug:
                    print('\nDiffusion Begin!')
                _field = self.__ElastixRegistrationFunction('bspline', level, fixed, _moving, fixed_mask, _moving_mask, params,
                                                            pfs=_route_stay_pfs, pms=_route_stay_pfs, points_metric_ratio=route_stay
                                                            )

                _trans.AddTransform(sitk.DisplacementFieldTransform(_field))
                _moving = sitk.Resample(moving, fixed, _trans, defaultPixelValue=self.default)
                _moving_mask = sitk.Resample(moving_mask, fixed, _trans, interpolator=sitk.sitkNearestNeighbor)

                if self.debug:
                    sitk.WriteImage(_moving, f'temp/{schedule}_{i+1}_diffusion_warped.mha')
                    sitk.WriteImage(_moving_mask, f'temp/{schedule}_{i+1}_diffusion_warped_mask.mha')
            
        field = sitk.TransformToDisplacementField(_trans, size=self.fixed.GetSize(), outputOrigin=self.fixed.GetOrigin(), outputSpacing=self.fixed.GetSpacing(), outputDirection=self.fixed.GetDirection())
        return field
    
    def __CreateMatlabEngine(self):
        # really slow..
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(r'..\CCM', nargout=0)
        self.eng.addpath(r'..\CCM\util', nargout=0)

    def __ElastixRegistrationFunction(self, method, level, fixed, moving, fixed_mask, moving_mask, params, pfs=[], pms=[], points_metric_ratio=0) -> sitk.Image:
        # unpack params: elastix
        iter=params['niter'][level]; step=params['step']; sigma=params['sigma']; schedule=params['nlevel'][level]
        
        parameter_map = self.__GetParameterMap(method)
        parameter_map['MaximumNumberOfIterations'] = [str(iter)]
        parameter_map['MaximumStepLength'] = [str(step)]
        parameter_map['FinalGridSpacingInPhysicalUnits'] = [str(sigma*schedule)]

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed)
        elastixImageFilter.SetMovingImage(moving)
        elastixImageFilter.SetFixedMask(fixed_mask)
        elastixImageFilter.SetMovingMask(moving_mask)
        elastixImageFilter.SetParameterMap(parameter_map)
        elastixImageFilter.SetNumberOfThreads(self.threads)

        if len(pfs) and len(pms) and points_metric_ratio > 0:
            image_metric_ratio = 1 - points_metric_ratio
            
            pts_f = self.__writeElxPointSetsFile(pfs,'f')
            pts_m = self.__writeElxPointSetsFile(pms,'m')
            elastixImageFilter.SetParameter( "Registration", "MultiMetricMultiResolutionRegistration" )
            elastixImageFilter.AddParameter( "Metric", "CorrespondingPointsEuclideanDistanceMetric" )
            elastixImageFilter.AddParameter( "Metric0Weight", str(image_metric_ratio) )
            elastixImageFilter.AddParameter( "Metric1Weight", str(points_metric_ratio) )
            
            elastixImageFilter.SetFixedPointSetFileName(pts_f)
            elastixImageFilter.SetMovingPointSetFileName(pts_m)

        if self.debug:
            elastixImageFilter.SetLogToFile(True)
            elastixImageFilter.SetLogFileName(f'temp/{schedule}_{method}_elastix.log')

        elastixImageFilter.Execute()

        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.ComputeDeformationFieldOn()
        transformixImageFilter.SetMovingImage(moving) # official bug
        transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.Execute()

        field = sitk.Cast(transformixImageFilter.GetDeformationField(), sitk.sitkVectorFloat64)
        return field
    
    def __writeElxPointSetsFile(self, points: np.ndarray, title: string):
        os.makedirs('temp', exist_ok=True)
        filename = 'temp/'+title+'.txt'

        f = open(filename, 'w')
        f.write('point\n')
        f.write(f'{points.shape[0]}\n')

        for point in points:
            f.write(np.array2string(point)[1:-1]+'\n')

        f.close()
        return filename

    def __GetParameterMap(self, transform_type):
        parameterMap = sitk.ParameterMap()

        parameterMap['Registration'] = ['MultiResolutionRegistration']
        parameterMap['Optimizer'] = ["AdaptiveStochasticGradientDescent"]
        parameterMap['AutomaticParameterEstimation'] = ["true"]
        parameterMap['NumberOfGradientMeasurements'] = ["10"]
        parameterMap['Metric'] = ["AdvancedNormalizedCorrelation"]

        parameterMap['NumberOfResolutions'] = ['1']
        parameterMap['FixedImagePyramid'] = ["FixedRecursiveImagePyramid"]
        parameterMap['MovingImagePyramid'] = ["MovingRecursiveImagePyramid"]

        parameterMap['DefaultPixelValue'] = [f'{self.default}']
        
        # For reproducibility:
        # Option1: set the random seed, however, failed. This is a bug in SimpleElastix: seed setting is no work.
        # As reported in issue #122: https://github.com/SuperElastix/SimpleElastix/issues/122
        parameterMap['ImageSampler'] = ["RandomCoordinate"]
        parameterMap['RandomSeed'] = ["121212"]
        parameterMap['NumberOfSpatialSamples'] = ['3000']
        parameterMap['NewSamplesEveryIteration'] = ["true"]
        
        # # Option2: use the Grid sampler
        # parameterMap['ImageSampler'] = ["Grid"]

        parameterMap['CheckNumberOfSamples'] = ["false"]
        parameterMap['Resampler'] = ["DefaultResampler"]
        parameterMap['HowToCombineTransforms'] = ["Compose"]
        parameterMap['Interpolator'] = ["BSplineInterpolator"]
        parameterMap['BSplineInterpolationOrder'] = ["2"]

        parameterMap['ResampleInterpolator'] = ["FinalBSplineInterpolator"]
        parameterMap['FinalBSplineInterpolationOrder'] = ["3"]
        parameterMap['WriteIterationInfo'] = ["false"]
        parameterMap['WriteResultImage'] = ["false"]
        
        parameterMap['ErodeMask'] = ["false"]

        if transform_type=='bspline':
            parameterMap['Transform'] = ["BSplineTransform"]
        if transform_type=='rigid':
            parameterMap['Transform'] = ["EulerTransform"]
            parameterMap['AutomaticScalesEstimation'] = ['true']            
            parameterMap['AutomaticTransformInitialization'] = ['true']            
        if transform_type=='affine':
            parameterMap['Transform'] = ["AffineTransform"]
            parameterMap['AutomaticScalesEstimation'] = ['true']            
            parameterMap['AutomaticTransformInitialization'] = ['true']         

        return parameterMap