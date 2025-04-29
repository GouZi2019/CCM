import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from skimage.morphology import convex_hull_image
from skimage.measure import ransac
from skimage.transform import AffineTransform
import cv2
import os

from .util import random_warper as fc
from . import gpu_tps

class MagneticPoint(object):

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
    
    def __init__(self, fixed: sitk.Image=None, moving: sitk.Image=None, gt_field: sitk.Image=None, transform: sitk.Transform=None, warper: fc.ImageWarper=None, 
                 pf: np.array=None, pm: np.array=None, debug=False, debug_prefix='',
                 create_mask=False, fixed_mask: sitk.Image=None, moving_mask: sitk.Image=None,
                 ) -> None:
        if fixed and moving:
            self.fixed = fixed
            self.moving = moving
            self.transform = sitk.Transform(self.fixed.GetDimension(), sitk.sitkIdentity) if transform is None else transform
            self.gt_field = gt_field
            self.warped = moving if transform is None else sitk.Resample(self.moving, self.fixed, self.transform)

        if warper:
            self.fixed = warper.img0
            self.moving = warper.img
            self.transform = sitk.Transform(self.fixed.GetDimension(), sitk.sitkIdentity)
            self.gt_field = warper.gt_field        
            self.warped = self.moving

        self.num_dimension = self.fixed.GetDimension()
        
        self.point_f = np.array(pf)
        self.point_m = np.array(pm)

        # 显示中间结果
        # show debug information
        self.debug = debug
        self.debug_prefix = debug_prefix

        if self.debug:
            if not os.path.exists('temp'):
                os.makedirs('temp')

        # 生成feature mask
        # create feature mask
        if fixed_mask is not None:
            self.fixed_mask = fixed_mask>0
        elif create_mask:
            self.fixed_mask = self.__CreateFeatureMask(self.fixed)
            if debug:
                sitk.WriteImage(self.fixed_mask, 'temp/fixed_mask.mha', True)
        else:
            self.fixed_mask = sitk.Image(self.fixed.GetSize(), sitk.sitkUInt8) + 1
            self.fixed_mask.CopyInformation(self.fixed)
            
        if moving_mask is not None:
            self.moving_mask = moving_mask>0
        elif create_mask:
            self.moving_mask = self.__CreateFeatureMask(self.moving)
            if debug:
                sitk.WriteImage(self.fixed_mask, 'temp/warped_mask.mha', True)
        else:
            self.moving_mask = sitk.Image(self.moving.GetSize(), sitk.sitkUInt8) + 1
            self.moving_mask.CopyInformation(self.moving)
        self.warped_mask = sitk.Resample(self.moving_mask, self.fixed, self.transform, interpolator=sitk.sitkNearestNeighbor)

    def __CreateFeatureMask(self, img):

        threshold = sitk.TriangleThreshold(img)
        mask = sitk.BinaryFillhole(sitk.Not(threshold))
        mask.CopyInformation(img)
        
        return sitk.Cast(mask, sitk.sitkUInt8)

    def __WarpImageAfterResample(self, field):
        clone_field = sitk.Image(field)
        transform = sitk.DisplacementFieldTransform(clone_field)
        composite_transform = sitk.CompositeTransform([self.transform, transform])
        return sitk.Resample(self.moving, self.fixed, composite_transform)

    def __WarpMaskAfterResample(self, field):
        clone_field = sitk.Image(field)
        transform = sitk.DisplacementFieldTransform(clone_field)
        composite_transform = sitk.CompositeTransform([self.transform, transform])
        return sitk.Resample(self.moving_mask, self.fixed, composite_transform, sitk.sitkNearestNeighbor)

    def GeneratePerfectMagneticPoint(self, number=2):
        
        if self.gt_field:
            
            size = self.gt_field.GetSize()

            if self.gt_field.GetDimension() == 2:

                ix_f = np.random.choice(size[0], number)
                iy_f = np.random.choice(size[1], number)

                point_f = np.zeros((number, 2))
                point_m = np.zeros((number, 2))

                delete_index = []
                for i in range(number):
                    point_f[i,:] = self.gt_field.TransformIndexToPhysicalPoint((int(ix_f[i]), int(iy_f[i])))
                    point_m[i,:] = point_f[i,:] + self.gt_field.GetPixel((int(ix_f[i]), int(iy_f[i])))

                    ix_m, iy_m = self.gt_field.TransformPhysicalPointToContinuousIndex(point_m[i,:])
                    if not (0<ix_m<size[0] and 0<iy_m<size[1]):
                        delete_index.append(i)

            elif self.gt_field.GetDimension() == 3:

                ix_f = np.random.choice(size[0], number)
                iy_f = np.random.choice(size[1], number)
                iz_f = np.random.choice(size[2], number)

                point_f = np.zeros((number, 3))
                point_m = np.zeros((number, 3))

                delete_index = []
                for i in range(number):
                    point_f[i,:] = self.gt_field.TransformIndexToPhysicalPoint((int(ix_f[i]), int(iy_f[i]), int(iz_f[i])))
                    point_m[i,:] = point_f[i,:] + self.gt_field.GetPixel((int(ix_f[i]), int(iy_f[i]), int(iz_f[i])))

                    ix_m, iy_m, iz_m = self.gt_field.TransformPhysicalPointToContinuousIndex(point_m[i,:])
                    if not (0<ix_m<size[0] and 0<iy_m<size[1] and 0<iz_m<size[2]):
                        delete_index.append(i)

            print('删除图像外金标准点对', len(delete_index),'对')
            print('delete gt correspondence outside image', len(delete_index),'pairs')
            self.point_f = np.delete(point_f, delete_index, axis=0)
            self.point_m = np.delete(point_m, delete_index, axis=0)                

            return self.point_f, self.point_m


    def GeneratePerfectMagneticPointWithNoise(self, gt_number=2, noise_number=2):
        
        if self.gt_field:
            total_number = gt_number+noise_number
            size = self.gt_field.GetSize()

            if self.gt_field.GetDimension() == 2:

                # 固定图像选若干个特征点
                # choose some feature points in fixed image
                ix_f = np.random.choice(size[0], total_number)
                iy_f = np.random.choice(size[1], total_number)

                # 加入噪声特征匹配点对
                # add noise feature matching points
                ix_m = np.random.choice(size[0], total_number)
                iy_m = np.random.choice(size[1], total_number)

                point_f = np.zeros((total_number, 2))
                point_m = np.zeros((total_number, 2))

                delete_index = [] # delete magnetic points outside the image
                for i in range(total_number):

                    point_f[i,:] = self.gt_field.TransformIndexToPhysicalPoint((int(ix_f[i]), int(iy_f[i])))
                    if i<noise_number:
                        point_m[i,:] = self.gt_field.TransformIndexToPhysicalPoint((int(ix_m[i]), int(iy_m[i])))                    
                    else:
                        point_m[i,:] = point_f[i,:] + self.gt_field.GetPixel((int(ix_f[i]), int(iy_f[i])))

                        check_ix_m, check_iy_m = self.gt_field.TransformPhysicalPointToContinuousIndex(point_m[i,:])
                        if not (0<check_ix_m<size[0] and 0<check_iy_m<size[1]):
                            delete_index.append(i)             

            elif self.gt_field.GetDimension() == 3:

                # 固定图像选若干个特征点
                # choose some feature points in fixed image
                ix_f = np.random.choice(size[0], total_number)
                iy_f = np.random.choice(size[1], total_number)
                iz_f = np.random.choice(size[2], total_number)

                # 加入噪声特征匹配点对
                # add noise feature matching points
                ix_m = np.random.choice(size[0], total_number)
                iy_m = np.random.choice(size[1], total_number)
                iz_m = np.random.choice(size[2], total_number)

                point_f = np.zeros((total_number, 3))
                point_m = np.zeros((total_number, 3))

                delete_index = []
                for i in range(total_number):

                    point_f[i,:] = self.gt_field.TransformIndexToPhysicalPoint((int(ix_f[i]), int(iy_f[i]), int(iz_f[i])))
                    if i<noise_number:
                        point_m[i,:] = self.gt_field.TransformIndexToPhysicalPoint((int(ix_m[i]), int(iy_m[i]), int(iz_m[i])))                    
                    else:
                        point_m[i,:] = point_f[i,:] + self.gt_field.GetPixel((int(ix_f[i]), int(iy_f[i]), int(iz_f[i])))

                        check_ix_m, check_iy_m, check_iz_m = self.gt_field.TransformPhysicalPointToContinuousIndex(point_m[i,:])
                        if not (0<check_ix_m<size[0] and 0<check_iy_m<size[1] and 0<check_iz_m<size[2]):
                            delete_index.append(i)


            print('删除图像外金标准点对', len(delete_index),'对')
            print('delete gt correspondence outside image', len(delete_index),'pairs')
            self.point_f = np.delete(point_f, delete_index, axis=0)
            self.point_m = np.delete(point_m, delete_index, axis=0)

            return self.point_f, self.point_m



    def GeneratePerfectMagneticPointWithNoiseAtEdge(self, gt_number=[50,50], noise_number=[50,50]):
        
        if self.gt_field:
            total_number = gt_number[0] + gt_number[1] + noise_number[0] + noise_number[1]
            noise_num = noise_number[0]+noise_number[1]

            size = self.gt_field.GetSize()

            if self.gt_field.GetDimension() == 2:

                # 先生成固定图像内部随机点
                # generate fixed image internal random points
                ix_f = np.random.choice(size[0], total_number)
                iy_f = np.random.choice(size[1], total_number)

                # 在边缘生成随机点：噪声
                # generate random points on the edge: noise
                for i in range(noise_number[0]):
                    # 随机选一个边缘
                    # randomly select an edge
                    random_edge = np.random.choice(4)
                    if random_edge == 0:
                        ix_f[i] = 0
                    elif random_edge == 1:
                        ix_f[i] = size[0]-1
                    elif random_edge == 2:
                        iy_f[i] = 0
                    else:
                        iy_f[i] = size[1]-1

                # 在边缘生成随机点：正确
                # generate random points on the edge: correct
                for i in range(gt_number[0]):
                    # 随机选一个边缘
                    # randomly select an edge
                    random_edge = np.random.choice(4)
                    if random_edge == 0:
                        ix_f[i+noise_num] = 0
                    elif random_edge == 1:
                        ix_f[i+noise_num] = size[0]-1
                    elif random_edge == 2:
                        iy_f[i+noise_num] = 0
                    else:
                        iy_f[i+noise_num] = size[1]-1

                # 加入噪声特征匹配点对
                # add noise feature matching points
                ix_m = np.random.choice(size[0], total_number)
                iy_m = np.random.choice(size[1], total_number)

                point_f = np.zeros((total_number, 2))
                point_m = np.zeros((total_number, 2))

                delete_index = []
                for i in range(total_number):

                    point_f[i,:] = self.gt_field.TransformIndexToPhysicalPoint((int(ix_f[i]), int(iy_f[i])))
                    if i<noise_num:
                        point_m[i,:] = self.gt_field.TransformIndexToPhysicalPoint((int(ix_m[i]), int(iy_m[i])))                    
                    else:
                        point_m[i,:] = point_f[i,:] + self.gt_field.GetPixel((int(ix_f[i]), int(iy_f[i])))

                        check_ix_m, check_iy_m = self.gt_field.TransformPhysicalPointToContinuousIndex(point_m[i,:])
                        if not (0<check_ix_m<size[0] and 0<check_iy_m<size[1]):
                            delete_index.append(i)        

            elif self.gt_field.GetDimension() == 3:

                # 先生成固定图像内部随机点
                # generate fixed image internal random points
                ix_f = np.random.choice(size[0], total_number)
                iy_f = np.random.choice(size[1], total_number)
                iz_f = np.random.choice(size[2], total_number)

                # 在边缘生成随机点：噪声
                # generate random points on the edge: noise
                for i in range(noise_number[0]):
                    # 选两个轴出来，来作为边缘
                    # randomly select two axes as edges
                    random_edge = np.random.choice(3, 2, replace=False)
                    for axis in random_edge:
                        first_or_end = np.random.choice(2)
                        if axis == 0:
                            ix_f[i] = size[0]-1 if first_or_end else 0
                        if axis == 1:
                            iy_f[i] = size[1]-1 if first_or_end else 0
                        if axis == 2:
                            iz_f[i] = size[2]-1 if first_or_end else 0

                # 在边缘生成随机点：正确
                # generate random points on the edge: correct
                for i in range(gt_number[0]):
                    # 选两个轴出来，来作为边缘
                    # randomly select two axes as edges
                    random_edge = np.random.choice(3, 2, replace=False)
                    for axis in random_edge:
                        first_or_end = np.random.choice(2)
                        if axis == 0:
                            ix_f[i+noise_num] = size[0]-1 if first_or_end else 0
                        if axis == 1:
                            iy_f[i+noise_num] = size[1]-1 if first_or_end else 0
                        if axis == 2:
                            iz_f[i+noise_num] = size[2]-1 if first_or_end else 0

                # 加入噪声特征匹配点对
                # add noise feature matching points
                ix_m = np.random.choice(size[0], total_number)
                iy_m = np.random.choice(size[1], total_number)
                iz_m = np.random.choice(size[2], total_number)

                point_f = np.zeros((total_number, 3))
                point_m = np.zeros((total_number, 3))

                delete_index = []
                for i in range(total_number):

                    point_f[i,:] = self.gt_field.TransformIndexToPhysicalPoint((int(ix_f[i]), int(iy_f[i]), int(iz_f[i])))
                    if i<noise_num:
                        point_m[i,:] = self.gt_field.TransformIndexToPhysicalPoint((int(ix_m[i]), int(iy_m[i]), int(iz_m[i])))                    
                    else:
                        point_m[i,:] = point_f[i,:] + self.gt_field.GetPixel((int(ix_f[i]), int(iy_f[i]), int(iz_f[i])))

                        check_ix_m, check_iy_m, check_iz_m = self.gt_field.TransformPhysicalPointToContinuousIndex(point_m[i,:])
                        if not (0<check_ix_m<size[0] and 0<check_iy_m<size[1] and 0<check_iz_m<size[2]):
                            delete_index.append(i)        

            print('删除图像外金标准点对', len(delete_index),'对')
            print('delete gt correspondence outside image', len(delete_index),'pairs')
            self.point_f = np.delete(point_f, delete_index, axis=0)
            self.point_m = np.delete(point_m, delete_index, axis=0)

            return self.point_f, self.point_m


    def __ChooseDescriptor(self, descriptor, **params):
        if descriptor == 'SIFT':
            return cv2.xfeatures2d.SIFT_create()
        if descriptor == 'SURF':
            return cv2.xfeatures2d.SURF_create()
        if descriptor == 'FAST':
            return cv2.FastFeatureDetector_create(threshold=params['threshold'])
        if descriptor == 'BRISK':
            return cv2.BRISK_create(radiusList=params['radius'], numberList=params['samplenum'], dMax=params['dmax'], dMin=params['dmin'])
        if descriptor == 'MSER':
            return cv2.MSER_create()
        if descriptor == 'ORB':
            return cv2.ORB_create()
        else:
            return cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()

    def __DetectAndDescripte2DSlice(self, image: sitk.Image=None, mask: sitk.Image=None, 
                                    detector=detectorFAST, detector_threshold=1,
                                    descriptor=descriptorBRISK, descriptor_radius=[0,3,6,9], descriptor_samplenum=[1,8,14,20], descriptor_dmax=6, descriptor_dmin=9):
        slice_img=sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkUInt8)).squeeze()
        slice_mask=sitk.GetArrayFromImage(mask).squeeze()
        
        if detector == self.detectorCANNY:
            slice_canny = sitk.CannyEdgeDetection(sitk.Cast(image, sitk.sitkFloat32), lowerThreshold=detector_threshold, upperThreshold=detector_threshold)
            slice_canny = sitk.Multiply(slice_canny, sitk.Cast(mask, sitk.sitkFloat32))
            slice_canny_array = sitk.GetArrayFromImage(slice_canny)
            
            # 换一下顺序是因为numpy是行列的顺序，而坐标需要x y的顺序
            # swap and y axis because numpy is row and column order, but coordinates need x and y order
            slice_index = np.where(slice_canny_array>0)[::-1]
            
            slice_kpt = []
            for slice_pt_x, slice_pt_y in zip(slice_index[0], slice_index[1]):
                slice_kp = cv2.KeyPoint(slice_pt_x, slice_pt_y, 1)
                slice_kpt.append(slice_kp)
        else:
            slice_detect=self.__ChooseDescriptor(detector, threshold=detector_threshold)
            slice_kpt=slice_detect.detect(slice_img, slice_mask)

        descript = self.__ChooseDescriptor(descriptor, radius=descriptor_radius, samplenum=descriptor_samplenum, dmax=descriptor_dmax, dmin=descriptor_dmin)
        (slice_kpt, slice_desc) = descript.compute(slice_img, slice_kpt)
        
        return [slice_kpt, slice_desc]


    def AreCorrespondenceGood(self, pfs=None, pms=None, error_threshold=1, mask: sitk.Image=None):
        gt_trans = sitk.DisplacementFieldTransform(sitk.Image(self.gt_field))

        if pfs is None and pms is None:
            pfs = self.point_f
            pms = self.point_m

        num_pf_in_mask = 0
        good_points_index = np.zeros(len(pfs)).astype(bool)
        
        for i in range(len(pfs)):
            pf = pfs[i,:]
            pm = pms[i,:]
            
            # check pf in mask
            if mask is not None:
                if mask.GetPixel(mask.TransformPhysicalPointToIndex(pf)) == 0:
                    continue
                
            num_pf_in_mask += 1
            gt_pm = gt_trans.TransformPoint(pf)
            error = np.linalg.norm(np.array(pm) - np.array(gt_pm))
            
            if error < error_threshold:
                good_points_index[i] = True

        correct_number = np.count_nonzero(good_points_index)
        if num_pf_in_mask == 0:
            return [0, 0, 0, good_points_index]
        else:
            return [correct_number, num_pf_in_mask - correct_number, correct_number/num_pf_in_mask*100, good_points_index]


    def __CvIndexToPhysicalPoint(self, keypoints: list=None, image: sitk.Image=None):
        num_points = len(keypoints)
        num_dimension = self.num_dimension
        
        index_points = np.zeros((num_points,num_dimension))
        physical_points = np.zeros((num_points,num_dimension))
        for i in range(len(keypoints)):
            keypoint = keypoints[i]
            
            # 这步为了把三维特征提取确实的z轴为0的维度补上
            # this step is to fill the z-axis of the three-dimensional feature extraction with 0
            index_points[i, 0:2] = keypoint.pt
            physical_point = image.TransformContinuousIndexToPhysicalPoint(index_points[i])
            physical_points[i]=physical_point
            
        return physical_points

    def GetCorrespondencePoints(self, 
                                feature_detector=detectorFAST, feature_detector_size=7, feature_detector_angle=90, feature_detector_ratio=3, feature_detector_num=1000, 
                                feature_descriptor=descriptorBRISK, feature_descriptor_radius=[0,3,6,9], feature_descriptor_samplenum=[1,8,14,20], feature_descriptor_dmax=6,  feature_descriptor_dmin=9,
                                feature_match_ratio=1, feature_match_crosscheck=True,
                                given_fixed_features=None, given_moving_features=None):
        
        # 如果给定了特征点和描述子，则直接使用
        # if given_fixed_features and given_moving_features, then use them directly
        if given_fixed_features and given_moving_features:
            tgt_ppt = given_fixed_features['points']
            tgt_desc = given_fixed_features['features']
            src_ppt = given_moving_features['points']
            src_desc = given_moving_features['features']
        else:
            # 否则检测特征点和描述子
            # otherwise detect feature points and descriptors
            
            if self.num_dimension == 2:
                # 先把图像转换为cv能操作的uint型
                # first convert the image to cv uint type
                cv_fixed = sitk.RescaleIntensity(self.fixed)
                cv_warped = sitk.RescaleIntensity(self.warped)
                
                # 计算特征检测阈值
                # calculate feature detection threshold
                if feature_detector == self.detectorFAST:
                    fixed_detector_threshold = int(np.std(sitk.GetArrayFromImage(cv_fixed)[np.where(sitk.GetArrayFromImage(self.fixed_mask)>0)]) / feature_detector_ratio)
                    warped_detector_threshold = int(np.std(sitk.GetArrayFromImage(cv_warped)[np.where(sitk.GetArrayFromImage(self.warped_mask)>0)]) / feature_detector_ratio)
                else:
                    fixed_detector_threshold = 0.1 / feature_detector_ratio
                    warped_detector_threshold = 0.1 / feature_detector_ratio    
                
                [tgt_kpt, tgt_desc] = self.__DetectAndDescripte2DSlice(image=cv_fixed, mask=self.fixed_mask,
                    detector=feature_detector, detector_threshold=fixed_detector_threshold,
                    descriptor=feature_descriptor, descriptor_radius=feature_descriptor_radius, descriptor_samplenum=feature_descriptor_samplenum, descriptor_dmax=feature_descriptor_dmax, descriptor_dmin=feature_descriptor_dmin)
                [src_kpt, src_desc] = self.__DetectAndDescripte2DSlice(image=cv_warped, mask=self.warped_mask,
                    detector=feature_detector, detector_threshold=warped_detector_threshold,
                    descriptor=feature_descriptor, descriptor_radius=feature_descriptor_radius, descriptor_samplenum=feature_descriptor_samplenum, descriptor_dmax=feature_descriptor_dmax, descriptor_dmin=feature_descriptor_dmin)
                tgt_ppt = self.__CvIndexToPhysicalPoint(tgt_kpt, self.fixed)
                src_ppt = self.__CvIndexToPhysicalPoint(src_kpt, self.warped)
                
            elif self.num_dimension == 3:
                # use my fast3d
                from . import fast3d
                fast = fast3d.FastFeature3D(size=feature_detector_size, angle=feature_detector_angle, detect_ratio=feature_detector_ratio, debug=self.debug)
                
                if feature_descriptor == self.descriptorBRISK:
                    tgt_ppt, tgt_desc, tgt_label = fast.GetFastFeaturesWithBRISKDescriptors(self.fixed, self.fixed_mask, radiusList=feature_descriptor_radius, numberList=feature_descriptor_samplenum, dMax=feature_descriptor_dmax)
                    src_ppt, src_desc, src_label = fast.GetFastFeaturesWithBRISKDescriptors(self.warped, self.warped_mask, radiusList=feature_descriptor_radius, numberList=feature_descriptor_samplenum, dMax=feature_descriptor_dmax)
                elif feature_descriptor == self.descriptorSIFT:
                    tgt_ppt, tgt_desc, tgt_label = fast.GetFastFeaturesWithSIFTDescriptors(self.fixed, self.fixed_mask)
                    src_ppt, src_desc, src_label = fast.GetFastFeaturesWithSIFTDescriptors(self.warped, self.warped_mask)
                else:
                    print('Error: 3D SIFT only support BRISK and SIFT descriptors.')
                    
                if self.debug:
                    sitk.WriteImage(tgt_label, f'temp/{self.debug_prefix}_fixed_points.mha', True)
                    sitk.WriteImage(src_label, f'temp/{self.debug_prefix}_moving_points.mha', True)
                    np.savez(f'temp/{self.debug_prefix}_fixed_features.npz', points=tgt_ppt, features=tgt_desc)
                    np.savez(f'temp/{self.debug_prefix}_moving_features.npz', points=src_ppt, features=src_desc)
                
        if self.debug:
            print('刚开始检测出来固定特征点', len(tgt_ppt), '个')
            print('刚开始检测出来浮动特征点', len(src_ppt), '个')
            print('fixed feature points', len(tgt_ppt), 'points')
            print('moving feature points', len(src_ppt), 'points')

        # 开始匹配
        # start matching
        if feature_descriptor == self.descriptorBRISK or feature_descriptor == self.descriptorORB:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=feature_match_crosscheck)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=feature_match_crosscheck)
            
        matches=matcher.match(queryDescriptors = tgt_desc, 
                              trainDescriptors = src_desc)
        matches = sorted(matches, key=lambda x: x.distance)

        # 去重复匹配点（固定一个特征点对应多个候选位置的，只保留最好的那个）
        # remove duplicate matching points (only keep the best one for a fixed feature point corresponding to multiple candidate positions)
        # 先把两个ppt按照match的顺序进行排序，然后用unique保留唯一tgt的点，src的也按照这个index进行索引
        # first sort the two ppt according to the match order, and then use unique to keep the unique tgt points, and src is also indexed according to this index
        num_match = len(matches)
        point_f=np.zeros((num_match,self.num_dimension))
        point_m=np.zeros((num_match,self.num_dimension))
        
        for i in range(len(matches)):
            point_f[i] = tgt_ppt[matches[i].queryIdx]
            point_m[i] = src_ppt[matches[i].trainIdx]

        # 去除SIFT可能会有次级主方向的点，没用
        # remove the points that SIFT may have secondary main direction, which is useless
        _, unique_id = np.unique(point_f, return_index=True, axis=0)
        unique_id_unsort = np.sort(unique_id)

        if self.debug:
            print('去除同位置点', len(point_f) - len(unique_id), '对')
        
        point_f = point_f[unique_id_unsort]
        point_m = point_m[unique_id_unsort]            
        
        # 保留ratio内的优质匹配（本文方法可以100%保留）
        # keep the quality matching within ratio (this method can keep 100%)
        keep_num = min(len(point_m), feature_detector_num)
        ratio_num = int(keep_num*feature_match_ratio)

        self.point_f = point_f[:ratio_num]
        self.point_m = point_m[:ratio_num]
        
        if self.debug:
            print('比例筛选后剩余特征点', ratio_num, '对')     

        return self.point_f, self.point_m            
    

    def ShowResults(self, point_f=None, point_m=None, field:sitk.Image=None, number: range=None, isRandom=False):
        
        # display result
        plt.figure(dpi=150)
        fixed_array = sitk.GetArrayFromImage(self.fixed)
        warped_array = sitk.GetArrayFromImage(self.warped)

        if field:
            s = sitk.GetArrayFromImage(field)
            sx = s[...,0]; sy = s[...,1]

            x=np.linspace(0,field.GetWidth()-1,field.GetWidth())
            y=np.linspace(0,field.GetHeight()-1,field.GetHeight())

            X, Y=np.meshgrid(x,y)
            point_f = np.dstack((X,Y)).reshape(-1,2)
            point_m = np.dstack((sx,sy)).reshape(-1,2) + point_f

            trans = sitk.DisplacementFieldTransform(field)
            warped_img = sitk.Resample(self.moving, self.fixed, trans)
            im=np.hstack((fixed_array, warped_array, sitk.GetArrayFromImage(warped_img)))
        else:
            im=np.hstack((fixed_array, warped_array))
        plt.imshow(im)

        if self.fixed.GetNumberOfComponentsPerPixel() == 1:
            plt.gray()
            
        if point_f is None and point_m is None:
            point_f = self.point_f
            point_m = self.point_m
            
            if point_f is None or point_m is None:
                plt.show()
                return
                
        if number is None:
            number = range(point_f.shape[0])
        else:
            number = range(point_f.shape[0]) if len(number)>point_f.shape[0] else number

        if isRandom:
            index = np.random.choice(point_f.shape[0], len(number))
            point_f = point_f[index, :]
            point_m = point_m[index, :]

        for i in number:
            pf_index = self.fixed.TransformPhysicalPointToContinuousIndex(point_f[i,:].tolist()) # plt和sitk坐标系都是x、y，所以这里不用转换。
            pm_index = self.moving.TransformPhysicalPointToContinuousIndex(point_m[i,:].tolist())

            xa = float(pf_index[0])
            ya = float(pf_index[1])
            xb = float(pm_index[0])+fixed_array.shape[1]
            yb = float(pm_index[1])
            c=np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa,ya), radius=2, color=c))
            plt.gca().add_artist(plt.Circle((xb,yb), radius=2, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.5)    

        plt.show()


    def __CheckFieldInMaskIsGood(self, mask, field):
        before_routes_AntsNCC = self.__GetMaskedMetric(self.fixed, self.warped, mask, self.warped_mask, sitk.TranslationTransform(self.num_dimension))
        after_routes_AntsNCC = self.__GetMaskedMetric(self.fixed, self.warped, mask, self.warped_mask, sitk.DisplacementFieldTransform(sitk.Image(field)))
        is_AntsNCC_good = after_routes_AntsNCC - before_routes_AntsNCC
        return is_AntsNCC_good
    
    def __GetMaskedMetric(self, fixed, moving, fixed_mask, moving_mask, trans):
        checker = sitk.ImageRegistrationMethod()
        checker.SetInitialTransform(trans)
        checker.SetMetricFixedMask(fixed_mask)
        checker.SetMetricMovingMask(moving_mask)
        checker.SetMetricAsANTSNeighborhoodCorrelation(3)
        return -checker.MetricEvaluate(sitk.Cast(fixed, sitk.sitkFloat64), sitk.Cast(moving, sitk.sitkFloat64))    

    def __BadEnding(self, total_flight_pfs):
        empty_field = sitk.Image(self.fixed.GetSize(), sitk.sitkVectorFloat64)
        empty_field.CopyInformation(self.fixed)
        
        this_flight_pfs = []
        return empty_field, this_flight_pfs, total_flight_pfs
    
    def GenerateFlightField(self, ccm_k=0.05, ccm_z=0.5, ccm_t=0.5, ccm_e=0.5, ccm_j=0.5, eng=None, route_merge_pfs=[], route_check: bool=True, mask_valid: bool=True, just_tps: bool=False, tps_lambda: float=0.0):

        # 如果点数不够聚类，直接BE
        # if not enough points to cluster, directly BE
        if ccm_k < 1:
            if len(self.point_f)*ccm_k < 10:
                return self.__BadEnding(route_merge_pfs)
        else:
            if len(self.point_f) < ccm_k:
                return self.__BadEnding(route_merge_pfs)

        if just_tps:
            # 不符合就直接把route merge的点加进来，然后一起生成tps
            # if not meet the condition, just add the route merge points and generate tps together
            this_flight_pfs = self.point_f
            this_flight_pms = self.point_m
            
            # 如果没有route merge，当前的routes就是最终的routes
            # if no route merge, the current routes are the final routes
            if len(route_merge_pfs) == 0:
                total_flight_pfs = this_flight_pfs
                total_flight_pms = this_flight_pms
            else:
                # 把处在sub route mask内的route merge的点去除掉
                # remove the route merge points that are in the sub route mask
                collective_routes_mask_array = np.sum([sitk.GetArrayFromImage(collective_routes_convex_masks[i]) for i in chosen_index], axis=0)
                collective_routes_mask = sitk.GetImageFromArray(collective_routes_mask_array)
                collective_routes_mask.CopyInformation(self.fixed)
                
                # 生成检查结果list，如果route merge pf所在mask位置为0，说明不在任意一个collective routes区域内
                # generate check result list, if the mask position of route merge pf is 0, it means it is not in any collective routes area
                is_route_merge_ok = [collective_routes_mask.GetPixel(collective_routes_mask.TransformPhysicalPointToIndex(route_merge_pf))==0 for route_merge_pf in route_merge_pfs]
                
                # 把符合调节的route merge点加入进来
                # add the route merge points that meet the adjustment
                total_flight_pfs = np.vstack((this_flight_pfs, route_merge_pfs[is_route_merge_ok]))
                total_flight_pms = np.vstack((this_flight_pms, route_merge_pfs[is_route_merge_ok]))

            self.flight_field = self.__CreateTPSFieldUsingPoints(total_flight_pfs, total_flight_pms, tps_lambda)
            self.flight_field.CopyInformation(self.fixed)
            return self.flight_field, this_flight_pfs, total_flight_pfs

        location = self.point_f
        movement = self.point_m - self.point_f

        matlab_location = matlab.double(location.tolist())
        matlab_movement = matlab.double(movement.tolist())
        
        matlab_ccm_k = float(ccm_k)
        matlab_ccm_z = float(ccm_z)
        matlab_ccm_t = float(ccm_t)
        matlab_ccm_e = float(ccm_e)
        matlab_ccm_j = float(ccm_j)

        self.collective_routes_label = np.array(eng.get_collected_points(matlab_location,matlab_movement, 
                                                                     matlab_ccm_k, matlab_ccm_z, matlab_ccm_t, matlab_ccm_e, matlab_ccm_j, self.debug)).squeeze()
        self.collective_routes_numbers = np.int(self.collective_routes_label.max())

        if self.debug:
            print('刚开始聚类出来了', self.collective_routes_numbers, '类')
            print('每个大类中分别有', [np.count_nonzero(self.collective_routes_label == i+1) for i in range(self.collective_routes_numbers)] , '点对')
            print('The number of clusters is', self.collective_routes_numbers)
            print('The number of points in each cluster is', [np.count_nonzero(self.collective_routes_label == i+1) for i in range(self.collective_routes_numbers)] , 'pairs')

        # 如果一个大区都没保留下来，那就返回一个空的
        # if no big region is left, return an empty one
        if self.collective_routes_numbers == 0:
            self.chosen_routes_number = 0
            if self.debug:
                print('并没有检测出一致性区域！')
                print('No consistency area detected!')
            return self.__BadEnding(route_merge_pfs)

        if route_check:
            # 得到各个区域的TPS形变场
            # get the TPS deformation field of each region
            initial_flight_fields = self.__CreateBigRegionTPSFields()

            # 得到各个区域的mask
            # get the mask of each region
            collective_routes_convex_masks = self.__CreateBigRegionMasks()

            routes_scroe_vector = np.zeros(self.collective_routes_numbers)
            routes_label = 0
            for mask, field in zip(collective_routes_convex_masks, initial_flight_fields):
                check_results = self.__CheckFieldInMaskIsGood(mask, field)

                if self.debug:
                    print(f'领头羊{routes_label+1}号: AntsNCC:{check_results}')
                    print(f'Collective routes {routes_label+1}: AntsNCC:{check_results}')
                
                routes_scroe_vector[routes_label] = check_results
                routes_label += 1

            # 得到筛选后的big regions
            # get the filtered big regions
            chosen_index = np.where(routes_scroe_vector>0)[0].tolist()
            self.chosen_routes_number = len(chosen_index)
            # 如果一个大区都没保留下来，那就返回一个空的
            # if no big region is left, return an empty one
            if self.chosen_routes_number == 0:
                if self.debug:
                    print('一致性区域没有一个考核通过！')
                    print('No consistency area passed the assessment!')
                return self.__BadEnding(route_merge_pfs)
            else:
                if self.debug:
                    print('考核通过的领头羊: ', chosen_index)
                    print('The leader passed the assessment: ', chosen_index)
                    collective_routes_mask_array = np.max([sitk.GetArrayFromImage(collective_routes_convex_masks[i])*(i+1) for i in chosen_index], axis=0)
                    collective_routes_mask = sitk.GetImageFromArray(collective_routes_mask_array)
                    collective_routes_mask.CopyInformation(self.fixed)
                    sitk.WriteImage(collective_routes_mask, f'temp/{self.debug_prefix}_collective_routes_mask.mha', True)

                    collective_routes_score_array = np.max([sitk.GetArrayFromImage(collective_routes_convex_masks[i])*routes_scroe_vector[i] for i in chosen_index], axis=0)
                    collective_routes_score = sitk.GetImageFromArray(collective_routes_score_array)
                    collective_routes_score.CopyInformation(self.fixed)
                    sitk.WriteImage(collective_routes_score, f'temp/{self.debug_prefix}_collective_routes_score.mha', True)
                        
                # 入选点的编号，入选的为True
                # the number of selected points, the selected ones are True
                self.collective_routes = np.array([i-1 in chosen_index for i in self.collective_routes_label])
                if self.debug:
                    print('入选的特征点共', np.count_nonzero(self.collective_routes), '对')
                    print('The selected feature points are', np.count_nonzero(self.collective_routes), 'pairs')
                    
                    if self.num_dimension == 2:
                        # 画入围特征点匹配结果
                        # draw the matching results of the enclosed feature points
                        inliers_index = np.where(self.collective_routes==True)
                        self.ShowResults(self.point_f[inliers_index], self.point_m[inliers_index])
                        
                # 如果只有一簇routes，并且没有需要merge的routes，且lambda为0，就直接采用这个routes，不要重复生成tps了
                # if there is only one cluster of routes, and there are no routes to be merged, and lambda is 0, just use this route, do not regenerate tps
                if self.chosen_routes_number == 1 and len(route_merge_pfs) == 0 and tps_lambda == 0.0:
                    self.flight_field = initial_flight_fields[chosen_index[0]]
                    
                    this_flight_pfs = self.point_f[self.collective_routes]
                    this_flight_pms = self.point_m[self.collective_routes]
                    
                    total_flight_pfs = this_flight_pfs
                    total_flight_pms = this_flight_pms
                else:
                    # 不符合就直接把route merge的点加进来，然后一起生成tps
                    # if not meet the condition, just add the route merge points and generate tps together
                    this_flight_pfs = self.point_f[self.collective_routes]
                    this_flight_pms = self.point_m[self.collective_routes]
                    
                    # 如果没有route merge，当前的routes就是最终的routes
                    # if no route merge, the current routes are the final routes
                    if len(route_merge_pfs) == 0:
                        total_flight_pfs = this_flight_pfs
                        total_flight_pms = this_flight_pms
                    else:
                        # 把处在collective routes mask内的route merge的点去除掉
                        # remove the route merge points that are in the collective routes mask
                        collective_routes_mask_array = np.sum([sitk.GetArrayFromImage(collective_routes_convex_masks[i]) for i in chosen_index], axis=0)
                        collective_routes_mask = sitk.GetImageFromArray(collective_routes_mask_array)
                        collective_routes_mask.CopyInformation(self.fixed)
                        
                        # 生成检查结果list，如果route merge pf所在mask位置为0，说明不在任意一个collective routes区域内
                        # generate check result list, if the mask position of route merge pf is 0, it means it is not in any collective routes area
                        is_route_merge_ok = [collective_routes_mask.GetPixel(collective_routes_mask.TransformPhysicalPointToIndex(route_merge_pf))==0 for route_merge_pf in route_merge_pfs]
                        
                        # 把符合调节的route merge点加入进来
                        # add the route merge points that meet the adjustment
                        total_flight_pfs = np.vstack((this_flight_pfs, route_merge_pfs[is_route_merge_ok]))
                        total_flight_pms = np.vstack((this_flight_pms, route_merge_pfs[is_route_merge_ok]))
                    
                    self.flight_field = self.__CreateTPSFieldUsingPoints(total_flight_pfs, total_flight_pms, tps_lambda)
                    self.flight_field.CopyInformation(self.fixed)

                if mask_valid:
                    # 验证整体形变好不好
                    # verify whether the overall deformation is good
                    check_results = self.__CheckFieldInMaskIsGood(self.fixed_mask, self.flight_field)

                    if self.debug:
                        print(f'最后整体验证: AntsNCC:{check_results}')
                        print(f'Final verification: AntsNCC:{check_results}')
                    is_routes_good = check_results
                    
                    if is_routes_good:
                        if self.debug:
                            print('验证通过！')
                            print('Verification passed!')
                        return self.flight_field, this_flight_pfs, total_flight_pfs
                    else:
                        if self.debug:
                            print('吃瓜群众拒绝本次形变！')
                            print('The flocks refuse this deformation!')
                        return self.__BadEnding(route_merge_pfs)
                else:
                    return self.flight_field, this_flight_pfs, total_flight_pfs
        else:
            # 即把所有的routes都加进来，然后一起生成tps
            # that is to add all routes and generate tps together
            self.collective_routes = np.array([i>0 for i in self.collective_routes_label])
            if self.debug:
                print('入选的特征点共', np.count_nonzero(self.collective_routes), '对')
                print('The selected feature points are', np.count_nonzero(self.collective_routes), 'pairs')
                
                if self.num_dimension == 2:
                    # 画入围特征点匹配结果
                    # draw the matching results of the enclosed feature points
                    inliers_index = np.where(self.collective_routes==True)
                    self.ShowResults(self.point_f[inliers_index], self.point_m[inliers_index])

            # 不符合就直接把route merge的点加进来，然后一起生成tps
            # if not meet the condition, just add the route merge points and generate tps together
            this_flight_pfs = self.point_f[self.collective_routes]
            this_flight_pms = self.point_m[self.collective_routes]
            
            # 如果没有route merge，当前的routes就是最终的routes
            # if no route merge, the current routes are the final routes
            if len(route_merge_pfs) == 0:
                total_flight_pfs = this_flight_pfs
                total_flight_pms = this_flight_pms
            else:
                # 把处在collective routes mask内的route merge的点去除掉
                # remove the route merge points that are in the collective routes mask
                collective_routes_mask_array = np.sum([sitk.GetArrayFromImage(collective_routes_convex_masks[i]) for i in chosen_index], axis=0)
                collective_routes_mask = sitk.GetImageFromArray(collective_routes_mask_array)
                collective_routes_mask.CopyInformation(self.fixed)
                
                # 生成检查结果list，如果route merge pf所在mask位置为0，说明不在任意一个collective routes区域内
                # generate check result list, if the mask position of route merge pf is 0, it means it is not in any collective routes area
                is_route_merge_ok = [collective_routes_mask.GetPixel(collective_routes_mask.TransformPhysicalPointToIndex(route_merge_pf))==0 for route_merge_pf in route_merge_pfs]
                
                # 把符合调节的route merge点加入进来
                # add the route merge points that meet the adjustment
                total_flight_pfs = np.vstack((this_flight_pfs, route_merge_pfs[is_route_merge_ok]))
                total_flight_pms = np.vstack((this_flight_pms, route_merge_pfs[is_route_merge_ok]))
            
            self.flight_field = self.__CreateTPSFieldUsingPoints(total_flight_pfs, total_flight_pms, tps_lambda)
            self.flight_field.CopyInformation(self.fixed)

            return self.flight_field, this_flight_pfs, total_flight_pfs


    def __CreateBigRegionMasks(self):

        # 先生成每个点带有其label的顶点图像
        # first generate the vertex image with its label for each point
        collective_routes_mask_point = sitk.Image(self.fixed.GetSize(), sitk.sitkUInt8)
        collective_routes_mask_point.CopyInformation(self.fixed)

        for pf, label in zip(self.point_f, self.collective_routes_label):
            collective_routes_mask_point.SetPixel(collective_routes_mask_point.TransformPhysicalPointToIndex(pf.tolist()), int(label))
        collective_routes_mask_point_array = sitk.GetArrayFromImage(collective_routes_mask_point)

        if self.num_dimension == 2 and self.debug:
            fig, axes = plt.subplots(1, self.collective_routes_numbers + 1)
            ax = axes.ravel()

            ax[0].set_title(f'Collective routes labels')
            ax[0].imshow(collective_routes_mask_point_array, cmap=plt.get_cmap('tab10'))
            ax[0].set_axis_off()

        collective_routes_convex_masks = []
        for routes_label in range(self.collective_routes_numbers):
            collective_routes_convex_mask_array = convex_hull_image(collective_routes_mask_point_array == routes_label+1, offset_coordinates=False) * 1 #这么做为了把bool转为数值 # this is to convert bool to numeric
            collective_routes_convex_mask = sitk.Cast(sitk.GetImageFromArray(collective_routes_convex_mask_array), sitk.sitkUInt8)
            collective_routes_convex_mask.CopyInformation(self.fixed)

            collective_routes_convex_masks.append(collective_routes_convex_mask)

            if self.num_dimension == 2 and self.debug:
                ax[routes_label+1].set_title(f'Mask {routes_label+1}')
                ax[routes_label+1].imshow(collective_routes_convex_mask_array, cmap=plt.cm.gray)
                ax[routes_label+1].set_axis_off()

        if self.num_dimension == 2 and self.debug:
            plt.tight_layout()
            plt.show()      

        return collective_routes_convex_masks  

    def __CreateTPSFieldUsingRegionIndex(self, chosen_index):
        pfs = []; pms = []
        for routes_label in chosen_index:
            index = np.where(self.collective_routes_label==routes_label+1)
            pfs.append(self.point_f[index])
            pms.append(self.point_m[index])
        
        pfs = np.vstack(pfs); pms = np.vstack(pms)
        field = gpu_tps.gpu_tps(pfs, pms, self.fixed)
        return field

    def __CreateBigRegionTPSFields(self):

        initial_flight_fields = []
        for routes_label in range(self.collective_routes_numbers):
            initial_flight_fields.append(self.__CreateTPSFieldUsingRegionIndex([routes_label]))
        return initial_flight_fields
    

    def __CreateTPSFieldUsingPoints(self, pfs, pms, tps_lambda=0.0):
        field = gpu_tps.gpu_tps(pfs, pms, self.fixed, tps_lambda)
        return field