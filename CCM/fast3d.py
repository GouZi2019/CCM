import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool
import time
import torch.nn.functional as F
import torch

class FastFeature3D(object):
    def __init__(self, size=7, angle=90, detect_ratio=10, nonmax_suppression=True, debug=True) -> None:
        self.radius = (size-1)//2
        self.size = self.radius * 2 + 1
        self.check_length = self.size*np.sin(angle/2*np.pi/180) # 求最大容忍弦长，用于第二步检验 # calculate the maximum length of the chord
        self.detect_ratio = detect_ratio
        self.nonmax_suppression = nonmax_suppression
        self.debug = debug
        
        self.first_check_index, self.second_check_index = self.__GetCheckRelativePositions()
    
    def __GetCheckRelativePositions(self):
        check_box = np.zeros((self.size,self.size,self.size))
        
        # for first check
        check_box[0,self.radius,self.radius] = check_box[-1,self.radius,self.radius] = 1
        check_box[self.radius,0,self.radius] = check_box[self.radius,-1,self.radius] = 1
        check_box[self.radius,self.radius,0] = check_box[self.radius,self.radius,-1] = 1
        first_check_index = np.where(check_box==1)
        
        # for second check
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    check_box[i,j,k] = np.linalg.norm((i-self.radius,j-self.radius,k-self.radius))
        second_check_index = np.where(np.logical_and(check_box > self.radius - 0.5, check_box < self.radius + 0.5))
        
        return first_check_index, second_check_index

    def FeatureDetect(self, img: sitk.Image, mask: sitk.Image) -> np.array:
        
        img_array = sitk.GetArrayFromImage(img)
        mask_array = sitk.GetArrayFromImage(mask)
        threshold = np.std(img_array[np.where(mask_array)]) / self.detect_ratio
        
        label_step1 = self.__FeatureDetectStep1(img_array, mask_array, threshold)
        label_step2, value_array = self.__FeatureDetectStep2(img_array, label_step1, threshold)

        indexes = np.where(label_step2)
        keypoints_value = value_array[indexes]
        
        num_keypoints = len(indexes[0])
        keypoints_physical = np.zeros((num_keypoints,3), dtype=float)
        keypoints_index = np.zeros((num_keypoints,3), dtype=int)
        for index in range(num_keypoints):
            i, j, k = indexes[2][index], indexes[1][index], indexes[0][index]
            keypoints_physical[index] = np.array(img.TransformIndexToPhysicalPoint((int(i), int(j), int(k))))
            keypoints_index[index] = np.array([i, j, k])

        keypoint_img = sitk.GetImageFromArray(label_step2)
        keypoint_img.CopyInformation(img)
        
        return keypoint_img, keypoints_physical, keypoints_index, keypoints_value

    def __FeatureDetectStep1(self, img_array: np.array, mask_array: np.array, threshold: float) -> np.array:
        import torch.nn.functional as F
        import torch
        
        time_start = time.time()
        check_kernel = torch.zeros((len(self.first_check_index[0]),1,self.size,self.size,self.size))
        check_kernel[:,0,self.radius,self.radius,self.radius] = -1
        for i in range(len(self.first_check_index[0])):
            check_kernel[i,0,self.first_check_index[0][i],self.first_check_index[1][i],self.first_check_index[2][i]] = 1
        
        torch_img_array = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0)
        torch_check_array = F.conv3d(torch_img_array, check_kernel, padding=self.radius)
        result_check_array = ((torch.abs(torch_check_array) > threshold).sum(dim=1) >= 4).squeeze()

        torch_mask_array = torch.from_numpy(mask_array).float()
        torch_label_array = result_check_array*torch_mask_array
        label_array = torch_label_array.numpy()
        
        time_end = time.time()
        if self.debug:
            print('Num of points in step1:', np.sum(label_array))
            print('Step1 time cost:', time_end-time_start, 's')
        
        return label_array        

    def __calculate_fast_value(self, center_value, check_values, threshold):
        bright_index = np.where(check_values > center_value + threshold)
        dark_index = np.where(check_values < center_value - threshold)
        
        overhead_threshold_abs = np.abs(check_values - center_value) - threshold
        fast_value = np.max([np.sum(overhead_threshold_abs[bright_index]), np.sum(overhead_threshold_abs[dark_index])])
        return fast_value

    def __FeatureDetectStep2(self, img_array: np.array, mask_array: np.array, threshold: float) -> np.array:
        mask_index = np.where(mask_array)
        label_array = np.zeros_like(img_array)
        value_array = np.zeros_like(img_array, np.float64)
        
        time_start = time.time()
        for i,j,k in zip(mask_index[0], mask_index[1], mask_index[2]):
            check_box = img_array[i-self.radius:i+self.radius+1, j-self.radius:j+self.radius+1, k-self.radius:k+self.radius+1]
            if all(size == self.size for size in check_box.shape):
                # true: we can get a check box for (i,j,k)
                center_value = img_array[i,j,k]

                second_check_values = check_box[self.second_check_index]
                second_false_index = np.where(np.abs(second_check_values-center_value)<threshold)
                second_false_index = [self.second_check_index[n][second_false_index] for n in range(3)]
                second_false_points = np.array(second_false_index).T
                
                if len(second_false_points) < 2:
                    label_array[i,j,k] = 1
                    value_array[i,j,k] = self.__calculate_fast_value(center_value, second_check_values, threshold)
                    continue
                
                distance_between_points = np.linalg.norm(second_false_points-second_false_points[:,None], axis=-1)
                max_distance = np.max(distance_between_points)
                
                # Second check!
                if max_distance < self.check_length:
                    label_array[i,j,k] = 1
                    value_array[i,j,k] = self.__calculate_fast_value(center_value, second_check_values, threshold)
        time_end = time.time()
        if self.debug:
            print('Num of points in step2:', np.sum(label_array))
            print('Step2 time cost:', time_end-time_start, 's')
        
        # nonmax_suppression
        from scipy.ndimage.filters import maximum_filter
        if self.nonmax_suppression:
            max_index = (value_array == maximum_filter(value_array,footprint=np.ones((self.radius,self.radius,self.radius))))
            nms_label_array = label_array*max_index
            nms_value_array = value_array*max_index
            
            if self.debug:
                print('Num of points before nms:', np.sum(label_array))
                print('Num of points after nms:', np.sum(nms_label_array))
            
            return nms_label_array, nms_value_array
        else:
            return label_array, value_array

    def CheckGivenPointsIsFastFeature(self, img: sitk.Image, mask: sitk.Image, points: np.array) -> np.array:
        img_array = sitk.GetArrayFromImage(img)
        mask_array = sitk.GetArrayFromImage(mask)
        threshold = np.std(img_array[np.where(mask_array)]) / self.detect_ratio

        num_points = len(points)
        zero_check_results = np.zeros(num_points)
        first_check_results = np.zeros(num_points)
        second_check_results = np.zeros(num_points)
        for id in range(num_points):
            index = img.TransformPhysicalPointToIndex(points[id])
            i, j, k = index[::-1]
            check_box = img_array[i-self.radius:i+self.radius+1, j-self.radius:j+self.radius+1, k-self.radius:k+self.radius+1]
            if all(size == self.size for size in check_box.shape):
                zero_check_results[id] = 1

                # true: we can get a check box for (i,j,k)
                center_value = img_array[i,j,k]
                first_check_values = check_box[self.first_check_index]
                
                # First check!
                pass_num = np.count_nonzero(np.abs(first_check_values-center_value)>threshold)
                if pass_num >= 4:
                    first_check_results[id] = 1

                    second_check_values = check_box[self.second_check_index]
                    second_false_index = np.where(np.abs(second_check_values-center_value)<threshold)
                    second_false_index = [self.second_check_index[n][second_false_index] for n in range(3)]
                    second_false_points = np.array(second_false_index).T
                    
                    if len(second_false_points) < 2:
                        second_check_results[id] = 1
                        continue
                    
                    distance_between_points = np.linalg.norm(second_false_points-second_false_points[:,None], axis=-1)
                    max_distance = np.max(distance_between_points)
                    
                    # Second check!
                    if max_distance < self.check_length:
                        second_check_results[id] = 1
                        
        return zero_check_results, first_check_results, second_check_results

    def FeatureDescribeWithBRISK(self, img: sitk.Image, indexes: np.array, radiusList=[0, 3, 6, 9], numberList=[1, 8, 14, 20], dMax=7, radius_num=None, radius_spacing=None, radius_thresh=None):
        img_array = sitk.GetArrayFromImage(img)

        # method1: use num, spacing, threshold, for testing
        if radius_num is not None and radius_spacing is not None and radius_thresh is not None:
            max_radius = radius_spacing * (radius_num-1)
            max_size = max_radius * 2 + 1
            check_box = np.zeros((max_size,max_size,max_size))
            distance_box = np.zeros((max_size,max_size,max_size))
            
            # for check
            for i in range(max_size):
                for j in range(max_size):
                    for k in range(max_size):
                        distance_box[i,j,k] = np.linalg.norm((i-max_radius,j-max_radius,k-max_radius))
                        
            # get index
            for i in range(radius_num):
                _radius = radius_spacing * i
                _check_index = np.where(np.abs(distance_box-_radius)<radius_thresh)
                check_box[_check_index] = 1
                
                if self.debug:
                    print('Radius:', _radius)
                    print('Num of points:', len(_check_index[0]))
                    print('------------------')
        else:
            # method2: use radiusList and numberList
            max_radius = np.max(radiusList)
            max_size = max_radius * 2 + 1          
            check_box = np.zeros((max_size,max_size,max_size))
            distance_box = np.zeros((max_size,max_size,max_size))
            
            # for check
            for i in range(max_size):
                for j in range(max_size):
                    for k in range(max_size):
                        distance_box[i,j,k] = np.linalg.norm((i-max_radius,j-max_radius,k-max_radius))
                        
            # get index
            for i in range(len(radiusList)):
                _radius = radiusList[i]
                _number = numberList[i]
                
                _abs_distance = np.abs(distance_box-_radius)
                # get _number of min distance
                _check_index = np.unravel_index(np.argsort(_abs_distance.flatten(), kind='stable')[:_number], _abs_distance.shape)
                check_box[_check_index] = 1
                
                if self.debug:
                    print('Radius:', _radius)
                    print('Num of points:', len(_check_index[0]))
                    print('------------------')
        
        time_start = time.time()

        check_index = np.where(check_box)
        N = len(check_index[0])
        if self.debug:
            os.makedirs('temp', exist_ok=True)
            sitk.WriteImage(sitk.GetImageFromArray(check_box), 'temp/check_box.mha', True)
            print('Total num of points:', N)
            print('------------------')    
        
        # distance matrix
        check_points = np.array(check_index).T
        # get upper triangle matrix: N*(N-1)/2, +1 to avoid 0
        distance_matrix = np.triu(np.linalg.norm(check_points-check_points[:,None], axis=-1) + 1) # N*N
        # get index of distance below dMax: +1 to recover
        feature_index = np.where(np.logical_and(distance_matrix > 0, distance_matrix < dMax+1))
        
        feature_bit_size = len(feature_index[0])
        feature_bit_pad = (8 - feature_bit_size % 8) % 8
        feature_size = int(np.ceil(feature_bit_size/8))
        if self.debug:
            print('feature_bit_size:', feature_bit_size)
            print('feature_size:', feature_size)
        
        # compute features using numpy
        is_valid = np.zeros(len(indexes))
        features = np.empty((len(indexes), feature_size), dtype=np.uint8)
        for index in range(len(indexes)):
            i, j, k = indexes[index][::-1]
            check_box = img_array[i-max_radius:i+max_radius+1, j-max_radius:j+max_radius+1, k-max_radius:k+max_radius+1]
            if all(size == max_size for size in check_box.shape):
                is_valid[index] = 1
                
                check_values = (check_box[check_index]).reshape(N,1)
                b_matrix = (check_values - check_values.T) > 0 # N*N
                
                feature_bits = np.pad(b_matrix[feature_index], (0, feature_bit_pad), 'constant')
                features[index] = np.packbits(feature_bits)

        time_end = time.time()
        if self.debug:
            print('Feature describe time cost:', time_end-time_start, 's')

        return features, is_valid

    def GetFastFeaturesWithBRISKDescriptors(self, img: sitk.Image, mask: sitk.Image, radiusList=[0, 3, 6, 9], numberList=[1, 8, 14, 20], dMax=7):
        
        # step1: 3D FAST feature detector
        keypoint_img, keypoints_physical, keypoints_index, _ = self.FeatureDetect(img, mask)
        
        # step2: 3D SIFT feature descriptor
        keypoint_descriptor, valid_index = self.FeatureDescribeWithBRISK(img, keypoints_index, radiusList=radiusList, numberList=numberList, dMax=dMax)
        
        return keypoints_physical[valid_index>0], keypoint_descriptor[valid_index>0], keypoint_img

    def GetFastFeaturesWithBRISKDescriptorsTest(self, img: sitk.Image, mask: sitk.Image, radius_num=4, radius_spacing=3, radius_thresh=0.1, dMax=7):
        
        # step1: 3D FAST feature detector
        keypoint_img, keypoints_physical, keypoints_index, _ = self.FeatureDetect(img, mask)
        
        # step2: 3D SIFT feature descriptor
        keypoint_descriptor, valid_index = self.FeatureDescribeWithBRISK(img, keypoints_index, radius_num=radius_num, radius_spacing=radius_spacing, radius_thresh=radius_thresh, dMax=dMax)
        
        return keypoints_physical[valid_index>0], keypoint_descriptor[valid_index>0], keypoint_img