import SimpleITK as sitk
import numpy as np
import gstools as gs

class ImageWarper(object):
    
    def __init__(self, img:sitk.Image, seg_list=[], background_value=0):
        self.img0 = img
        self.img = img
        self.seg_list = seg_list if isinstance(seg_list, list) else [seg_list]
        self.trans = sitk.CompositeTransform(self.img0.GetDimension())
        self.field = sitk.TransformToDisplacementField(self.trans, size=self.img0.GetSize(), outputOrigin=self.img0.GetOrigin(), outputSpacing=self.img0.GetSpacing(), outputDirection=self.img0.GetDirection())

        self.gt_trans = sitk.CompositeTransform(self.img0.GetDimension())
        self.gt_field = sitk.TransformToDisplacementField(self.gt_trans, size=self.img0.GetSize(), outputOrigin=self.img0.GetOrigin(), outputSpacing=self.img0.GetSpacing(), outputDirection=self.img0.GetDirection())

        self.default = background_value

    def __GetImageCenter(self):
        spacing = np.array(self.img0.GetSpacing())
        origin = np.array(self.img0.GetOrigin())
        size = np.array(self.img0.GetSize())

        center = origin + size/2 * spacing
        return center


    def WelcomeNewTransformation(self, transform, inverse_transform):
        self.trans.AddTransform(transform)
        self.field = sitk.TransformToDisplacementField(self.trans, size=self.img0.GetSize(), outputOrigin=self.img0.GetOrigin(), outputSpacing=self.img0.GetSpacing(), outputDirection=self.img0.GetDirection())

        self.img = sitk.Resample(self.img0, self.img0, self.trans, defaultPixelValue=self.default, interpolator=sitk.sitkBSplineResamplerOrder3)

        warped_seg_list = []
        for seg in self.seg_list:
            warped_seg = sitk.Resample(seg, self.img0, self.trans, sitk.sitkNearestNeighbor)
            warped_seg_list.append(warped_seg)
        self.warped_seg_list = warped_seg_list

        self.gt_trans = sitk.CompositeTransform([inverse_transform, self.gt_trans])
        self.gt_field = sitk.TransformToDisplacementField(self.gt_trans, size=self.img0.GetSize(), outputOrigin=self.img0.GetOrigin(), outputSpacing=self.img0.GetSpacing(), outputDirection=self.img0.GetDirection())



    def RandomRigidTransform(self, max_translation=50, max_rotation=180):
        # 2D or 3D?
        if self.img0.GetDimension() == 2:
            # create random transform
            translation = 2 * (np.random.random(2)-0.5) * max_translation
            rotation = 2 * (np.random.random(1)-0.5) * max_rotation*np.pi/180
            center = self.__GetImageCenter()

            transform = sitk.Euler2DTransform()
            transform.SetTranslation(translation)
            transform.SetAngle(np.double(rotation))
            transform.SetCenter(center)
        else:
            # create random transform
            translation = 2 * (np.random.random(3)-0.5) * max_translation
            rotation = 2 * (np.random.random(3)-0.5) * max_rotation*np.pi/180
            center = self.__GetImageCenter()

            transform = sitk.Euler3DTransform()
            transform.SetTranslation(translation)
            transform.SetRotation(rotation[0], rotation[1], rotation[2])
            transform.SetCenter(center)

        print('Rigid translation: ',translation, '\nRigid rotation: ', rotation)
        self.WelcomeNewTransformation(transform, transform.GetInverse())

        return self


    def Translation(self, translation):
        # 2D or 3D?
        if self.img0.GetDimension() == 2:
            transform = sitk.Euler2DTransform()
        else:
            transform = sitk.Euler3DTransform()
        
        transform.SetTranslation(np.double(np.array(translation)))

        self.WelcomeNewTransformation(transform, transform.GetInverse())

        return self


    def Rotation(self, rotation):
        # 2D or 3D?
        if self.img0.GetDimension() == 2:
            transform = sitk.Euler2DTransform()
            transform.SetAngle(rotation*np.pi/180)
        else:
            transform = sitk.Euler3DTransform()
            transform.SetRotation(rotation[0]*np.pi/180, rotation[1]*np.pi/180, rotation[2]*np.pi/180)

        transform.SetCenter(self.__GetImageCenter())

        self.WelcomeNewTransformation(transform, transform.GetInverse())

        return self


    def GivenRigidTransform(self, translation=None, rotation=None):
        # 2D or 3D?
        if self.img0.GetDimension() == 2:
            transform = sitk.Euler2DTransform()
            if rotation:
                transform.SetAngle(rotation*np.pi/180) 
        else:
            transform = sitk.Euler3DTransform()
            if rotation:
                transform.SetRotation(rotation[0]*np.pi/180, rotation[1]*np.pi/180, rotation[2]*np.pi/180)

        transform.SetCenter(self.__GetImageCenter())

        if translation:
            transform.SetTranslation(np.double(np.array(translation)))

        self.WelcomeNewTransformation(transform, transform.GetInverse())       
        
        return self


    def __CreateRandomVelocityField(self, step, sigma):
        shape = self.img0.GetSize()

        # create random velocity field
        if self.img0.GetDimension() == 2:
            vx_array = 2 * step * (np.random.random(shape)-0.5).T
            vy_array = 2 * step * (np.random.random(shape)-0.5).T
            v_array = np.stack((vx_array,vy_array), axis=-1)
        else:
            vx_array = 2 * step * (np.random.random(shape)-0.5).T
            vy_array = 2 * step * (np.random.random(shape)-0.5).T
            vz_array = 2 * step * (np.random.random(shape)-0.5).T
            v_array = np.stack((vx_array,vy_array,vz_array), axis=-1)

        # get random velocity field as sitk img
        v = sitk.GetImageFromArray(v_array, isVector=True)
        v.CopyInformation(self.img0)

        # smooth velocity field
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigma)
        v = gaussian.Execute(v)

        return v

    def RandomNonrigidWarpImage(self, step=3, sigma=3, angle=0, useGST=False):

        if step == 0:
            return self
        else:
            # get random velocity field
            if useGST:
                v = self.__CreateGSTRandomVelocityField(step, sigma, angle)
            else:
                v = self.__CreateRandomVelocityField(step, sigma)

        # Find n, scaling parameter
        v_array = sitk.GetArrayFromImage(v)

        if self.img0.GetDimension() == 2:
            vx_array = v_array[:,:,0]
            vy_array = v_array[:,:,1]
            normv2 = vx_array**2 + vy_array**2
        else:
            vx_array = v_array[:,:,0]
            vy_array = v_array[:,:,1]
            vz_array = v_array[:,:,2]
            normv2 = vx_array**2 + vy_array**2 + vz_array**2
            
        m = np.sqrt(np.max(normv2))
        n = np.ceil(np.log2(m/0.5))

        # Scale it (so it's close to 0)
        v_array = v_array * (2**(-n))
        little_v = sitk.GetImageFromArray(v_array.astype(np.double), isVector=True)
        little_v.CopyInformation(self.img0)
        little_v_inverse = sitk.GetImageFromArray(-v_array.astype(np.double), isVector=True)
        little_v_inverse.CopyInformation(self.img0)

        # square it n times
        s = sitk.DisplacementFieldTransform(little_v)
        s_inverse = sitk.DisplacementFieldTransform(little_v_inverse)

        trans2field = sitk.TransformToDisplacementFieldFilter()
        trans2field.SetReferenceImage(self.img0)

        for __ in range(int(n)):
            s_temp = sitk.CompositeTransform([s,s])
            s_inverse_temp = sitk.CompositeTransform([s_inverse,s_inverse])

            # 这么做的原因是：如果用CompositeTransform进行不断AddTransform，它不会算中间结果，最后会累加为1000多个场，很耗时
            # reason for this: if you use CompositeTransform to keep adding transforms, it won't calculate the intermediate results, and finally it will accumulate to 1000+ fields, which is very time-consuming
            s = sitk.DisplacementFieldTransform(trans2field.Execute(s_temp))
            s_inverse = sitk.DisplacementFieldTransform(trans2field.Execute(s_inverse_temp))

        self.WelcomeNewTransformation(s, s_inverse)

        return self


    def __CreateGSTRandomVelocityField(self, step=3, sigma=3, angle=0):

        shape = self.img0.GetSize()
        model = gs.Gaussian(dim=self.img0.GetDimension(), var=step, len_scale=sigma, angles=angle)
        srf = gs.SRF(model, generator="IncomprRandMeth")

        if self.img0.GetDimension() == 2:

            x = np.arange(shape[0]) 
            y = np.arange(shape[1])
            field_array = srf((x, y), mesh_type="structured", post_process=False)

        elif self.img0.GetDimension() == 3:

            x = np.arange(shape[0])
            y = np.arange(shape[1])
            z = np.arange(shape[2])
            field_array = srf((x, y, z), mesh_type="structured", post_process=False)

        field = sitk.GetImageFromArray(field_array.transpose(), isVector=True)
        field.CopyInformation(self.img0)

        return field


class Assessment(object):
    def __init__(self, f_seg_list, m_seg_list):
        self.f_seg_list = f_seg_list if isinstance(f_seg_list, list) else [f_seg_list]
        self.m_seg_list = m_seg_list if isinstance(m_seg_list, list) else [m_seg_list]


    def GetDICE(self, index=None):

        if index:
            diceFilter = sitk.LabelOverlapMeasuresImageFilter()
            diceFilter.Execute(self.f_seg_list[index], self.m_seg_list[index])
            return diceFilter.GetDiceCoefficient()
        else:
            DICE_list = []
            for f_seg, m_seg in zip(self.f_seg_list, self.m_seg_list):
                diceFilter = sitk.LabelOverlapMeasuresImageFilter()
                diceFilter.Execute(f_seg, m_seg)
                DICE_list.append(diceFilter.GetDiceCoefficient())
            return DICE_list


    def GetHDD(self, index=None):

        if index:
            hausdorffFilter = sitk.HausdorffDistanceImageFilter()
            hausdorffFilter.Execute(self.f_seg_list[index], self.m_seg_list[index])
            return hausdorffFilter.GetHausdorffDistance()
        else:
            HDD_list = []
            for f_seg, m_seg in zip(self.f_seg_list, self.m_seg_list):
                hausdorffFilter = sitk.HausdorffDistanceImageFilter()
                hausdorffFilter.Execute(f_seg, m_seg)
                HDD_list.append(hausdorffFilter.GetHausdorffDistance())
            return HDD_list


    def AssessRegistration(self):

        for i, [dice, hdd] in enumerate(zip(self.GetDICE(), self.GetHDD())):
            print('#', i, ' seg:')

            print('DICE: ', dice)
            print('HDD: ', hdd)
