import torch
import torch.utils.data as data
from PIL import Image
import os
import csv

def pil_loader(path):
    """
    Open path as file to avoid ResourceWarning and handle potential image errors gracefully.
    :param path: image path
    :return: image data or None if the image is corrupted
    """
    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('L')
    except (IOError, Image.UnidentifiedImageError):
        print(f"Warning: Skipping corrupted image file {path}")
        return None  # 返回 None 表示这个图像无效

def accimage_loader(path):
    """
    compared with PIL, accimage loader eliminates useless function within class, so that it is faster than PIL
    :param path: image path
    :return: image data
    """
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    """
    choose accimage as image loader if it is available, PIL otherwise
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def get_video(video_path, frame_indices):
    """
    generate a video clip which is a list of selected frames
    :param video_path: path of video folder which contains video frames
    :param frame_indices: list of selected indices of frames. e.g. if index is 1, then selected frame's name is "img_1.png"
    :return: a list of selected frames which are PIL.Image or accimage form
    """
    image_reader = get_default_image_loader()
    video = []
    for image_index in frame_indices:
        image_name = 'img_' + str(image_index) + '.png'
        image_path = os.path.join(video_path, image_name)
        img = image_reader(image_path)
        video.append(img)
    return video

def get_clips(video_path, video_begin, video_end, label, view, sample_duration):
    """
    be used when validation set is generated. be used to divide a video interval into video clips
    :param video_path: validation data path
    :param video_begin: begin index of frames
    :param video_end: end index of frames
    :param label: 1(normal) / 0(anormal)
    :param view: front_depth / front_IR / top_depth / top_IR
    :param sample_duration: how many frames should one sample contain
    :return: a list which contains  validation video clips
    """
    clips = []
    sample = {
        'video': video_path,
        'label': label,
        'subset': 'validation',
        'view': view,
    }
    interval_len = (video_end - video_begin + 1)
    num = int(interval_len / sample_duration)
    for i in range(num):
        sample_ = sample.copy()
        sample_['frame_indices'] = list(range(video_begin, video_begin + sample_duration))
        clips.append(sample_)
        video_begin += sample_duration
    if interval_len % sample_duration != 0:
        sample_ = sample.copy()
        sample_['frame_indices'] = list(range(video_begin, video_end+1)) + [video_end] * (sample_duration - (video_end - video_begin + 1))
        clips.append(sample_)
    return clips


def listdir(path):
    """
    show every files or folders under the path folder
    """
    for f in os.listdir(path):
            yield f


def make_dataset(root_path, subset, view, sample_duration, type=None):
    """
    :param root_path: root path of the dataset"
    :param subset: train / validation
    :param view: front_depth / front_IR / top_depth / top_IR
    :param sample_duration: how many frames should one sample contain
    :param type: during training process: type = normal / anormal ; during validation or test process: type = None
    :return: list of data samples, each sample is in form {'video':video_path, 'label': 0/1, 'subset': 'train'/'validation', 'view': 'front_depth' / 'front_IR' / 'top_depth' / 'top_IR', 'action': 'normal' / other anormal actions}
    """
    dataset = []
    if subset == 'train' and type == 'normal':
        # load normal training data
        train_folder_list = list(filter(lambda string: string.find('Tester') != -1, list(listdir(root_path))))

        for train_folder in train_folder_list:
            normal_video_list = list(filter(lambda string: string.split('_')[0] == 'normal', list(listdir(os.path.join(root_path, train_folder)))))

            for normal_video in normal_video_list:
                video_path = os.path.join(root_path, train_folder, normal_video, view)
                if not os.path.exists(video_path):
                    print(f"Video path doesn't exit: {video_path}")
                    continue

                n_frames = len(os.listdir(video_path))
                if n_frames <= 0:
                    print(f"Path {video_path} does't contain any data")
                    continue

                sample = {
                    'video': video_path,
                    'label': 1,
                    'subset': 'train',
                    'view': view,
                    'action': 'normal'
                }
                for i in range(0, n_frames, sample_duration):
                    sample_ = sample.copy()
                    sample_['frame_indices'] = list(range(i, min(n_frames, i + sample_duration)))
                    if len(sample_['frame_indices']) < sample_duration:
                        for j in range(sample_duration-len(sample_['frame_indices'])):
                            sample_['frame_indices'].append(sample_['frame_indices'][-1])
                    dataset.append(sample_)


    elif subset == 'train' and type == 'anormal':
        #load anormal training data
        train_folder_list = list(filter(lambda string: string.find('Tester') != -1, list(listdir(root_path))))

        for train_folder in train_folder_list:
            anormal_video_list = list(filter(lambda string: string.split('_')[0] != 'normal', list(listdir(os.path.join(root_path, train_folder)))))

            for anormal_video in anormal_video_list:
                video_path = os.path.join(root_path, train_folder, anormal_video, view)
                if not os.path.exists(video_path):
                    print(f"Video path doesn't exit: {video_path}")
                    continue
                n_frames = len(os.listdir(video_path))
                if n_frames <= 0:
                    print(f"Path {video_path} does't contain any data")
                    continue
                sample = {
                    'video': video_path,
                    'label': 0,
                    'subset': 'train',
                    'view': view,
                    'action': anormal_video,
                }

                for i in range(0, n_frames, sample_duration):
                    sample_ = sample.copy()
                    sample_['frame_indices'] = list(range(i, min(n_frames, i + sample_duration)))
                    if len(sample_['frame_indices']) < sample_duration:
                        for j in range(sample_duration-len(sample_['frame_indices'])):
                            sample_['frame_indices'].append(sample_['frame_indices'][-1])

                    dataset.append(sample_)

    elif subset == 'validation' and type == None:
        #load valiation data as well as thier labels
        csv_path = os.path.join(root_path, 'LABEL.csv')
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[-1] == '':
                    continue
                if row[0] != '':
                    which_val_path = os.path.join(root_path, row[0].strip())
                if row[1] != '':
                    video_path = os.path.join(which_val_path, row[1], view)
                video_begin = int(row[2])
                video_end = int(row[3])
                if row[4] == 'N':
                    label = 1
                elif row[4] == 'A':
                    label = 0
                clips = get_clips(video_path, video_begin, video_end, label, view, sample_duration)
                dataset = dataset + clips
    else:
        print('!!!DATA LOADING FAILURE!!!CANT FIND CORRESPONDING DATA!!!PLEASE CHECK INPUT!!!')
    return dataset


class DAD(data.Dataset):
    """
    Generate normal training / abnormal training / validation dataset according to requirement.
    """
    def __init__(self,
                 root_path,
                 subset,
                 view,
                 sample_duration=16,
                 expected_length=16,
                 type=None,
                 get_loader=get_video,
                 spatial_transform=None,
                 temporal_transform=None,
                 last_valid_clip=None,
                 ):
        self.data = make_dataset(root_path, subset, view, sample_duration, type)
        self.sample_duration = sample_duration
        self.subset = subset
        self.loader = get_loader
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.expected_length = 16
        self.last_valid_clip=None

    def __getitem__(self, index):
        if self.subset == 'train':
            video_path = self.data[index]['video']
            frame_indices = self.data[index]['frame_indices']

            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)

            # 从 loader 中加载图片
            clip = self.loader(video_path, frame_indices)
            #
            # # 输出每张图像的路径和文件名
            # for i, img in enumerate(clip):
            #     if img is not None:
            #         frame_path = os.path.join(video_path, f"{frame_indices[i]:04d}.png")
            #         print(f"Loaded image from {frame_path}")
            #     else:
            #         print(f"Skipping None image at frame {frame_indices[i]} in {video_path}")



            # 过滤掉加载失败的图像
            clip = [self.spatial_transform(img) for img in clip if img is not None]
            # if len(clip) == 0:  # 如果所有图像都加载失败，则跳过这个 index
            #     print(f"Skipping corrupted trainclip at index {index}")
            #     return None, None

            if len(clip) == 0:
                if self.last_valid_clip is not None:
                    # 使用上一个有效 clip 的最后一张图片进行填充
                    print(f"Using last valid trainclip to fill corrupted clip at index {index}")
                    clip = [self.last_valid_clip[-1]] * self.expected_length
                else:
                    # 如果没有上一个有效的 clip，可以选择跳过该样本或抛出异常
                    print(f"Skipping corrupted trainclip at index {index}")
                    return None, None

            # 检查clip的长度是否满足要求
            if len(clip) < self.expected_length:
                print(f"incomplete trainclip at index {index} with length {len(clip)}")
                clip += [clip[-1]] * ( self.expected_length - len(clip))

            # # 输出每个图像的 shape 信息
            # for i, img in enumerate(clip):
            #     if img is not None:
            #         print(f"Image at index {i} in clip has shape: {img.size()}")

            # 在 stack 前再次检查所有图像形状
            # for i, img in enumerate(clip):
            #     if img.size() != torch.Size([1, 112, 112]):  # 根据预期形状调整
            #         print(f"Unexpected shape at index {i} in clip at index {index}: {img.size()}")

            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)  # (channels, timesteps, height, width)
            # 验证 shape 是否符合要求 [1, 16, 112, 112]
            expected_shape = torch.Size([1, self.expected_length, 112, 112])
            if clip.shape != expected_shape:
                print(f"Error: Unexpected shape {clip.shape} at index {index}. Adjusting shape.")

                # 如果通道数不正确（应为1），则取第一个通道
                if clip.shape[0] != 1:
                    clip = clip[:1, :, :, :]

                # 如果时间步长不正确（应为self.expected_length），进行填充或截断
                if clip.shape[1] < self.expected_length:
                    clip = torch.cat([clip, clip[:, -1:, :, :].repeat(1, self.expected_length - clip.shape[1], 1, 1)],
                                     dim=1)
                elif clip.shape[1] > self.expected_length:
                    clip = clip[:, :self.expected_length, :, :]

                # 最后检查并打印修正后的形状
                print(f"Adjusted shape: {clip.shape}")
            #print(f"trainclip：{len(clip)}")
            #print(f"Training sample at index {index} - Final clip shape before returning: {clip.shape}")

            # 更新 last_valid_clip
            self.last_valid_clip = clip

            return clip, index

        elif self.subset == 'validation':
            video_path = self.data[index]['video']
            ground_truth = self.data[index]['label']
            frame_indices = self.data[index]['frame_indices']

            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)

            clip = self.loader(video_path, frame_indices)

            # 输出每张图像的路径和文件名
            #for i, img in enumerate(clip):
                #if img is not None:
                    #frame_path = os.path.join(video_path, f"{frame_indices[i]:04d}.png")
                    #print(f"Loaded image from {frame_path}")
                #else:
                    #print(f"Skipping None image at frame {frame_indices[i]} in {video_path}")

            # 过滤掉加载失败的图像
            clip = [self.spatial_transform(img) for img in clip if img is not None]
            if len(clip) == 0:
                print(f"Skipping corrupted valclip at index {index}")
                return None, None

            # 检查clip的长度是否满足要求
            if len(clip) < self.expected_length:
                print(f"incomplete valclip at index {index} with length {len(clip)}")
                clip += [clip[-1]] * (self.expected_length - len(clip))

            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            #print(f"valclip：{len(clip)}")

            self.last_valid_clip = clip

            return clip, ground_truth

        else:
            print('!!!DATA LOADING FAILURE!!! CANT FIND CORRESPONDING DATA!!! PLEASE CHECK INPUT!!!')

    def __len__(self):
        return len(self.data)  # 返回数据集的样本数量