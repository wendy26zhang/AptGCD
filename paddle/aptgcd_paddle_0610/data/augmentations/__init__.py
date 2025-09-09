import paddle


def get_transform(transform_type="imagenet", image_size=32, args=None):
    if transform_type == "imagenet":
        mean = 0.485, 0.456, 0.406
        std = 0.229, 0.224, 0.225
        interpolation = args.interpolation
        crop_pct = args.crop_pct
        train_transform = paddle.vision.transforms.Compose(
            transforms=[
                paddle.vision.transforms.Resize(
                    size=int(image_size / crop_pct), interpolation=interpolation
                ),
                paddle.vision.transforms.RandomCrop(size=image_size),
                paddle.vision.transforms.RandomHorizontalFlip(prob=0.5),
                paddle.vision.transforms.ColorJitter(),
                paddle.vision.transforms.ToTensor(),
                paddle.vision.transforms.Normalize(
                    mean=paddle.to_tensor(data=mean), std=paddle.to_tensor(data=std)
                ),
            ]
        )
        test_transform = paddle.vision.transforms.Compose(
            transforms=[
                paddle.vision.transforms.Resize(
                    size=int(image_size / crop_pct), interpolation=interpolation
                ),
                paddle.vision.transforms.CenterCrop(size=image_size),
                paddle.vision.transforms.ToTensor(),
                paddle.vision.transforms.Normalize(
                    mean=paddle.to_tensor(data=mean), std=paddle.to_tensor(data=std)
                ),
            ]
        )
    else:
        raise NotImplementedError
    return train_transform, test_transform
