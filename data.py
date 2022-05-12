import splitfolders
# test, train, val
splitfolders.ratio("Images", output="Augmented",seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)
# augment
import torchvision.transforms as T
# training augmentation types: horizontal flip, rotation, crop, normalize
# 12 augmentation strategies
train_transform = T.Compose(T.GaussianBlur(),
                        T.ColorJitter(),
                        T.RandomHorizontalFLip(),
                        T.RandomRotation(),
                        T.RandomVerticalFlip(),
                        T.GaussianBlur(),
                        T.LinearTransformation(),
                        T.adjust_brightness(),
                        T.adjust_contrast(),
                        T.adjust_hue(),
                        T.adjust_saturation(),
                        T.adjust_sharpness())
