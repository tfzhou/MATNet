def get_dataset_davis_youtube_ehem(args, split, image_transforms=None,
                                   target_transforms=None, augment=False,
                                   inputRes=None):

    from .davis2017_youtubevos_ehem import DAVISLoader as MyChosenDataset

    dataset = MyChosenDataset(args, split=split, transform=image_transforms,
                              target_transform=target_transforms,
                              augment=augment, inputRes=inputRes)
    return dataset
