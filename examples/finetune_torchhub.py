'''
A script to finetune the convnextv2-atto model on any dataset using torch hub.

Available models:
- convnextv2_atto
- convnextv2_unet_atto

Available checkpoints:
- pt-all_mod_atto_1M_64_uncertainty_56-8
- pt-all_mod_atto_1M_64_unweighted_56-8
- pt-all_mod_atto_1M_128_uncertainty_112-16
- pt-S2_atto_1M_64_uncertainty_56-8

Input to all models is Sentinel-2 imagery with 12 channels.

'''


import torch
from timm.loss import LabelSmoothingCrossEntropy
import torchmetrics



def load_geobench_dataloaders():
    from geobench import GEO_BENCH_DIR
    from geobenchdataset import get_geobench_dataloaders
    processed_dir = GEO_BENCH_DIR
    (train_dataloader, val_dataloader), task = get_geobench_dataloaders(
        dataset_name='m-eurosat', # dataset name (m-bigearthnet, m-eurosat, m-cashew-plant, m-SA-crop-type)
        processed_dir=processed_dir, # directory to save processed beton files if we use FFCV
        num_workers=16, # number of worker threads
        batch_size_per_device=4, # batch size
        splits=["train", "val"],
        partition="default",
        geobench_bands_type="full", # use all 13 bands
        no_ffcv=True, # use standard pytorch dataloader
        seed=0,
    )
    return train_dataloader, val_dataloader, task


def main():

    # dataloaders
    train_dataloader, val_dataloader, task = load_geobench_dataloaders()
    num_classes = task.num_classes
    samples, targets, _, _ = next(iter(train_dataloader))
    in_channels = samples.shape[1]
    print('in_channels:', in_channels)
    print('num_classes:', num_classes)

    ########## model ##########
    mmearth_model = torch.hub.load('vishalned/mmearth-train', 
                                   'MPMAE', 
                                   model_name='convnextv2_atto', # convnextv2_unet_atto for segmentation tasks
                                   ckpt_name='pt-all_mod_atto_1M_64_uncertainty_56-8',
                                   pretrained=False,
                                   linear_probe=True,
                                   num_classes=num_classes,
                                   in_chans=in_channels,
                                   patch_size=8,
                                   img_size=56)
    
    n_parameters = sum(p.numel() for p in mmearth_model.parameters() if p.requires_grad)
    print("Model = %s" % str(mmearth_model))
    print("number of params:", n_parameters)
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.2)

    # TODO: uncomment this for segmentation tasks
    # criterion = torch.nn.CrossEntropyLoss()


    optimizer = torch.optim.AdamW(mmearth_model.parameters(), lr=0.001, weight_decay=0.3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    metric = torchmetrics.MetricCollection({"Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average="micro")})

    # TODO: uncomment this for Jaccard metric i.e for segmentation tasks
    # metric = torchmetrics.MetricCollection({"Jaccard": torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, average="macro")})

    # train
    for epoch in range(100):
        mmearth_model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            samples = batch[0]
            targets = batch[1]
            optimizer.zero_grad()
            outputs = mmearth_model(samples)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            score = metric(outputs, targets)
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}, Accuracy: {score["Accuracy"]}')
        epoch_metrics = metric.compute()
        print(epoch_metrics)


        # evaluate
        mmearth_model.eval()
        with torch.no_grad():
            for batch_idx, (samples, targets) in enumerate(val_dataloader):
                outputs = mmearth_model(samples)
                loss = criterion(outputs, targets)
                score = metric(outputs, targets)
                print(f'Validation: Loss: {loss.item()}, Accuracy: {score["Accuracy"]}')
            epoch_metrics = metric.compute()
            print(epoch_metrics)
   

    
    



if __name__ == "__main__":
    main()
