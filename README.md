# crda

Category and Relation Aware Data Augmentation for 3D Visual Grounding

## Installation

1. Please refer to the installation process of [Vil3dref](https://github.com/cshizhe/vil3dref).

2. For the scene augmentation data, please download the object feature from [here](https://drive.google.com/file/d/1VrOoqS-yHrd-9R54AIuRkjq5mGoU0xlD/view?usp=sharing) and put it under the `/og3d_src/` folder.

3. For the annotation augmentation data, please first refer to the `/datasets/referit3d/readme.md`. Then download the files ([1](https://drive.google.com/file/d/1qc9aeasdaQk-E2Z1i2lEiY5OdROoSDWz/view?usp=sharing) and [2](https://drive.google.com/file/d/13FdfrwAjfhjandPh4MYIWNrur_XrbAKU/view?usp=sharing)) and put them under the `/og3d_src/` folder.

## Training

1. Please refer to the training process of [Vil3dref](https://github.com/cshizhe/vil3dref) to train the teacher model and pointnet encoder.

* You can also use the pre-trained models:

    - `/data0/shared/lisizhe_tmp/vil3dref_result/datasets_vil3dref/exprs_neurips22/pcd_clf_pre/ckpts/model_epoch_95.pt`

    - `data0/shared/lisizhe_tmp/vil3dref_result/datasets_vil3dref/exprs_neurips22/gtlabels/nr3d-try/ckpts/model_epoch_50.pt`

2. Train the student model with 3d point clouds

```bash
configfile=configs/nr3d_gtlabelpcd_mix_model.yaml
python train_mix.py --config $configfile \
    --output_dir YOUR_OUTPUT_DIR \
    --resume_files THE_POINTNET_ENCODER_PATH \
    THE_TEACHER_MODEL_PATH
```

## Inference

1. One branch

```bash
configfile=configs/nr3d_gtlabelpcd_mix_model.yaml
python train_mix.py --config $configfile \
    --output_dir YOUR_OUTPUT_DIR \
    --resume_files THE_STUDENT_MODEL_PATH \
    --test
```

2. The extire model

    First use step 1 to gain the prediction of each branch. Then use the `/og3d_src/test_crda.py` to aggregrate the logits and get the final prediction.

* You can also use the pre-trained checkpoints to verify the results. (Overall Accuracy **65.78**)

    - `/data0/shared/lisizhe_tmp/vil3dref_result/datasets_vil3dref-0/exprs_neurips22/gtlabelpcd_mix/nr3d-aug0-masktxt/ckpts/model_epoch_98.pt`

    - `/home/lisizhe/vil3dref-baseline/datasets/exprs_neurips22/gtlabelpcd_mix/nr3d-aug50-p5n5/ckpts/model_epoch_91.pt`