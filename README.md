# Neural Point Light Fields (CVPR 2022)

<img src="https://light.princeton.edu/wp-content/uploads/2022/03/overview_pointLF.png">

### [Project Page](https://light.princeton.edu/publication/neural-point-light-fields) 
#### Julian Ost, Issam Laradji, Alejandro Newell, Yuval Bahat, Felix Heide


Neural Point Light Fields represent scenes with a light field living on a sparse point cloud. As neural volumetric 
rendering methods require dense sampling of the underlying functional scene representation, at hundreds of samples 
along with a ray cast through the volume, they are fundamentally limited to small scenes with the same objects 
projected to hundreds of training views. Promoting sparse point clouds to neural implicit light fields allows us to 
represent large scenes effectively with only a single implicit sampling operation per ray.

These point light fields are a function of the ray direction, and local point feature neighborhood, allowing us to 
interpolate the light field conditioned training images without dense object coverage and parallax. We assess the 
proposed method for novel view synthesis on large driving scenarios, where we synthesize realistic unseen views that 
existing implicit approaches fail to represent. We validate that Neural Point Light Fields make it possible to predict 
videos along unseen trajectories previously only feasible to generate by explicitly modeling the scene.

---

### Data Preparation
#### Waymo

1. Download the compressed data of the Waymo Open Dataset:
[Waymo Validation Tar Files](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_3_1/validation?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)

2. To run the code as used in the paper, store as follows: `./data/validation/validation_xxxx/segment-xxxxxx`
   
3. Neural Point Light Fields is well tested on the segments mentioned in the [Supplementary](https://light.princeton.edu/wp-content/uploads/2022/04/NeuralPointLightFields-Supplementary.pdf) and shown in the experiment group `pointLF_waymo_local`.

---

### Requirements

Environment setup
```
conda create -n NeuralPointLF python=3.7
conda activate NeuralPointLF
```
Install required packages
```
conda install -c pytorch -c conda-forge pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install jupyterlab
pip install scikit-image matplotlib imageio plotly opencv-python
conda install pytorch3d -c pytorch3d
conda install -c open3d-admin -c conda-forge open3d
```
Add large-scale ML toolkit
```
pip install --upgrade git+https://github.com/haven-ai/haven-ai
```

---
### Training and Validation
In the first run of a scene, the point clouds will be preprocessed, which might take some time. 
If you want to train on unmerged point cloud data set `merge_pcd=False` in the config file. 

Train one specific scene from the Waymo Open data set:
```
python trainval.py -e pointLF_waymo_server -sb <save dir> -d ./data/waymo/validation --epoch_size
100 --num_workers=<num available CPU workers for pcd preprocessing>
```

Reconstruct the training path (`--render_only=True`)
```
python trainval.py -e pointLF_waymo_server -sb <save dir> -d ./data/waymo/validation --epoch_size
100 --num_workers=<num available CPU workers for pcd preprocessing> --render_only=True
```

Argument Descriptions:
```
-e  [Experiment group to run like 'mushrooms' (the rest of the experiment groups are in exp_configs/sps_exps.py)] 
-sb [Directory where the experiments are saved]
-r  [Flag for whether to reset the experiments]
-d  [Directory where the datasets are aved]
```

**_Disclaimer_**: The codebase is optimized to run on larger GPU servers with a lot of free CPU memory. 
To test on local and low memory, choose `pointLF_waymo_local` instead of `pointLF_waymo_server`. 
Adjustments of batch size, chunk size and number of rays will have an effect on necessary resources.

---
### Visualization of Results

Follow these steps to visualize plots. Open `results.ipynb`, run the first cell to get a dashboard like in the gif below, click on the "plots" tab, then click on "Display plots". Parameters of the plots can be adjusted in the dashboard for custom visualizations.

<p align="center" width="100%">
<img width="65%" src="scripts/vis.gif">
</p>

---
#### Citation
```
@InProceedings{ost2022pointlightfields,
    title   = {Neural Point Light Fields},
    author  = {Ost, Julian and Laradji, Issam and Newell, Alejandro and Bahat, Yuval and Heide, Felix},
    journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year    = {2022}
}
```


