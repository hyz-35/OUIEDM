
<div align=left><div>

# OUIEDM: Underwater Image Enhancement with One-step Diffusion Model

<div align=left><div>

## Datasets
Real Underwater Data including LSUI and UIEBD for [google drive] and [Baidu Drive] are released.

We use a underwater camera captured real-world underwater video in Shenzhen, Guangdong province, China. The underwater video includes 300 frames for [google drive] and [Baidu Drive] are released. If you want to use the  underwater video, please cite our [paper].
<div align=left><div>
  
## Underwater Enhancement and dehazing

<div align=center><img src="images_effect\effect.jpg" width="500" height="400" > <img src="images_effect\effect.gif" width="500" height="400" >

<div align=left><div>

## Testing
You can download the pretrain models in [Google Drive].

After downloading, extract the pretrained model into the project folder and replace the checkpoint folder, and then run test.OUIEDM.py. 
The code will use the pretrained model to automatically process all the images in your input folder and output the results to your output folder. 

For the test environmental requirements, you can install from the requirements.txt using the following code or build the environmental requirements on your own.
```python
pip install -r requirements.txt
