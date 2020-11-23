### Code for the 2019 november workshops's at the City of Science and Industry.

--- 

The code in this repo was realized in a week-end in November 2019, for two 
AI workshops at the City of Science and Industry in Paris. Do not expect 
the code to be clean, efficient, or up to date with current Python or AI standards.

The code has been made and tested with Python 3.6 and PyTorch 1.3.0
If you intend to use this code as part of a project, it would be a 
good idea to upgrade to the newest versions of Python and PyTorch.

The 'primary' file is the main_workshop_code.py file, which contains all the 
executive logic to run a pre-trained yolo3 object recognition neural network model.
It was coded in live with the audiences at the workshop's in under 45 minutes.
Expect french comments in this file. Do not hesitate to remove them, they hav
little to no importance for someone with even the primary bases of Python. 

---

### Requirements :
- Python 3.6 or higher (tested with Python 3.6)
- PyTorch 1.3.0 or higher (tested with PyTorch 1.3.0)
- TorchVision 0.4.1 or higher (tested with Torchvision 0.4.1)
- CV2
- PIL

---

### Installation :
After installing the requirements to run the code, you will need the weights of the
object recognition neural network. **The file is 236MB large (too big for GitHub, 
and need to be download manually).** 

Weights file download links :
- Official download (~50KBs download speed) : https://pjreddie.com/media/files/yolov3.weights
- Inoft download (~20MBs download speed) : https://inoft-open-source.s3.eu-west-3.amazonaws.com/AI-Workshop_City-of-Science-and-Industry_November-2019/yolov3.weights

The Inoft download link is provided for free by Inoft. If the link is down, 
send me a direct mail to `robinson@inoft.com`
---

### Notice :
The code in this repo is open-sourced under the MIT license. The text content in the 
presentation file too is also under the MIT license. Yet, the images in the folder 
'presentation/images' and the videos in the 'videos' folder (jean_claude.mp4 and 
isa.wmv) are not under MIT and cannot be used outside of your personal use.

---

#### While installing an old version of PyTorch

In case there is an issue with Pytorch, where the _C module cannot be imported, uninstall
pytorch and torchvision with `pip uninstall torch` and `pip uninstall torchvision`, then 
re-install them for the gpu if you are using the gpu, and only for the cpu if using the 
cpu : `pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html`

---

### Credits :

For the object recognition neural network, I used a pre-trainer Yolo3 model designed 
by Joseph Redmon and Ali Farhadi's, with an implementation by Erik Lindernoren’s.
https://pjreddie.com/media/files/papers/YOLOv3.pdf
https://pjreddie.com/darknet/yolo/


This article has been a huge help for making the workshop possible with such tight deadlines. A lot of 
the code in this repo has been inspired or copied from this article Some code has been taken from 
this article : https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98
