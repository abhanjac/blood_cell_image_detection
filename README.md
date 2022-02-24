# Objective: 
This project shows a deep learning based approach to detect and identify different types of blood cells and also detect the presence of malarial parasites in the blood. This algorithm takes in digitized blood smear images and then uses modified versions of YoloV2 model to identify the different objects with an overall mean average precision of 0.95.

**A *Trailer* of Final Result:**

---

This project is to create a neural network to classify and detect the locations of different types of blood cells from a digital blood smear image.

The neural network used here is a modified version of the [Yolo-V2](https://arxiv.org/abs/1612.08242v1) convolutional neural network. The pdf of the paper is also present [here](extra_files/YoloV2_paper.pdf).
The whole idea is that, a big digitized blood smear image (may be 1000 x 1000 pixels) will be obtained from a digital microscope and then it will be broken up into **224 x 224** segments. These segments will be given to the network to detect and classify the different objects. Then after the inference is done, these 224 x 224 segments are stitched back to the size of the original image (from the microscope). The RGB colored image obtained from a digital microscope showing the different blood cells in a blood smear on a slide at **40x** magnification.
The locations of the identified objects are marked by the network using bounding boxes. So there has to be ground truth bounding box information in the training dataset as well.
The network is first trained using a training dataset and validated with a validation dataset. After that it has to tested on a completely unseen test dataset to check its performance.
The images in the dataset used here has the following types of cells: **2** types of RBCs: **Infected RBC** (RBCs infected with malarial parasite), **Healthy RBC**; 
**5** types of WBCs: **Eosinophil**, **Basophil**, **Neutrophil**, **Lymphocytes**, and **Monocytes**; 
**2** types of Platelets: **Thrombocytes** (individual platelet cells) and **Platelet Clumps** (bunch of platelets appearing as a cluster).
So overall there are **9** objects.


