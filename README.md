# SMIRK Hugging Face Implementation

This is the hugging face implementation of the paper and project included within this repo: [https://github.com/georgeretsi/smirk?tab=readme-ov-file](https://github.com/georgeretsi/smirk?tab=readme-ov-file).

# Instructions to Host On Hugging Face:
- Download this github repo as a zip  
- Create a new hugging face space  
- Place the pretrained model under a pretrained_models folder within the root  
- Ensure you are using a GPU  
- To upload/remove images/videos, enter the samples folder at the root and choose the files you want to add/delete and it will re run to show the 3d mesh  
- A Working version can be found here: [https://huggingface.co/spaces/icampara/6166Mobile](https://huggingface.co/spaces/icampara/6166Mobile)  

- Also download the zip smirkmobile2.zip  
- Unzip the file  
- Open the folder in a IDE and ensure expo is imported  
- Run `npx expo start` (assuming you have node installed)  
- Scan the QR code after you download the Expo Go application  
- Use the application which computes from the hugging face (ensure you are on the same network)  
- Upload video and photo and you will be presented with the mesh  

Demo should appear as so:  
![ScreenRecording_10-31-202518-30-40_1-ezgif com-video-to-gif-converter (1)](https://github.com/user-attachments/assets/dfa2244e-56f0-41e8-b9d6-db6fd0b05b53)
