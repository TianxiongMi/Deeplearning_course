# Deeplearning_course
Distinguish human actions: talking to people and talking on phone <br />
This post uses Pytorch to write a stacked LSTM RNN neural network to distinguish the two actions. <br />
The original json files can be transformed by read_json.py <br />
The two csv files can be directly used in deeplearning_homework to get training result. <br />
The csv files for the 5 test videos are uploaded in a zip file. <br />
A trained model is uploaded: in order to use it, you need to download the 'checkpoint.pth' first, load it in the script, and then choose any video you like, run read_json.py to transform it into a csv file, and make sure you use the right length and timestep in the function: generate_sample_video_graph. <br />
Reference: https://github.com/SmitSheth/Human-Activity-Recognition
