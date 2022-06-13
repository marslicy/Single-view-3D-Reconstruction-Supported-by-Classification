import getpass
import socket

username = getpass.getuser()

hostname = socket.gethostname()
voxroot = "/home/yang/GitHub/ml3d_final_dataset/ShapeNetVox32"
imgroot = "/home/yang/GitHub/ml3d_final_dataset/ShapeNetRendering"
logroot = "./logs"
if hostname == "iWorkstation":
    voxroot = "/home/yang/GitHub/ml3d_final_dataset/ShapeNetVox32"
    imgroot = "/home/yang/GitHub/ml3d_final_dataset/ShapeNetRendering"
    logroot = "./logs"
elif hostname == "":
    voxroot = ""
    imgroot = ""
    logroot = "./logs"
elif hostname == "":
    voxroot = ""
    imgroot = ""
    logroot = "./logs"
