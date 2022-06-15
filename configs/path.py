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
elif hostname == "lihuoxingdeMBP.local":
    voxroot = "/Users/chenyangli/Codes/3dml-project/data/ShapeNetVox32"
    imgroot = "/Users/chenyangli/Codes/3dml-project/data/ShapeNetRendering"
    logroot = "./logs"
elif hostname == "":
    voxroot = ""
    imgroot = ""
    logroot = "./logs"
