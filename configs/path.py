import getpass
import socket

username = getpass.getuser()

hostname = socket.gethostname()
voxroot = "./data/ShapeNetVox32"
imgroot = "./data/ShapeNetRendering"
logroot = "./logs"
priorroot = "./data/prior"
if hostname == "iWorkstation":
    voxroot = "/home/yang/Codes/ml3d_final_dataset/ShapeNetVox32"
    imgroot = "/home/yang/Codes/ml3d_final_dataset/ShapeNetRendering"
    logroot = "./logs"
    priorroot = "./data/prior"
elif username == "chenyangli":
    voxroot = "/Users/chenyangli/Codes/3dml-project/data/ShapeNetVox32"
    imgroot = "/Users/chenyangli/Codes/3dml-project/data/ShapeNetRendering"
    priorroot = "/Users/chenyangli/Codes/3dml-project/data/prior"
    logroot = "./logs"
