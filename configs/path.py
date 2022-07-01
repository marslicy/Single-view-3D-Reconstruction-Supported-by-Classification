import getpass
import socket

username = getpass.getuser()

hostname = socket.gethostname()
voxroot = "/home/yang/GitHub/ml3d_final_dataset/ShapeNetVox32"
imgroot = "/home/yang/GitHub/ml3d_final_dataset/ShapeNetRendering"
logroot = "./logs"
if hostname == "iWorkstation":
    voxroot = "/home/yang/Codes/ml3d_final_dataset/ShapeNetVox32"
    imgroot = "/home/yang/Codes/ml3d_final_dataset/ShapeNetRendering"
    logroot = "./logs"
elif hostname == "lihuoxingdeMBP.local":
    voxroot = "/Users/chenyangli/Codes/3dml-project/data/ShapeNetVox32"
    imgroot = "/Users/chenyangli/Codes/3dml-project/data/ShapeNetRendering"
    logroot = "./logs"
elif hostname == "container-4ae511b752-7e696d3b":
    voxroot = "/root/autodl-tmp/3dml-project/data/ShapeNetVox32"
    imgroot = "/root/autodl-tmp/3dml-project/data/ShapeNetRendering"
    logroot = "./logs"
