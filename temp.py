import urllib.request
import subprocess

# 定义Python版本号
python_version = "3.10.6"

# 根据操作系统选择合适的Python安装程序下载链接
download_link = "https://www.python.org/ftp/python/{0}/python-{0}-amd64.exe".format(python_version)

# 下载Python安装程序
urllib.request.urlretrieve(download_link, "python-{0}-amd64.exe".format(python_version))

# 执行Python安装程序
subprocess.run(["python-{0}-amd64.exe".format(python_version), "/passive", "InstallAllUsers=1", "PrependPath=1"])

# 删除下载的安装程序
subprocess.run(["del", "python-{0}-amd64.exe".format(python_version)])