echo 'Updating your packages'
sudo apt-get update && sudo apt-get upgrade -y
echo 'Installing Python and the Python Package Installer'
sudo apt-get install python3 python3-pip -y
echo 'Downloading code from Github'
curl https://codeload.github.com/qwertpi/iris-ai/zip/master -o iris.zip
echo 'unzipping code'
unzip iris.zip
cd iris-master
echo 'Installing the required python libraries'
sudo pip3 install -r requirements.txt
sudo apt-get install libhdf5-serial-dev
echo 'Install complete'
