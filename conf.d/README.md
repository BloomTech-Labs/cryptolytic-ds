Within the ssh login provided, you can access the API under the directory given, This is a map of the configuration files\
which you can find inside of the system, by either running the command the file is named (crontab -e for example.) In the\
folders, they are located in their respective conf.d directories, nginx for example is /etc/nginx/conf.d, the configuration\
Files are in that directory. To give some explanation on this I suppose I'd better inform you that the server's configuration\
is a big part of this model. r, and your configuration files will always be in the root name of the IP you're given to SSH into.\
You will first need to login to the server using the .pem fingerprint I will send to you.
```bash
ssh -i emmett.pem ec2-user@3.135.82.188
```
The server as of now is configured exclusively for crontab. There is also a configured high-performance load balancer, as well as supervisor. Here are our dependent applications:
- nginx, pronounced Engine-X is a high-performance web-server and load balancer, It's known for being quick, and is well built\
with over 10 years of open-source development behind it. Forget Apache! Next, 
- supervisor runs bash commands to run a script or something similar.
- Crontab, accessed with crontab -e is a way of getting workers to do your jobs at specific times, and is used to retrain the\
model on fresh data. This will of course be expanded with more and more models.
In the amazon server, I have left the conf.ds inside of their respective folders, \
```bash
cd /etc/nginx/conf.d
nano 3.135.82.188.conf
```
This will bring you into the load balancer configuration. Updating this to run with say, a domain would be as simple as changing the server name in the top. \
**To edit crontabs, use crontab -e**
