Within the ssh login provided, you can access the API under the directory given, This is a map of the configuration files\
which you can find inside of the system, by either running the command the file is named (crontab -e for example.) In the\
folders, they are located in their respective conf.d directories, nginx for example is /etc/nginx/conf.d, the configuration\
Files are in that directory. To give some explanation on this I suppose I'd better inform you that the server's configuration\
is a big part of this model. r, and your configuration files will always be in the root name of the IP you're given to SSH into.\
The server is completely automated to retrain the model on a new freshly populated database, and\
leaves us with a lot more scalability and speed in the long run. So it might be important to know what everything is:\
- nginx, pronounced Engine-X is a high-performance web-server and load balancer, It's known for being quick, and is well built\
with over 10 years of open-source development behind it. Forget Apache! Next, 
- supervisor runs bash commands to run a script or something similar.\
- Crontab, accessed with crontab -e is a way of getting workers to do your jobs at specific times, and is used to retrain the\
model on fresh data. This will of course be expanded with more and more models.
