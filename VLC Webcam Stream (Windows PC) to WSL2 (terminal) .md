## VLC Webcam Stream to WSL2 

> **How to connect the VLC stream from the Windows PC to a terminal that executes in the WSL2 virtual environment.**

### **_NOTE_**
*** 
This process took _a lot_ of different approaches and many tools that are not needed. For instance, even though we can connect the **WSL2** terminal to the **Windows** terminal through a **Node JS** server, also leveraging websockets, it is not an ideal option at least for my system. 

> The issue is that the stream either with UDP or RTSP (especially when using the **FFMPEG** service) will successfully initiate a server that the client can attach to, but the client did not receive any frames, due to connectivity or compatibility problems. **Be very careful of what protocol you choose to stream with as the WSL2 connection to the Windows machine will mostly determine what you need!**
> ***

# Steps 

## VLC Stream Windows
1. If you haven't already download the [ **_VLC Media Player_**](https://www.videolan.org/vlc/). 
2. Once Installed, open the GUI to set the following settings:   
	- Press Ctrl-S to open Stream or Media &rarr; Open Network Stream  
	- Leave the URL address blank as the rest of steps will create it by default. Click Capture Device tab
	- On Capture Device tab, leave Capture Mode on default (_DirectShow_). 
	- Select the Video device name and Audio device name to use the HW (_This depends on the vcodec and mux you want to use later._)
	-   Use the Drop Down list beside the **Play** button and select **Stream** 
3.  This opens the Stream Output Tab. 
Source is dshow:\\. Click Next. 
4. The Destination Setup opens. Use the drop down list to select what the destination will be (File, HTTP, RTP, RTSP etc.). For us the HTTP protocol is what connects the stream to the WSL2 client. Select HTTP and click Add. 
5. Specify the Port Number (_by default it is set to 8080_) and the path (_the name of the stream for the client to find_ )
	For our case set Port: 8080 and path /live Click Next
6. Check that Active Transcoding is checked. This selects the stream encapsulation along with the video and audio codec. 
> _Tradeoffs on performance and quality of the stream have to be made. If compression and quality is the aim then **H264** codec is ideal. For good frame-by-frame inference, **MJPEG** should be the choice. If we are focusing on low-latency streams then **MPEG-TS** is the best._ 

7. Select the appropriate for you codec and encapsulation. We use the MPEG-TS encapsulation with H264 video codec (no audio). 
> This is a custom **Profile**, to create one select the symbol next to X (hovering reveals "create new profile"). Click Save and Next.  Otherwise you can edit already created profiles by pressing the tool symbol. 

8. Once you are done click Next and the Option Setup opens up. Check the Generated stream output string to confirm the stream characteristics and destination. **_Remember :8080/path represents the local host machine or VLC will transmit this stream to every device on the local network at the specific port._** 
9. Click Stream and VLC will try to connect to the URL and then start capturing and transmiting if successful. 


## WSL2 FFMPEG stream reception 

> This approach only is tested for WSL2. If you haven't already you can install WSL2 from [here](https://learn.microsoft.com/en-us/windows/wsl/install) 

Make sure that FFMPEG is installed and can be used by terminal. You can follow this tutorial to install [ffmpeg](https://phoenixnap.com/kb/install-ffmpeg-ubuntu)!

Ensure that FFMPEG is functional in WSL terminal `$ ffmpeg`. If it is successfully working you should see something like this:
![ffmpeg version](../_resources/Screenshot%202024-09-12%20140914.png)


To be able to connect the WSL2 and the Windows machine we need to change the firewall admissions for the local network. Specifically: 
* On Windows, search for _Windows Defender Firewall with Advanced Security_ and Click on it. 
  * Select Inbound Rules 
  * Scroll until you find **_File and Printer Sharing (Echo Request -...)_**. The `...` can either be IPv4 or IPV6. There should  be 4 options in total. They are also categorized  for Domain or Private based on the Profile we want. 
  * Enable what you need. Enabling all is the easiest.  
  * Once you have enable them close the window. 

> You can also create a New Rule to only allow a single machine or address to access the network. Consider this option if the above does not work. 


We can now acquire the VLC stream with ffmpeg with: 
`ffmpeg -i http://<your-ip>:8080/live -c copy output.mp4`

This will take the stream and save it in an output file. Conversion from vcodec is done automatically from the ffmpeg command. To verify that the capture of the stream is working, you should see something like: 

![Screenshot 2024-09-12 141432.png](../_resources/Screenshot%202024-09-12%20141432.png)
***
### Congratulations!!!
#### You have successfully streamed from VLC and Windows to WSL2. 

> Remember: Check the video output to see that it actually worked!! 


*** 
#### Some Pottential issues 

A.  Local host may not be correctly set in WSL. You should check the IP of the windows machine and manually use that. This means that in the `ffmpeg` command you have to specify the IP, eg: 1xx.1xx.xxx.xxx:8080. Using `localhost:8080`, will not work. 

B. Streaming with RTSP option does not work with FFMPEG even though it is supported. Most of the stack overflow community have found other workarounds, through FFserver is also not very functional. Check [stackoverflow-ffmpeg](https://stackoverflow.com/questions/26999595/what-steps-are-needed-to-stream-rtsp-from-ffmpeg) and [otherlink](https://www.videoexpertsgroup.com/glossary/how-to-stream-rtsp-using-ffmpeg). These did not work for us even when we created a node js server with stream capabilities using `node-rtsp-stream` module. The ffmpeg would not transmit the stream when the `-f rtsp rtsp://<your-ip>:port/path` was used. 

C. VLC Media player will sometimes not connect to the localhost, as the Port is already working for another process (_this can be a previous vlc process that stayed_).To identify the process run a windows terminal as administrator and use `netstat -anob | findstr :<port>`
and kill the pid that will appear with taskkill. More information in [close ports](https://stackoverflow.com/questions/8688949/how-to-close-tcp-and-udp-ports-via-windows-command-line)! 
