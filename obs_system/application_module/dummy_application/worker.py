import tkinter as tk
from tkinter import ttk

class MainWindow():
    """
    The `MainWindow` class is the main window of the application,
    which displays the configuration interface for picking the model , 
    source and processing operation. 
    Tracking is always enabled with the GUI    
    """
        
    def __init__(self):

        # Create the main application window
        self.root = tk.Tk()
        self.root.title("Model Selection Interface")
        self.root.geometry("450x450")

        self.camera_id = None
        self.stream_key = None 
        self.selected_model = None
        self.use_batching = None
        self.selected_stream = None
        self.ip_address = None


        # Model selection
        self.model_choice = tk.StringVar(value="Yolov8n")
        model_label = tk.Label(self.root, text="Select a Model")
        model_label.pack(pady=10)

        model_options = ["Yolov5s", "Yolov5n", "Yolov8s", "Yolov8n"]
        model_menu = ttk.Combobox(self.root, textvariable=self.model_choice, values=model_options, state="readonly")
        model_menu.pack(pady=10)

        # Batching selection
        self.batching_choice = tk.StringVar(value="No")
        batching_label = tk.Label(self.root, text="Use performance optimizer?")
        batching_label.pack(pady=10)

        batching_menu = ttk.Combobox(self.root, textvariable=self.batching_choice, values=["Yes", "No"], state="readonly")
        batching_menu.pack(pady=10)
        batching_menu.bind("<<ComboboxSelected>>", self.on_batching_select)
        
        # Stream selection (appears only when batching is selected)
        self.stream_label = tk.Label(self.root, text="Select the Input Stream")
        self.stream_choice = tk.StringVar(value="Video")
        self.stream_choice_menu = ttk.Combobox(self.root, textvariable=self.stream_choice, values=["Video", "Local Network Streaming", "Live Streaming"], state="readonly")
        self.stream_choice_menu.bind("<<ComboboxSelected>>", self.on_stream_select)

        # Webcam Frame
        self.webcam_frame = tk.Frame(self.root)
        camera_id_label = tk.Label(self.webcam_frame, text="Enter video path:")
        camera_id_label.pack(pady=5)
        self.camera_id_entry = tk.Entry(self.webcam_frame)
        self.camera_id_entry.pack(pady=5)

        # Local Network Streaming Frame
        self.network_frame = tk.Frame(self.root)
        ip_label = tk.Label(self.network_frame, text="Enter local network stream address:")
        ip_label.pack(pady=5)
        self.ip_entry = tk.Entry(self.network_frame)
        self.ip_entry.pack(pady=5)

        warning_label = tk.Label(self.network_frame, text="Warning: Ensure a platform is already streaming to the local network.", fg="red")
        warning_label.pack(pady=5)

        # Live Streaming Frame
        self.live_stream_frame = tk.Frame(self.root)
        stream_key_label = tk.Label(self.live_stream_frame, text="Enter Streaming Key (HTTP Protocol preferably):")
        stream_key_label.pack(pady=5)
        self.stream_key_entry = tk.Entry(self.live_stream_frame)
        self.stream_key_entry.pack(pady=5)

        # Apply button to execute the chosen settings
        apply_button = tk.Button(self.root, text="Apply", command=self.apply_settings)
        apply_button.pack(pady=20)

        # Start the Tkinter event loop
        self.root.mainloop()

    # Function to be executed when the apply button is pressed
    def apply_settings(self):
        self.selected_model = self.model_choice.get()
        self.use_batching = self.batching_choice.get()
        
        if self.use_batching == "Yes":
            self.selected_stream = self.stream_choice.get()
            if self.selected_stream == "Video":
                self.camera_id = self.camera_id_entry.get()
                print(f"Model: {self.selected_model}, Batching: {self.use_batching}, Stream: {self.selected_stream}, Video Path: {self.camera_id}")
            elif self.selected_stream == "Local Network Streaming":
                self.ip_address = self.ip_entry.get()
                print(f"Model: {self.selected_model}, Batching: {self.use_batching}, Stream: {self.selected_stream}, Address: {self.ip_address}")
            elif self.selected_stream == "Live Streaming":
                self.stream_key = self.stream_key_entry.get()
                print(f"Model: {self.selected_model}, Batching: {self.use_batching}, Stream: {self.selected_stream}, Streaming Key: {self.stream_key}")
        else:
            print(f"Model: {self.selected_model}, Batching: {self.use_batching}. No stream selection is not implemented.")
            quit()
        # Close the application window
        self.root.destroy()

    # Callback for showing/hiding options based on selected stream
    def on_stream_select(self,event):
        stream_type = self.stream_choice.get()
        if stream_type == "Video":
            self.webcam_frame .pack(pady=10)
            self.network_frame.pack_forget()
            self.live_stream_frame.pack_forget()
        elif stream_type == "Local Network Streaming":
            self.webcam_frame .pack_forget()
            self.network_frame.pack(pady=10)
            self.live_stream_frame.pack_forget()
        elif stream_type == "Live Streaming":
            self.webcam_frame .pack_forget()
            self.network_frame.pack_forget()
            self.live_stream_frame.pack(pady=10)


    # Callback for showing/hiding batching-dependent options
    def on_batching_select(self,event):
        if self.batching_choice.get() == "Yes":
            self.stream_label.pack(pady=10)
            self.stream_choice_menu.pack(pady=10)
        else:
            self.stream_label.pack_forget()
            self.stream_choice_menu.pack_forget()
            self.webcam_frame.pack_forget()
            self.network_frame.pack_forget()
            self.live_stream_frame.pack_forget()

    def get_configuration(self):
        if self.selected_model is None:
            raise ValueError("No model selected.")
        config = {
            "model_name": self.selected_model.lower(),
            "stream": self.use_batching,
        }
        config['source'] = None
        if self.camera_id: 
            config['source'] = self.camera_id
        elif self.ip_address:
            config['source'] = self.ip_address
        elif self.stream_key:
            config['source'] = self.stream_key
        
        return config









