import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.font_manager import FontProperties

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class AudioSpectrumMarkerApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Spectrum Marker")
        master.geometry("1500x800")  

        # Application state
        self.audio_path = None
        self.y = None
        self.sr = None
        self.spectrogram = None
        self.markers = []
        self.markers_history = []  
        self.max_history = 50  
        self.is_labeling_mode = False  

        # Define marker colors
        self.marker_color = '#8B0000'  # Dark red
        self.highlight_color = '#FF0000'  # Highlight red
        self.highlight_spans = []  # Store highlight span objects

        # Zoom and pan related variables
        self.zoom_scale = 1.0
        self.initial_xlim = None
        self.initial_ylim = None
        self.zoom_base = 1.2  # Zoom base
        self.zoom_speed = 0.1  # Zoom speed
        self.pan_speed = 1.0  # Pan speed
        self.last_pan_update = 0  # Last pan update time
        self.pan_update_interval = 1/60  # Pan update interval (60fps)

        # Create main frame
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create toolbar
        self.create_toolbar()

        # Create left-right split panel
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Create left panel (spectrogram)
        self.left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_frame, weight=2)  

        # Create right panel (marker data)
        self.right_frame = ttk.Frame(self.paned_window)
        self.right_frame.pack_propagate(False)  
        self.paned_window.add(self.right_frame, weight=1)  

        # Set initial split position
        self.master.update()  
        total_width = self.paned_window.winfo_width()
        self.paned_window.sashpos(0, int(total_width * 2/3))  

        # Bind window resize event
        self.master.bind('<Configure>', self.on_window_resize)

        # Create spectrogram area
        self.create_spectrogram_area()

        # Create marker data area
        self.create_marker_data_area()

    def create_toolbar(self):
        toolbar_frame = ttk.Frame(self.main_frame)
        toolbar_frame.pack(fill=tk.X)

        # Import audio button
        import_btn = ttk.Button(toolbar_frame, text="Open Audio", command=self.load_audio)
        import_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Mode switch button
        self.mode_btn = ttk.Button(toolbar_frame, text="Label Mode: Off", command=self.toggle_mode)
        self.mode_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Undo button
        self.undo_btn = ttk.Button(toolbar_frame, text="Undo", command=self.undo_last_marker, state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Export markers button
        export_btn = ttk.Button(toolbar_frame, text="Export", command=self.export_markers)
        export_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Add zoom buttons
        zoom_in_btn = ttk.Button(toolbar_frame, text="Zoom In", command=self.zoom_in)
        zoom_in_btn.pack(side=tk.LEFT, padx=5, pady=5)

        zoom_out_btn = ttk.Button(toolbar_frame, text="Zoom Out", command=self.zoom_out)
        zoom_out_btn.pack(side=tk.LEFT, padx=5, pady=5)

        reset_zoom_btn = ttk.Button(toolbar_frame, text="Reset Zoom", command=self.reset_zoom)
        reset_zoom_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Add clip button
        clip_btn = ttk.Button(toolbar_frame, text="Clip Audio", command=self.clip_audio)
        clip_btn.pack(side=tk.LEFT, padx=5, pady=5)

    def create_spectrogram_area(self):
        # Spectrogram container
        self.spectrogram_frame = ttk.Frame(self.left_frame)
        self.spectrogram_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas container
        canvas_frame = ttk.Frame(self.spectrogram_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        canvas_widget = self.canvas.get_tk_widget()
        
        # Create scrollbars
        self.h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        self.v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        
        # Layout canvas and scrollbars
        canvas_widget.grid(row=0, column=0, sticky='nsew')
        self.h_scrollbar.grid(row=1, column=0, sticky='ew')
        self.v_scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Configure grid weights
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Configure scrollbars
        self.h_scrollbar.config(command=self.on_x_scroll)
        self.v_scrollbar.config(command=self.on_y_scroll)
        canvas_widget.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)

        # Set scroll range
        self.update_scrollbar_range()

        # Add interaction events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Initialize interaction variables
        self.start_time = None
        self.marker_rect = None
        self.is_panning = False
        self.pan_start_x = None
        self.pan_start_y = None
        self.last_xlim = None
        self.last_ylim = None

    def update_scrollbar_range(self):
        if self.spectrogram is not None:
            # Get current view range
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            
            # Get data range
            total_duration = len(self.y) / self.sr
            max_freq = self.sr / 2
            
            # Calculate scrollbar position
            x_size = (xmax - xmin) / total_duration
            y_size = (ymax - ymin) / max_freq
            
            # Update scrollbars
            self.h_scrollbar.set(xmin/total_duration, xmax/total_duration)
            self.v_scrollbar.set(ymin/max_freq, ymax/max_freq)

    def on_x_scroll(self, *args):
        if self.spectrogram is not None:
            # Get current view range
            xmin, xmax = self.ax.get_xlim()
            view_width = xmax - xmin
            
            # Calculate movement amount
            if args[0] == 'scroll':
                scroll_amount = int(args[1])  # 1 or -1
                move = view_width * 0.1 * scroll_amount
            elif args[0] == 'moveto':
                total_duration = len(self.y) / self.sr
                target_pos = float(args[1]) * total_duration
                move = target_pos - xmin
            
            # Apply movement
            self.ax.set_xlim(xmin + move, xmax + move)
            self.canvas.draw()
            self.update_scrollbar_range()

    def on_y_scroll(self, *args):
        if self.spectrogram is not None:
            # Get current view range
            ymin, ymax = self.ax.get_ylim()
            view_height = ymax - ymin
            
            # Calculate movement amount
            if args[0] == 'scroll':
                scroll_amount = int(args[1])  # 1 or -1
                move = view_height * 0.1 * scroll_amount
            elif args[0] == 'moveto':
                max_freq = self.sr / 2
                target_pos = float(args[1]) * max_freq
                move = target_pos - ymin
            
            # Apply movement
            self.ax.set_ylim(ymin + move, ymax + move)
            self.canvas.draw()
            self.update_scrollbar_range()

    def create_marker_data_area(self):
        # Title
        title_frame = ttk.Frame(self.right_frame)
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        title_label = ttk.Label(title_frame, text="Marker Data", font=('Arial', 12, 'bold'))
        title_label.pack(side=tk.LEFT)

        # Create treeview frame
        tree_frame = ttk.Frame(self.right_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create marker list and scrollbars
        self.marker_tree = ttk.Treeview(tree_frame, 
                                      columns=('ID', 'Name', 'Start', 'End', 'Duration', 'Action'), 
                                      show='headings', 
                                      height=20,
                                      selectmode='extended')
        
        # Bind selection event
        self.marker_tree.bind('<<TreeviewSelect>>', self.on_marker_select)
        
        # Bind drag events
        self.marker_tree.bind('<Button-1>', self.on_drag_start)
        self.marker_tree.bind('<B1-Motion>', self.on_drag_motion)
        self.marker_tree.bind('<ButtonRelease-1>', self.on_drag_release)
        
        # Initialize drag variables
        self.drag_data = {'item': None, 'index': None}
        
        # Set column headers
        columns = {
            'ID': ('No.', 50),
            'Name': ('Name', 150),
            'Start': ('Start Time(s)', 120),
            'End': ('End Time(s)', 120),
            'Duration': ('Duration(s)', 120),
            'Action': ('Action', 60)
        }
        
        for col, (text, width) in columns.items():
            self.marker_tree.heading(col, text=text)
            self.marker_tree.column(col, width=width, minwidth=width)

        # Bind double-click event for name editing
        self.marker_tree.bind('<Double-1>', self.on_double_click)
        
        # Add scrollbars
        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.marker_tree.yview)
        x_scroll = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.marker_tree.xview)
        
        # Configure treeview scrolling
        self.marker_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        # Use grid layout
        self.marker_tree.grid(row=0, column=0, sticky='nsew')
        y_scroll.grid(row=0, column=1, sticky='ns')
        x_scroll.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Button frame
        btn_frame = ttk.Frame(self.right_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        # 按钮
        delete_btn = ttk.Button(btn_frame, text="Delete", command=self.delete_marker)
        delete_btn.pack(side=tk.LEFT, padx=5)

        export_btn = ttk.Button(btn_frame, text="Export", command=self.export_markers)
        export_btn.pack(side=tk.LEFT, padx=5)

    def load_audio(self):
        self.audio_path = filedialog.askopenfilename(
            title="Select Audio File", 
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg")]
        )
        if self.audio_path:
            try:
                self.y, self.sr = librosa.load(self.audio_path)
                self.generate_spectrogram()
                
                # 检查是否存在对应的JSON标记文件
                json_path = os.path.splitext(self.audio_path)[0] + '.json'
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            loaded_markers = json.load(f)
                            self.markers = []
                            for marker in loaded_markers:
                                # Ensure marker contains all required fields
                                if all(key in marker for key in ['start_time', 'end_time', 'name']):
                                    self.markers.append({
                                        'start_time': marker['start_time'],
                                        'end_time': marker['end_time'],
                                        'name': marker['name']
                                    })
                    except Exception as e:
                        messagebox.showwarning("Warning", f"Failed to load marker file: {str(e)}")
                else:
                    # If no JSON file found, clear markers
                    self.markers = []
                
                # Reset other states
                self.markers_history = []
                self.undo_btn.config(state=tk.DISABLED)
                self.is_labeling_mode = False
                self.mode_btn.config(text="Label Mode: Off")
                self.canvas.get_tk_widget().config(cursor="")
                self.update_marker_list()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio: {str(e)}")

    def generate_spectrogram(self):
        self.ax.clear()
        D = librosa.stft(self.y)
        self.spectrogram = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        img = librosa.display.specshow(
            self.spectrogram, 
            sr=self.sr, 
            x_axis='time', 
            y_axis='hz',
            ax=self.ax
        )
        
        # Remove color bar
        if hasattr(self, 'colorbar'):
            self.colorbar.remove()
            
        plt.title(f'Spectrogram: {os.path.basename(self.audio_path)}')
        
        # Save initial view range
        self.initial_xlim = self.ax.get_xlim()
        self.initial_ylim = self.ax.get_ylim()
        
        self.canvas.draw()
        
        # Update scrollbars range
        self.update_scrollbar_range()

    def toggle_mode(self):
        self.is_labeling_mode = not self.is_labeling_mode
        self.mode_btn.config(text=f"Label Mode: {'On' if self.is_labeling_mode else 'Off'}")
        
        # Reset all states
        self.start_time = None
        self.is_panning = False
        if self.marker_rect:
            self.marker_rect.remove()
            self.marker_rect = None
        self.canvas.draw()
        
        # Update mouse cursor
        if self.is_labeling_mode:
            self.canvas.get_tk_widget().config(cursor="cross")  
        else:
            self.canvas.get_tk_widget().config(cursor="")  

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if self.is_labeling_mode:
            # Label mode: left click to start marking
            if event.button == 1:
                self.start_time = event.xdata
                # Select color for preview
                if hasattr(self, 'marker_rect') and self.marker_rect:
                    self.marker_rect.remove()
                self.marker_rect = self.ax.axvspan(
                    event.xdata, 
                    event.xdata, 
                    color=self.marker_color, 
                    alpha=0.3
                )
                self.canvas.draw()
        else:
            # Drag mode: both left and right buttons can drag
            if event.button in [1, 3]:
                self.is_panning = True
                self.pan_start_x = event.xdata
                self.pan_start_y = event.ydata
                self.last_xlim = self.ax.get_xlim()
                self.last_ylim = self.ax.get_ylim()

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return

        if self.is_labeling_mode:
            # Label mode: update marker preview
            if self.start_time is not None and event.xdata is not None:
                if hasattr(self, 'marker_rect') and self.marker_rect:
                    self.marker_rect.remove()
                self.marker_rect = self.ax.axvspan(
                    self.start_time, 
                    event.xdata, 
                    color=self.marker_color, 
                    alpha=0.3
                )
                self.canvas.draw()
        else:
            # Drag mode: update view position
            if self.is_panning and event.xdata is not None and event.ydata is not None:
                current_time = event.guiEvent.time
                # Limit update frequency
                if current_time - self.last_pan_update >= self.pan_update_interval:
                    dx = (event.xdata - self.pan_start_x) * self.pan_speed
                    dy = (event.ydata - self.pan_start_y) * self.pan_speed
                    
                    # Use animation effect to smoothly move
                    new_xlim = (self.last_xlim[0] - dx, self.last_xlim[1] - dx)
                    new_ylim = (self.last_ylim[0] - dy, self.last_ylim[1] - dy)
                    
                    self.ax.set_xlim(new_xlim)
                    self.ax.set_ylim(new_ylim)
                    self.canvas.draw()
                    
                    self.last_pan_update = current_time

    def on_release(self, event):
        if event.inaxes != self.ax:
            return

        if self.is_labeling_mode:
            # Label mode: complete marking
            if event.button == 1 and self.start_time is not None:
                end_time = event.xdata
                if hasattr(self, 'marker_rect') and self.marker_rect:
                    self.marker_rect.remove()
                    self.marker_rect = None
                self.add_marker(self.start_time, end_time)
                self.start_time = None
                self.preview_color = None
        else:
            # Drag mode: end dragging
            if event.button in [1, 3]:
                self.is_panning = False

    def on_scroll(self, event):
        # Disable mouse wheel zooming
        pass

    def add_marker(self, start_time, end_time):
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        
        # Save current state to history
        self.markers_history.append(list(self.markers))
        if len(self.markers_history) > self.max_history:
            self.markers_history.pop(0)
        
        # Add new marker
        self.markers.append({
            'start_time': start_time, 
            'end_time': end_time,
            'name': f'Segment {len(self.markers) + 1}',
            'color': self.marker_color  # Use single color
        })
        
        # Save current view range
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        # Update marker list and image
        self.update_marker_list()
        self.redraw_spectrogram()
        
        # Restore view range
        self.ax.set_xlim(current_xlim)
        self.ax.set_ylim(current_ylim)
        self.canvas.draw()
        
        self.undo_btn.config(state=tk.NORMAL)

    def undo_last_marker(self):
        if self.markers_history:
            self.markers = list(self.markers_history.pop())  
            self.update_marker_list()
            self.redraw_spectrogram()
            
            # If no more history records, disable undo button
            if not self.markers_history:
                self.undo_btn.config(state=tk.DISABLED)

    def delete_marker(self):
        selected_items = self.marker_tree.selection()
        if selected_items:
            # Save current state to history
            self.markers_history.append(list(self.markers))
            if len(self.markers_history) > self.max_history:
                self.markers_history.pop(0)
            
            # Delete selected markers
            for item in selected_items:
                index = self.marker_tree.index(item)
                if 0 <= index < len(self.markers):
                    self.markers.pop(index)
            
            self.update_marker_list()
            self.redraw_spectrogram()
            self.undo_btn.config(state=tk.NORMAL)

    def delete_marker_by_index(self, event, index):
        # Make sure the clicked column is the delete button column
        region = self.marker_tree.identify('region', event.x, event.y)
        column = self.marker_tree.identify_column(event.x)
        
        if column == '#6':  # Action column
            # Save current state to history
            self.markers_history.append(list(self.markers))
            if len(self.markers_history) > self.max_history:
                self.markers_history.pop(0)
            
            # Delete marker
            if 0 <= index < len(self.markers):
                self.markers.pop(index)
                self.update_marker_list()
                self.redraw_spectrogram()
                self.undo_btn.config(state=tk.NORMAL)

    def on_marker_select(self, event):
        # Clear previous highlights
        for span in self.highlight_spans:
            span.remove()
        self.highlight_spans.clear()
        
        # Get all selected items
        selected_items = self.marker_tree.selection()
        
        if selected_items:
            # Create highlight for each selected item
            for item in selected_items:
                idx = self.marker_tree.index(item)
                if 0 <= idx < len(self.markers):
                    marker = self.markers[idx]
                    # Create new highlight span
                    span = self.ax.axvspan(
                        marker['start_time'],
                        marker['end_time'],
                        color=self.highlight_color,
                        alpha=0.5,
                        zorder=3  # Ensure highlight appears above other markers
                    )
                    self.highlight_spans.append(span)
            
            self.canvas.draw()

    def on_drag_start(self, event):
        # Get clicked item
        item = self.marker_tree.identify_row(event.y)
        if item:
            # Save dragged item and index
            self.drag_data['item'] = item
            self.drag_data['index'] = self.marker_tree.index(item)

    def on_drag_motion(self, event):
        if self.drag_data['item']:
            # Get item at current mouse position
            target_item = self.marker_tree.identify_row(event.y)
            if target_item and target_item != self.drag_data['item']:
                # Get target position
                target_index = self.marker_tree.index(target_item)
                
                # Move item
                self.marker_tree.move(self.drag_data['item'], '', target_index)
                
                # Update drag data
                self.drag_data['index'] = target_index

    def on_drag_release(self, event):
        if self.drag_data['item']:
            # Get final position
            final_index = self.marker_tree.index(self.drag_data['item'])
            if final_index != self.drag_data['index']:
                # Save current state to history
                self.markers_history.append(list(self.markers))
                if len(self.markers_history) > self.max_history:
                    self.markers_history.pop(0)
                
                # Move marker data
                marker = self.markers.pop(self.drag_data['index'])
                self.markers.insert(final_index, marker)
                
                # Update display
                self.update_marker_list()
                self.redraw_spectrogram()
                self.undo_btn.config(state=tk.NORMAL)
            
            # Clear drag data
            self.drag_data = {'item': None, 'index': None}

    def update_marker_list(self):
        # Clear existing items
        self.marker_tree.delete(*self.marker_tree.get_children())
        
        # Remove all old delete buttons
        for widget in self.marker_tree.winfo_children():
            widget.destroy()
        
        # Add all markers
        for i, marker in enumerate(self.markers, 1):
            start_time = marker['start_time']
            end_time = marker['end_time']
            duration = end_time - start_time
            
            # Update marker name with sequence number
            if marker['name'].startswith('Segment '):
                marker['name'] = f'Segment {i}'
            
            # Insert item and set label color
            item = self.marker_tree.insert('', 'end', values=(
                f"{i}",
                marker['name'],
                f"{start_time:.3f}",
                f"{end_time:.3f}",
                f"{duration:.3f}",
                "×"  # Delete button text
            ))
            
            # Set row label color
            self.marker_tree.tag_configure(f"row_{i}", background=self.marker_color)
            self.marker_tree.item(item, tags=(f"row_{i}",))

            # Add click event for delete button column
            self.marker_tree.tag_bind(f"row_{i}", '<Button-1>', 
                lambda e, idx=i-1: self.delete_marker_by_index(e, idx))

    def redraw_spectrogram(self):
        if self.spectrogram is not None:
            # Save current view range
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            
            # Clear previous highlights
            for span in self.highlight_spans:
                span.remove()
            self.highlight_spans.clear()
            
            self.ax.clear()
            librosa.display.specshow(
                self.spectrogram,
                sr=self.sr,
                x_axis='time',
                y_axis='hz',
                ax=self.ax
            )
            
            # Redraw all markers
            for marker in self.markers:
                self.ax.axvspan(
                    marker['start_time'], 
                    marker['end_time'], 
                    color=marker['color'], 
                    alpha=0.3,
                    zorder=2  # Normal marker layer
                )
            
            # Restore previous view range
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
            
            # Re-add highlights for selected items
            selected_items = self.marker_tree.selection()
            if selected_items:
                for item in selected_items:
                    idx = self.marker_tree.index(item)
                    if 0 <= idx < len(self.markers):
                        marker = self.markers[idx]
                        span = self.ax.axvspan(
                            marker['start_time'],
                            marker['end_time'],
                            color=self.highlight_color,
                            alpha=0.5,
                            zorder=3
                        )
                        self.highlight_spans.append(span)
            
            self.canvas.draw()

    def export_markers(self):
        if not self.markers:
            messagebox.showwarning("Warning", "No markers to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save Marker Data",
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.markers, f, indent=2, ensure_ascii=False)  
            messagebox.showinfo("Success", f"Markers exported to {filename}")

    def clip_audio(self):
        if not self.audio_path or not self.markers:
            messagebox.showwarning("Warning", "Please load an audio file and add markers first!")
            return

        # Get audio file name (without extension) and directory
        audio_dir = os.path.dirname(self.audio_path)
        audio_name = os.path.splitext(os.path.basename(self.audio_path))[0]
        
        # Create output directory
        output_dir = os.path.join(audio_dir, "clips")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare marker data format
        labels = []
        for marker in self.markers:
            label_data = {
                "start_time": marker["start_time"],
                "end_time": marker["end_time"],
                "name": marker["name"]
            }
            labels.append(label_data)

        # Save marker data to JSON file
        json_path = os.path.join(audio_dir, f"{audio_name}.json")
        with open(json_path, 'w') as f:
            json.dump(labels, f, indent=2)

        try:
            # Call clip function
            from clip import process_audio_file
            process_audio_file(self.audio_path, json_path, output_dir)
            messagebox.showinfo("Success", f"Audio clips have been saved to {output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clip audio: {str(e)}")

    def zoom_in(self):
        if self.spectrogram is not None:
            self.zoom_scale *= 1.2
            self.update_zoom()

    def zoom_out(self):
        if self.spectrogram is not None:
            self.zoom_scale /= 1.2
            self.update_zoom()

    def reset_zoom(self):
        if self.spectrogram is not None:
            self.zoom_scale = 1.0
            if self.initial_xlim and self.initial_ylim:
                self.ax.set_xlim(self.initial_xlim)
                self.ax.set_ylim(self.initial_ylim)
                self.redraw_spectrogram()

    def update_zoom(self):
        if self.spectrogram is not None and self.initial_xlim and self.initial_ylim:
            # Get center point of initial range
            center_x = sum(self.initial_xlim) / 2
            center_y = sum(self.initial_ylim) / 2
            
            # Calculate new display range
            initial_width = self.initial_xlim[1] - self.initial_xlim[0]
            initial_height = self.initial_ylim[1] - self.initial_ylim[0]
            
            new_width = initial_width / self.zoom_scale
            new_height = initial_height / self.zoom_scale
            
            # Set new display range, keeping center point unchanged
            self.ax.set_xlim(center_x - new_width/2, center_x + new_width/2)
            self.ax.set_ylim(center_y - new_height/2, center_y + new_height/2)
            
            # Redraw spectrogram and markers
            self.redraw_spectrogram()

    def on_double_click(self, event):
        # Get clicked item and column
        item = self.marker_tree.identify('item', event.x, event.y)
        column = self.marker_tree.identify_column(event.x)
        
        # Only allow editing when clicking the "Name" column
        if column == '#2':  
            self.edit_marker_name(item)

    def edit_marker_name(self, item):
        # Create edit window
        edit_window = tk.Toplevel(self.master)
        edit_window.title("Edit Name")
        edit_window.geometry("300x100")
        edit_window.transient(self.master)  
        
        # Get index and name of currently selected marker
        index = self.marker_tree.index(item)
        current_name = self.markers[index]['name']
        
        # Create input field and label
        ttk.Label(edit_window, text="Name:").pack(padx=5, pady=5)
        entry = ttk.Entry(edit_window, width=40)
        entry.insert(0, current_name)
        entry.pack(padx=5, pady=5)
        entry.select_range(0, tk.END)  
        entry.focus()  
        
        def save_name():
            new_name = entry.get().strip()
            if new_name:  
                # Save current state to history
                self.markers_history.append(list(self.markers))
                if len(self.markers_history) > self.max_history:
                    self.markers_history.pop(0)
                
                # Update name
                self.markers[index]['name'] = new_name
                self.update_marker_list()
                self.undo_btn.config(state=tk.NORMAL)
            edit_window.destroy()
        
        def cancel():
            edit_window.destroy()
        
        # Create button frame
        btn_frame = ttk.Frame(edit_window)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        # Add OK and Cancel buttons
        ttk.Button(btn_frame, text="OK", command=save_name).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key to OK
        edit_window.bind('<Return>', lambda e: save_name())
        # Bind Escape key to Cancel
        edit_window.bind('<Escape>', lambda e: cancel())

    def on_window_resize(self, event):
        # Only handle resize events from the main window
        if event.widget == self.master:
            # Recalculate and set the separator position
            total_width = self.paned_window.winfo_width()
            self.paned_window.sashpos(0, int(total_width * 2/3))

def main():
    root = tk.Tk()
    app = AudioSpectrumMarkerApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()