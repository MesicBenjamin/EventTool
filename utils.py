import time
import numpy as np
import pandas as pd

from bokeh.models import ColumnDataSource, Slider, Spinner, FreehandDrawTool, Button, Select, Div
from bokeh.plotting import figure
from bokeh.layouts import row
from bokeh.events import SelectionGeometry

np.random.seed(1234)
COLORS = np.random.choice(range(256), size=(1000,3))

# -------------------------------------------------
CONFIG = {
    'anno' :{
        'class'        : ['A', 'B', 'C'],
        'instance_max' : 10,                
        'feature_max'  : 10           
    },

    'event': {
        'x_min' : 0,
        'x_max' : 250, #1280, 
        'y_min' : 0,
        'y_max' : 200, #720,
        't_step': int(1e3),
        't_max' : int(1e5),
        't_min' : 0,
        'dt_step': int(1e1),
        'dt_max' : int(2e4),
        'dt_min' : 1,
    },
    'gui' : {
        'fig_w' : 600,
        'fig_h' : 600,

        'colormap' : ['p', 'class', 'feature', 'instance'],
        'refresh' : 15 # FPS
    }
}    

# -------------------------------------------------
def filter_df(timestamp, dtimestamp, df, f_timestamp_idx=None):
    '''
    Given df, timestamp and dtimestamp
    return events as df that are within timestamp window
    '''
      
    if f_timestamp_idx is not None:

        df_len = len(df.index)
        
        idx_min = f_timestamp_idx(timestamp-2*dtimestamp) - 0.005*df_len
        idx_max = f_timestamp_idx(timestamp+3*dtimestamp) + 0.005*df_len
        idx_min = int(max(idx_min, 0))
        idx_max = int(min(idx_max, df_len))

        df = df[idx_min:idx_max]

    df_filter = (df['timestamp'] >= timestamp) & (df['timestamp']<= timestamp+dtimestamp)
    df = df[df_filter]
    
    return df

def IntfromRGBA(rgba):
    red, green, blue, alpha = rgba
    return (alpha << 24) + (red << 16) + (green << 8) + blue

def RGBAfromInt(argb_int):
    blue =  argb_int & 255
    green = (argb_int >> 8) & 255
    red =   (argb_int >> 16) & 255
    alpha = (argb_int >> 24) & 255
    return (red, green, blue, alpha)

# -------------------------------------------------

class EventTool(object):

    def __init__(self, streams, config=CONFIG):

        self.t  = 0         # current timestamp (global time)
        self.dt = 20000     # current dtimestamp (frame window)

        self.t_min = config['event']['t_min']
        self.t_max = config['event']['t_max']
        self.t_step = config['event']['t_step']

        self.dt_min = config['event']['dt_min']
        self.dt_max = config['event']['dt_max']
        self.dt_step = config['event']['dt_step']

        self.gui_t = time.time()
        self.gui_refresh_rate = config['gui']['refresh']
        self.gui_colormap = config['gui']['colormap']

        self.anno = config['anno']

        self.streams = {ID:Stream(ID, data, self.t, self.dt, config) for ID, data in streams.items()}

        self.init_widgets()

        self.update()

    def update(self):

        for ID, s in self.streams.items():

            s.update(self.t, self.dt, self.colormap_selection.value)

    def init_widgets(self):

        # ------------------
        # Timestamp sliders
        
        # timestamp
        self.timestamp_slider = Slider(
            start=self.t_min, end=self.t_max,
            value=self.t, step=self.t_step,
            width=600,
            title=None
        )
        self.timestamp_spinner = Spinner(
            low=self.t_min, high=self.t_max,
            value=self.t, step=self.t_step,
            width=100
        )
        self.timestamp_slider.on_change('value', self.callback_timestamp)
        self.timestamp_slider.js_link('value', self.timestamp_spinner, 'value')    
        self.timestamp_spinner.js_link('value', self.timestamp_slider, 'value')
        
        # dtimestamp
        self.dtimestamp_slider = Slider(
            start=self.dt_min, end=self.dt_max,
            value=self.dt, step=self.dt_step,
            width=600,
            title=None
        )
        self.dtimestamp_spinner = Spinner(
            low=self.dt_min, high=self.dt_max,
            value=self.dt, step=self.dt_step,
            width=100
        )
        self.dtimestamp_slider.on_change('value', self.callback_dtimestamp)
        self.dtimestamp_slider.js_link('value', self.dtimestamp_spinner, 'value')    
        self.dtimestamp_spinner.js_link('value', self.dtimestamp_slider, 'value')

        # ------------------
        # Colormap widgets        
        self.colormap_selection = Select(
            title="",
            value='none',
            options=['none'] + self.gui_colormap,
            width=120,
        )
        self.colormap_selection.on_change('value', self.callback_colormap)

        # ------------------
        # Annotations widgets           
        widget_width = 120
        
        self.class_selection = Select(
            title="Class:",
            value='none',
            options=['none'] + self.anno['class'],
            width=widget_width,
        )
        
        self.instance_selection = Select(
            title="Instance:",
            value='none',
            options=['none'] + [str(i) for i in range(self.anno['instance_max'])],
            width=widget_width,
        )

        self.feature_selection = Select(
            title="Feature:",
            value='none',
            options=['none'] + [str(i) for i in range(self.anno['feature_max'])],
            width=widget_width,
        )

        self.annotate_button = Button(
            label="Annotate",
            button_type="success",
            width=widget_width
        )
        self.annotate_button.on_click(self.callback_annotate)
        
        self.export_button = Button(
            label="Export",
            button_type="success",
            width=widget_width
        )
        self.export_button.on_click(self.callback_export)
        
        self.clear_button = Button(
            label="Clear",
            button_type="success",
            width=widget_width
        )
        self.clear_button.on_click(self.callback_clear)

    def app_function(self, doc):

        # -------------------------
        # GUI

        # Add timestamp sliders
        doc.add_root(row(Div(text="Timestamp", width=80), self.timestamp_slider, self.timestamp_spinner))
        doc.add_root(row(Div(text="DTimestamp", width=80), self.dtimestamp_slider, self.dtimestamp_spinner))        
        doc.add_root(row(Div(text="Colormap", width=80), self.colormap_selection))

        # Add stream main plot
        fig_keys = list(self.streams)

        # Choose layout wrt number of streams, currently option for 1 or 2 streams
        if len(fig_keys) == 1:
            grid = self.streams[fig_keys[0]].figure
            
        elif len(fig_keys) == 2:
            grid = row([self.streams[fig_keys[0]].figure, self.streams[fig_keys[1]].figure], sizing_mode='scale_width')
            
        doc.add_root(grid)

        # Add annotation widgets
        doc.add_root(row(self.class_selection, self.instance_selection, self.feature_selection))
        doc.add_root(row(self.annotate_button, self.clear_button, self.export_button))

        return doc

    def check_time(self):

        t = time.time()

        if t - self.gui_t < 1./self.gui_refresh_rate:
            return False
            
        self.gui_t = t        
            
        return True

    def callback_timestamp(self, attr, old, new):
        
        if not self.check_time():
            return

        self.t = new
        self.update()

    def callback_dtimestamp(self, attr, old, new):
        
        if not self.check_time():
            return

        self.dt = new
        self.update()

    def callback_export(self):

        for ID, s in self.streams.items():
            s.export_anno()
        
    def callback_annotate(self):

        anno_value = self.convert_class_to_anno()

        if anno_value is None:
            return

        for ID, s in self.streams.items():
            s.annotate(anno_value)
    
        self.update()

    def callback_clear(self):

        for ID, s in self.streams.items():
            s.clear_annotation()

        self.update()

    def callback_colormap(self, attr, old, new):

        self.update()

    def convert_class_to_anno(self):

        class_value = self.class_selection.value
        feature_value = self.feature_selection.value
        instance_value = self.instance_selection.value

        if class_value == 'none':
            return None        

        class_value = self.anno['class'].index(class_value ) + 1        
        anno = class_value*1e6

        if not feature_value  == 'none':
            anno += int(feature_value)

        if not instance_value == 'none':
            anno += int(instance_value)*10e3

        return anno

    @staticmethod
    def convert_anno_to_class(anno_value):

        class_value = int(anno_value/1e6)
        instance_value = int((anno_value%1e6)/1e3)
        feature_value = int(anno_value%1e3)

        return class_value, instance_value, feature_value

class Stream(object):

    def __init__(self, ID, data, t, dt, config):

        self.ID = ID 
        self.df = data # pandas dataframe

        self.x_min = config['event']['x_min']
        self.y_min = config['event']['y_min']
        self.x_max = config['event']['x_max']
        self.y_max = config['event']['y_max']

        self.h, self.w = config['gui']['fig_h'], config['gui']['fig_w']

        # Add annotation column for mask storage
        if 'anno' not in self.df:
            self.df['anno'] = -1

        self.selected_idx = []

        # Used for selection query
        self.df['id'] = self.df.index

        # Create mapping from timestamp to dataframe idx. Used for faster df selection
        df_sample = self.df.sample(n=10000)
        x, y = df_sample['timestamp'], df_sample['id']
        x = x.append(pd.Series(np.zeros((1))).T, ignore_index=True)
        y = y.append(pd.Series(np.zeros((1))).T, ignore_index=True) 
        w = (x==0).astype(int) + 1e-4 # add more weight to point that goes through 0
        self.map_t_idx = np.poly1d(np.polyfit(x, y, 10, w=w))           

        # bokeh source
        img = np.zeros((self.y_max, self.x_max), dtype=np.uint32)
        self.source = ColumnDataSource({'value': [img]})

        # Plot colormap
        self.colormap = None
        
        # Plot
        self.figure = figure(
            width=self.w, height=self.h,
            x_range=[self.x_min, self.x_max],
            y_range=[self.y_min, self.y_max],
            title='Stream: {}'.format(self.ID),
            tools='lasso_select, box_select, pan, wheel_zoom, reset'
        )

        self.figure.image_rgba('value', source=self.source, x=0, y=0, dw=self.x_max, dh=self.y_max)
                        
        self.figure.on_event(SelectionGeometry, self.callback_selection)

        # Drawing pen
        renderer = self.figure.multi_line([], [], line_width=5, alpha=1, color='black')
        self.figure.add_tools(FreehandDrawTool(renderers=[renderer], num_objects=10))
            
    def update(self, t, dt, colormap):
        
        self.colormap = colormap

        self.df_filtered = filter_df(t, dt, self.df, self.map_t_idx)
        
        img = self.draw()

        self.source.data['value'] = [img]

    def callback_selection(self, event):

        self.selected_idx = []

        if event.geometry['type'] == 'rect':

            # Selection coordinates
            x0 = event.geometry['x0']
            x1 = event.geometry['x1']
            y0 = event.geometry['y0']
            y1 = event.geometry['y1']

            mask = (self.df_filtered['x'] > x0) & (self.df_filtered['x'] < x1) & (self.df_filtered['y'] > y0) & (self.df_filtered['y'] < y1)

            # Selected indices
            self.selected_idx = list(self.df_filtered[mask]['id'])

            img = self.draw()

            self.source.data['value'] = [img]

        elif event.geometry['type'] == 'poly':
            pass # TBD
            
        else:
            return

    def export_anno(self):

        self.df['anno'].to_csv(self.ID + '.csv', encoding='utf-8', index=False)

    def annotate(self, anno_value):

        if anno_value == 'none':
            return

        if not len(self.selected_idx):
            return
        
        self.df['anno'][self.df['id'].isin(self.selected_idx)] = anno_value

    def clear_annotation(self):

        if not len(self.selected_idx):
            return
        
        self.df['anno'][self.df['id'].isin(self.selected_idx)] = -1

    def draw(self, alpha=120):

        img = np.zeros((self.y_max, self.x_max), dtype=np.uint32)

        cmap = self.colormap

        # Default none colormap
        if cmap == 'none':

            if len(self.selected_idx):

                img[(self.df_filtered['y'],self.df_filtered['x'])] = IntfromRGBA([0, 0, 0, alpha])

                # highlight selected
                selected_y = self.df_filtered['y'][self.df_filtered['id'].isin(self.selected_idx)]
                selected_x = self.df_filtered['x'][self.df_filtered['id'].isin(self.selected_idx)]
                img[(selected_y,selected_x)] = IntfromRGBA([0, 0, 0, 255])

            else:
                img[(self.df_filtered['y'],self.df_filtered['x'])] = IntfromRGBA([0, 0, 0, 255])

        # Other colormaps
        elif cmap == 'p':

            colors = [[255, 120, 0], [0, 120, 255]]

            if len(self.selected_idx):

                for c,col in enumerate(colors):
                    df_filtered = self.df_filtered[self.df_filtered['p']==c]
                    img[(df_filtered['y'],df_filtered['x'])] = IntfromRGBA([*col, alpha])

                    # highlight selected
                    df_filtered = df_filtered[df_filtered['id'].isin(self.selected_idx)]
                    selected_y = df_filtered['y']
                    selected_x = df_filtered['x']
                    img[(selected_y,selected_x)] = IntfromRGBA([*col, 255])
    
            else:
                
                for c,col in enumerate(colors):

                    df_filtered = self.df_filtered[self.df_filtered['p']==c]
                    img[(df_filtered['y'],df_filtered['x'])] = IntfromRGBA([*col, 255])

        else:

            colors = COLORS
            classes = self.df_filtered['anno'].unique() 

            if len(self.selected_idx):

                for c in classes:

                    df_filtered = self.df_filtered[self.df_filtered['anno']==c]
                    if c == -1:
                        col = [0,0,0]
                    else:
                        clas, feature, instance = EventTool.convert_anno_to_class(c)
                        
                        if cmap == 'class':
                            col = colors[clas]
                        elif cmap == 'feature':
                            col = colors[feature]
                        elif cmap == 'instance':
                            col = colors[instance]

                    img[(df_filtered['y'],df_filtered['x'])] = IntfromRGBA([*col, alpha])                        

                    # highlight selected
                    df_filtered = df_filtered[df_filtered['id'].isin(self.selected_idx)]
                    selected_y = df_filtered['y']
                    selected_x = df_filtered['x']
                    img[(selected_y,selected_x)] = IntfromRGBA([*col, 255])
    
            else:
                
                for c in classes:

                    if c == -1:
                        col = [0,0,0]
                    else:
                        clas, feature, instance = EventTool.convert_anno_to_class(c)
                        col = colors[clas]
                        
                    df_filtered = self.df_filtered[self.df_filtered['anno']==c]
                    img[(df_filtered['y'],df_filtered['x'])] = IntfromRGBA([*col, 255])

        return img