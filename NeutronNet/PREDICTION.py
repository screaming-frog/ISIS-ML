import CNN

import numpy as np
import wx
import os
import matplotlib

matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
from keras import backend as K


class SnaptoCursor(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
        self.marker, = ax.plot([0],[0], marker="o", color="crimson", zorder=3)
        self.x = x
        self.y = y
        self.txt = ax.text(0.7, 0.9, '')

    def mouse_move(self, event):
        if not event.inaxes: return
        x, y = event.xdata, event.ydata
        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        self.ly.set_xdata(x)
        self.marker.set_data([x],[y])
        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.txt.set_position((x,y))
        self.ax.figure.canvas.draw_idle()


class CanvasPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)

    def draw(self, result, bins):
        self.axes.plot(bins, result)
        self.canvas = FigureCanvas(self, -1, self.figure)

    def clear_graph(self):
        print('ree')
        self.axes.clear()
        self.canvas = FigureCanvas(self, -1, self.figure)
        # self.axes.plot()



class MyFrame(wx.Frame):

    def __init__(self, parent):

        ####################### NEURAL NET STUFF ####################################################
        self.neural_net = CNN.nn()
        self.neural_net.load_model('n_class_model', 'MODELS\\N_CLASSIFICATION_NET.h5')
        self.neural_net.load_model('multi_sld_model', 'MODELS\\2_LAYER_SLD_NET.h5')
        #self.neural_net.load_model('single_sld_model', 'MODELS\\1_LAYER_SLD_NET.h5')
        self.neural_net.load_model('multi_d_model', 'MODELS\\2_LAYER_D_NET.h5')
        #self.neural_net.load_model('single_d_model' 'MODELS\\1_LAYER_D_NET.h5')
        self.data = None
        self.x = None
        self.layer_output = None
        self.layer = None
        self.d_output = None
        self.sld_output = None
        self.x_placeholder = K.placeholder(shape=[1, 150, 2 , 1])
        #############################################################################################

        ############################ GUI STUFF #####################################################
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title='Neutron Net', pos=wx.DefaultPosition, style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        self.currentDirectory = os.getcwd()
        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)

        self.m_statusBar = self.CreateStatusBar(1, wx.STB_SIZEGRIP, wx.ID_ANY)
        self.m_statusBar.SetStatusText('Ready')

        self.m_menubar = wx.MenuBar(0)
        self.m_menu = wx.Menu()
        self.m_openfile = self.m_menu.Append(wx.NewId(), "Load file")
        self.m_menubar.Append(self.m_menu, u"File")
        self.SetMenuBar(self.m_menubar)

        toolbar = wx.ToolBar(self, -1)
        self.ToolBar = toolbar
        toolbar.SetToolBitmapSize((50,50))
        self.runbtn = toolbar.AddTool(-1,'',wx.Bitmap("lightningbolt.png"))
        self.clearbtn = toolbar.AddTool(-1,'',wx.Bitmap("redcross.png"))
        self.Bind(wx.EVT_TOOL, self.clear, self.clearbtn)
        export = toolbar.AddTool(-1,'',wx.Bitmap("genx.png"))
        #self.Bind(wx.EVT_TOOL, self.export_to_genx, export)
        toolbar.Realize()

        self.panel = wx.Panel(self)

        vbox = wx.BoxSizer(wx.VERTICAL)
        self.predict_button = wx.Button(self.panel, label='Predict')
        vbox.Add(self.predict_button, 0, wx.ALIGN_CENTER_HORIZONTAL)
        vbox.AddStretchSpacer(2)

        vbox_inner = wx.BoxSizer(wx.HORIZONTAL)
        vbox.Add(vbox_inner)
        vbox_inner.AddStretchSpacer(10)

        self.list_ctrl = wx.ListCtrl(self.panel, size=(-1, 100),style=wx.LC_REPORT|wx.BORDER_SUNKEN)
        self.list_ctrl.InsertColumn(0, 'x')
        self.list_ctrl.InsertColumn(1, 'y')
        vbox_inner.Add(self.list_ctrl,0, wx.ALL|wx.EXPAND)
        vbox_inner.AddStretchSpacer(10)

        self.sld_graph = CanvasPanel(self.panel)
        self.sld_graph.figure.suptitle('Scattering length density')
        vbox_inner.Add(self.sld_graph, 1, wx.EXPAND | wx.ALIGN_RIGHT)
        vbox_inner.AddStretchSpacer(10)

        self.d_graph = CanvasPanel(self.panel)
        self.d_graph.figure.suptitle('Thickness')
        vbox_inner.Add(self.d_graph, 1, wx.EXPAND | wx.ALIGN_RIGHT)
        vbox_inner.AddStretchSpacer(10)

        vbox.AddStretchSpacer(10)
        self.output_box = wx.TextCtrl(self.panel, style = wx.TE_READONLY | wx.TE_MULTILINE)
        vbox.Add(self.output_box, 10, wx.EXPAND|wx.ALL, 5)

        b2 = wx.Button(self.panel, label="Clear")
        b2.Bind(wx.EVT_BUTTON, self.clear)
        vbox.AddStretchSpacer(2)
        vbox.Add(b2, 0, wx.ALIGN_CENTER_HORIZONTAL)
        vbox.AddStretchSpacer(2)

        self.panel.SetSizer(vbox)
        self.Fit()

        #############################################################################################
        self.Bind(wx.EVT_MENU, self.on_open_file, self.m_openfile)
        self.sld_graph.canvas.mpl_connect('motion_notify_event', self.UpdateStatusBar)
        self.d_graph.canvas.mpl_connect('motion_notify_event', self.UpdateStatusBar)
    def __del__(self):
        pass

    def on_open_file(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory,
            defaultFile="",
            wildcard="datafiles (*.csv,*.txt, *.dat)|*.csv;*.txt;*.dat",
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPaths()
            print("You chose the following file(s):")
            print(path[0])

            self.data = np.loadtxt(path[0], delimiter=',')
            xtemp = self.data[:,:2]
            self.x = xtemp.reshape((1,)+xtemp.shape)
            self.m_statusBar.SetStatusText('Data loaded')
            for item in xtemp:
                self.list_ctrl.Append(item)
            self.x = CNN.preprocess_x(self.x, train_bool=False)
            self.m_statusBar.SetStatusText('Data preprocessed')
        self.Bind(wx.EVT_BUTTON, self.on_predict_click, self.predict_button)

        dlg.Destroy()


    # def draw(self):
    #     self.sld_axes.plot(sld_)

    def clear(self, event):
        print('reee')
        self.list_ctrl.DeleteAllItems()
        self.data = None
        self.x = None
        # self.sld_graph = CanvasPanel(self.panel)
        # self.d_graph = CanvasPanel(self.panel)
        # self.sld_graph.draw(0,0)
        self.sld_graph.clear_graph()
        self.d_graph.clear_graph()

        self.output_box.SetValue("")

    def on_predict_click(self, event):

        self.layer_output = self.neural_net.n_class_model.predict(self.x)
        self.layer = np.argmax(self.layer_output)
        if self.layer == 0:
            self.m_statusBar.SetStatusText("No layers detected.")
        if self.layer == 1:
            self.sld_output = self.neural_net.single_sld_model.predict(self.x)
            self.d_output = self.neural_net.single_d_model.predict(self.x)

            self.sld_graph.draw(self.sld_output[0], self.neural_net.sld_bins)
            self.d_graph.draw(self.d_output[0], self.neural_net.d_bins)
        if self.layer == 2:
            print('asdf')
            self.sld_output = self.neural_net.multi_sld_model.predict(self.x)
            self.d_output = self.neural_net.multi_d_model.predict(self.x)

            self.sld_graph.draw(self.sld_output[0][0], self.neural_net.sld_bins)
            self.sld_graph.draw(self.sld_output[1][0], self.neural_net.sld_bins)
            self.d_graph.draw(self.d_output[0][0], self.neural_net.d_bins)
            self.d_graph.draw(self.d_output[1][0], self.neural_net.d_bins)

        self.output_box.SetValue(str(self.neural_net.result_parsing(self.sld_output, 'sld', layer_num=self.layer)))
        self.output_box.AppendText('\n')
        self.output_box.AppendText(str(self.neural_net.result_parsing(self.d_output, 'd', layer_num=self.layer)))
        print(self.sld_output[0][0])

    def UpdateStatusBar(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.m_statusBar.SetStatusText("x= " + str(x) + "  y=" + str(y))




if __name__ == "__main__":
    app = wx.App()
    frame = MyFrame(None)
    frame.Show()
    app.MainLoop()