import wx
import pylab
from single_factor_models.src.portfolio import main_var, main_es

class MyForm(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, title='My Form')
        
         # Add a panel so it looks correct on all platforms
        self.panel = wx.Panel(self, wx.ID_ANY)
        # create the handlers
        bmp = wx.ArtProvider.GetBitmap(wx.ART_NORMAL_FILE, wx.ART_OTHER, (16, 16))
        titleIco = wx.StaticBitmap(self.panel, wx.ID_ANY, bmp)
        title = wx.StaticText(self.panel, wx.ID_ANY, 'Economic Capital Calculator (EC=VaR|Es - EL)')
        
        bmp = wx.ArtProvider.GetBitmap(wx.ART_TIP, wx.ART_OTHER, (16, 16))
        inputNumOfCreditsIco = wx.StaticBitmap(self.panel, wx.ID_ANY, bmp)
        labelNumOfCredits = wx.StaticText(self.panel, wx.ID_ANY, 'number of credits')
        self.inputTxtNumOfCredits = wx.TextCtrl(self.panel, wx.ID_ANY, '40')
        
        inputPdIco = wx.StaticBitmap(self.panel, wx.ID_ANY, bmp)
        labelPd = wx.StaticText(self.panel, wx.ID_ANY, 'default probability')
        self.inputTxtPd = wx.TextCtrl(self.panel, wx.ID_ANY,'0.01')
        
        inputCorrIco = wx.StaticBitmap(self.panel, wx.ID_ANY, bmp)
        labelCorr = wx.StaticText(self.panel, wx.ID_ANY, 'correlation')
        self.inputTxtCorr = wx.TextCtrl(self.panel, wx.ID_ANY, '0.2')
        
        inputLGDIco = wx.StaticBitmap(self.panel, wx.ID_ANY, bmp)
        labelLGD = wx.StaticText(self.panel, wx.ID_ANY, 'loss given default')
        self.inputTxtLGD = wx.TextCtrl(self.panel, wx.ID_ANY, '1.0')
        
        inputPercentageStepIco = wx.StaticBitmap(self.panel, wx.ID_ANY, bmp)
        labelPercentageStep = wx.StaticText(self.panel, wx.ID_ANY, 'var percentage step')
        self.inputTxtPercentageStep = wx.TextCtrl(self.panel, wx.ID_ANY, '1%')

        inputVarStepIco = wx.StaticBitmap(self.panel, wx.ID_ANY, bmp)
        labelVarStep = wx.StaticText(self.panel, wx.ID_ANY, 'var value step')
        self.inputTxtVarStep = wx.TextCtrl(self.panel, wx.ID_ANY, '0.02')
        
        varBtn = wx.Button(self.panel, wx.ID_ANY, 'var')
        esBtn = wx.Button(self.panel, wx.ID_ANY, 'es')
        self.Bind(wx.EVT_BUTTON, self.onVAR, varBtn)
        self.Bind(wx.EVT_BUTTON, self.onEs, esBtn)
        # Wrap the handler in the BoxSizers
        topSizer        = wx.BoxSizer(wx.VERTICAL)
        titleSizer      = wx.BoxSizer(wx.HORIZONTAL)
        inputNumOfCreditsSizer   = wx.BoxSizer(wx.HORIZONTAL)
        inputPdSizer   = wx.BoxSizer(wx.HORIZONTAL)
        inputCorrSizer = wx.BoxSizer(wx.HORIZONTAL)
        inputLGDSizer  = wx.BoxSizer(wx.HORIZONTAL)
        inputPercentageStepSizer  = wx.BoxSizer(wx.HORIZONTAL)
        inputVarStepSizer  = wx.BoxSizer(wx.HORIZONTAL)
        btnSizer        = wx.BoxSizer(wx.HORIZONTAL)
        
        titleSizer.Add(titleIco, 0, wx.ALL, 5)
        titleSizer.Add(title, 0, wx.ALL, 5)
        
        inputNumOfCreditsSizer.Add(inputNumOfCreditsIco, 0, wx.ALL, 5)
        inputNumOfCreditsSizer.Add(labelNumOfCredits, 0, wx.ALL, 5)
        
        inputNumOfCreditsSizer.Add(self.inputTxtNumOfCredits, 1, wx.ALL|wx.EXPAND, 5)
        
        inputPdSizer.Add(inputPdIco, 0, wx.ALL, 5)
        inputPdSizer.Add(labelPd, 0, wx.ALL, 5)
        inputPdSizer.Add(self.inputTxtPd, 1, wx.ALL|wx.EXPAND, 5)
        
        inputCorrSizer.Add(inputCorrIco, 0, wx.ALL, 5)
        inputCorrSizer.Add(labelCorr, 0, wx.ALL, 5)
        inputCorrSizer.Add(self.inputTxtCorr, 1, wx.ALL|wx.EXPAND, 5)
        
        inputLGDSizer.Add(inputLGDIco, 0, wx.ALL, 5)
        inputLGDSizer.Add(labelLGD, 0, wx.ALL, 5)
        inputLGDSizer.Add(self.inputTxtLGD, 1, wx.ALL|wx.EXPAND, 5)

        inputPercentageStepSizer.Add(inputPercentageStepIco, 0, wx.ALL, 5)
        inputPercentageStepSizer.Add(labelPercentageStep, 0, wx.ALL, 5)
        inputPercentageStepSizer.Add(self.inputTxtPercentageStep, 1, wx.ALL|wx.EXPAND, 5)

        inputVarStepSizer.Add(inputVarStepIco, 0, wx.ALL, 5)
        inputVarStepSizer.Add(labelVarStep, 0, wx.ALL, 5)
        inputVarStepSizer.Add(self.inputTxtVarStep, 1, wx.ALL|wx.EXPAND, 5)
        
        btnSizer.Add(varBtn, 0, wx.ALL, 5)
        btnSizer.Add(esBtn, 0, wx.ALL, 5)
        # Add the BoxSizers to the main BoxSizer
        topSizer.Add(titleSizer, 0, wx.CENTER)
        topSizer.Add(wx.StaticLine(self.panel,), 0, wx.ALL|wx.EXPAND, 5)
        topSizer.Add(inputNumOfCreditsSizer, 0, wx.ALL|wx.EXPAND, 5)
        topSizer.Add(inputPdSizer, 0, wx.ALL|wx.EXPAND, 5)
        topSizer.Add(inputCorrSizer, 0, wx.ALL|wx.EXPAND, 5)
        topSizer.Add(inputLGDSizer, 0, wx.ALL|wx.EXPAND, 5)
        topSizer.Add(inputPercentageStepSizer, 0, wx.ALL|wx.EXPAND, 5)
        topSizer.Add(inputVarStepSizer, 0, wx.ALL|wx.EXPAND, 5)
        topSizer.Add(wx.StaticLine(self.panel), 0, wx.ALL|wx.EXPAND, 5)
        topSizer.Add(btnSizer, 0, wx.ALL|wx.CENTER, 5)
        
        self.panel.SetSizer(topSizer)
        topSizer.Fit(self)

    def get_grid(self):
        perc_step = float(self.inputTxtPercentageStep.Value.split('%')[0])/100.0
        var_step = float(self.inputTxtVarStep.Value)
        if perc_step > 0.30 or perc_step < 0.0:
            raise Exception("wrong percentage value : 0% < " + self.inputTxtPercentageStep.Value + " < 30% " )
        if var_step > (0.99 - 0.8) or var_step < 0.0:
            raise Exception("wrong loss step value : 0.0 < %s < %s " % 
                            (self.inputTxtVarStep.Value, (0.99 - 0.8)))
        x_list = pylab.arange(0.0, 0.30, perc_step)
        y_list = pylab.arange(0.8, 0.99, var_step)
        return x_list, y_list
                            
    def onVAR(self, event):
        # Do something
        try:
            x_list, y_list = self.get_grid()
            num_of_credits = int(self.inputTxtNumOfCredits.Value)
            default_probability = float(self.inputTxtPd.Value)
            correlation = float(self.inputTxtCorr.Value)
            loss_given_default = float(self.inputTxtLGD.Value)
            main_var(num_of_credits, default_probability, 
                     correlation, loss_given_default, x_list)
        except Exception, e:
            wx.MessageBox('%s'%e, 'Error', wx.OK | wx.ICON_ERROR)


 
    def onEs(self, event):
        try:
            x_list, y_list = self.get_grid()
            num_of_credits = int(self.inputTxtNumOfCredits.Value)
            default_probability = float(self.inputTxtPd.Value)
            correlation = float(self.inputTxtCorr.Value)
            loss_given_default = float(self.inputTxtLGD.Value)
            main_es(num_of_credits, default_probability, correlation, 
                    loss_given_default, x_list, y_list)
        except Exception, e:
            wx.MessageBox('%s'%e, 'Error', wx.OK | wx.ICON_ERROR)
        
 
    def closeProgram(self):
        self.Close()
 
 
# Run the program
if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = MyForm().Show()
    app.MainLoop()
