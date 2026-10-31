# This code creates the objects and prepares data to be used
# in an interactive Juyptyer notebook, implementing the 
# Orchestration Engine
#
# Aug 2019

import numpy as np
import glob

# get the list of FFT stubs
ffts = glob.glob('../Soundbank_McGill/FFTBank/*.txt')
ffts = [f.split('\\')[1].split('.')[0] for f in ffts]

# Get the 3-char list of instruments in the FFT Bank 
global instruments 
instruments = set([f.split('-')[0] for f in ffts])
instruments = list(instruments)
instruments.sort()
#instruments

# Target sound. Later we'll make an upload widget and a construction tab
# target = flu-a5-ord-mf

target_file = '../Soundbank_McGill/FFTBank/flu-a5-ord-mf.txt'
target = np.loadtxt(target_file, dtype='float32')


from scipy.optimize import nnls  # non-negative least square
import ipywidgets as widgets

selrbuts = []

# Dictionary to get real instrument names
inst_name = {
'afl':'Alto Flute',
'alx':'Alto Sax',
'bcl':'Bass Clarinet',
'bfl':'Bass Flute',
'brx':'Baritone Sax',
'bsn':'Bassoon',
'bsx':'Bass Saxophone',
'btb':'Bass Trombone',
'cbc':'Contra Bass Clarinet',
'cbs':'Contra Bassoon',
'clb':'Bb Clarinet',
'cle':'Eb Clarinet',
'cro':'Crotales',
'ctb':'Contrabass',
'enh':'English Horn',
'flu':'Flute',
'frh':'French Horn',
'glk':'Glockenspiel',
'mrb':'Marimba',
'obo':'Oboe',
'pic':'Piccolo',
'pno':'Piano',
'spx':'Soprano Sax',
'tba':'Tuba',
'tbb':'Tubular Bells',
'tnx':'Tenor Sax',
'tpc':'C Trumpet',
'trb':'Tenor Trombone',
'vib':'Vibraphone',
'vla':'Viola',
'vlc':'Cello',
'vln':'Violin',
'xyl':'Xylophone'
}



for i in instruments:
    selrbuts.append(widgets.Checkbox(
        value=False,
        description=inst_name[i],
        disabled=False))

        
bankgrid = widgets.GridBox(selrbuts, layout=widgets.Layout(grid_template_columns="repeat(5, 150px)"))


keeptopN = widgets.IntText(
    value=8,
    min=1,
    max=50,
    step=1,
    description='Keep top N',
    disabled=False,
#    continuous_update=False,
#    orientation='horizontal',
    readout=True,
    readout_format='d'
)



        
        
        
