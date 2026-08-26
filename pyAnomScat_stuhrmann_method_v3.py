import sys
from turtle import width

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMenu, QFileDialog
from PyQt6 import uic
import glob
import os
import numpy as np
import subprocess
import time
import shutil
import pyqtgraph as pg
import threading
import warnings
import re
from scanf import scanf 
import xraydb

warnings.filterwarnings('ignore')

# globel parameters that can be modified later or recovered later
param = dict()
param['energy resolution'] = 1.4e-4   # Si 111
param['test'] = True
param['energy_offset_momo'] = 3.0     # eV offset between recorded energy and real energy in eV
param['import_ascii_delimiter']=None
param['import_ascii_comments']='#'
param['default_element'] = 26

param['color_plots'] = ['#FF0000','#FF8429','#FFFF10','#D6EF39','#7BC618','#299C39','#089494','#00A5C6','#083194','#31007B','#8C007B','#CE007B']
param['color4line'] = ['#36454F','#E41A1C','#377EB8','#4DAF4A','#984EA3','#FF7F00','#A65628','#F781BF','#008B8B','#000080','#800000','#808000','#006400']
param['active_color_palette'] = 'default'
param['linewidth_show'] = 1
param['linestyle_show'] = QtCore.Qt.PenStyle.DotLine
param['linewidth_use'] = 2
param['linestyle_use'] = Qt.PenStyle.SolidLine
param['plotstyle'] = 'loglog'  # 'loglog', kratky

param['feff_plot_offset'] = 500 
param['feff_plot_stepsize'] = 0.2
param['feff_plot_f1f2'] = False
param['feff_plot_color'] = '#FF0000'
param['feff_plot_markerstyle'] = 'x'
param['feff_plot_markersize'] = 3
param['feff_plot_linestyle'] = Qt.PenStyle.DotLine
param['feff_plot_linewidth'] = 1

def str2bool(value):
    return value.lower() in ("yes", "true", "t", "1", b'true')


class PeriodicTableDialog(QtWidgets.QDialog):
    """Compact periodic-table selector returning an atomic number."""

    # Main-table positions use the standard 18-group layout. Lanthanides and
    # actinides are shown on two separate rows below it.
    _periods = (
        ((1, 1), (2, 18)),
        tuple(zip(range(3, 11), (1, 2, 13, 14, 15, 16, 17, 18))),
        tuple(zip(range(11, 19), (1, 2, 13, 14, 15, 16, 17, 18))),
        tuple((z, group) for z, group in zip(range(19, 37), range(1, 19))),
        tuple((z, group) for z, group in zip(range(37, 55), range(1, 19))),
        ((55, 1), (56, 2), (57, 3))
        + tuple((z, group) for z, group in zip(range(72, 87), range(4, 19))),
        ((87, 1), (88, 2), (89, 3))
        + tuple((z, group) for z, group in zip(range(104, 119), range(4, 19))),
    )

    def __init__(self, current_z=26, parent=None):
        super().__init__(parent)
        self.selected_z = None
        self.setWindowTitle("Select Resonant Element")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(
            "Select the resonant element from the periodic table:"
        )
        layout.addWidget(title)

        table = QtWidgets.QGridLayout()
        table.setHorizontalSpacing(3)
        table.setVerticalSpacing(3)
        for period_index, period in enumerate(self._periods):
            for atomic_number, group in period:
                self._add_element_button(
                    table, atomic_number, period_index, group - 1, current_z
                )

        lanthanides = range(58, 72)
        actinides = range(90, 104)
        table.addWidget(QtWidgets.QLabel("Lanthanides"), 7, 0, 1, 2)
        table.addWidget(QtWidgets.QLabel("Actinides"), 8, 0, 1, 2)
        for column, atomic_number in enumerate(lanthanides, start=2):
            self._add_element_button(table, atomic_number, 7, column, current_z)
        for column, atomic_number in enumerate(actinides, start=2):
            self._add_element_button(table, atomic_number, 8, column, current_z)
        layout.addLayout(table)

        current_symbol = xraydb.atomic_symbol(current_z)
        self.selection_label = QtWidgets.QLabel(
            f"Current: {current_symbol} (Z={current_z})"
        )
        layout.addWidget(self.selection_label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_element_button(self, layout, atomic_number, row, column, current_z):
        symbol = xraydb.atomic_symbol(atomic_number)
        name = xraydb.atomic_name(atomic_number).title()
        button = QtWidgets.QPushButton(symbol, self)
        button.setFixedSize(43, 38)
        button.setToolTip(f"{name} — Z={atomic_number}")
        if atomic_number == current_z:
            button.setStyleSheet("font-weight: bold; border: 2px solid #268bd2;")
        button.clicked.connect(
            lambda _checked=False, z=atomic_number: self._select_element(z)
        )
        layout.addWidget(button, row, column)

    def _select_element(self, atomic_number):
        self.selected_z = int(atomic_number)
        self.accept()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.appPath = os.path.dirname(os.path.realpath(__file__))  
        # The integrated window replaces this with pyFAI's current input path.
        self.appImportPath = self.appPath
        self.curves = list()
        self.curve_headers = list()
        self.curve_units = list()
        self.curve_has_error = list()
        self.factors = dict()
        self.lastcolorindex = -1
        self.Z = param['default_element']
        uic.loadUi(os.path.join(self.appPath, 'pyAnomScat_stuhrmann_method.ui'), self)
        self.initUI()
        self.initPlot()

    def initUI(self):
        # Import belongs to the File menu. Hide the legacy control-panel button
        # and remove both Quit controls; the window's normal close button is
        # sufficient and does not risk quitting the host pyFAI application.
        self.pushButton.hide()
        self.pushButton_7.hide()
        self.menuFile.removeAction(self.actionQuit)
        self.actionQuit.setVisible(False)
        self.actionImportASCII = QtGui.QAction("Import ASCII...", self)
        self.actionImportASCII.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.actionImportASCII.triggered.connect(self.openFileDialog)
        self.menuFile.addAction(self.actionImportASCII)
        self.pushButton_2.hide()
        self.actionImportHDF = QtGui.QAction("Import HDF/Nexus...", self)
        self.actionImportHDF.setToolTip(
            "HDF/Nexus import is not implemented in PyAnomScat"
        )
        self.actionImportHDF.setEnabled(False)
        self.menuFile.addAction(self.actionImportHDF)

        # Closing/hiding old menu controls leaves gaps in the Designer grid.
        # Reinsert the remaining controls as two compact, left-aligned columns.
        for control in (
            self.pushButton, self.pushButton_2, self.pushButton_7,
            self.pushButton_4, self.pushButton_5, self.pushButton_8,
            self.pushButton_6, self.pushButton_3, self.pushButton_9,
        ):
            self.gridLayout_3.removeWidget(control)
        spacer_item = self.gridLayout_3.itemAtPosition(1, 3)
        if spacer_item is not None:
            self.gridLayout_3.removeItem(spacer_item)
        self.gridLayout_3.addWidget(self.pushButton_3, 1, 0)
        self.gridLayout_3.addWidget(self.pushButton_9, 1, 1)
        self.gridLayout_3.addWidget(self.pushButton_4, 1, 2)
        self.gridLayout_3.addWidget(self.pushButton_8, 1, 3)
        self.gridLayout_3.addWidget(self.pushButton_5, 2, 0)
        self.gridLayout_3.addWidget(self.pushButton_6, 2, 1)
        for column in range(4):
            self.gridLayout_3.setColumnStretch(column, 0)
        self.gridLayout_3.setColumnStretch(4, 1)
        self.pushButton_6.clicked.connect(self.openExportDialog)
        self.pushButton_3.clicked.connect(self.setElement)
        self.pushButton_3.setEnabled(True)
        self._update_element_button()
        self.pushButton_8.clicked.connect(self.setChemicalShift)
        self.pushButton_4.clicked.connect(self.setMonoShift)
        self.pushButton_9.clicked.connect(self.getAnomalousFactors)
        self.pushButton_5.clicked.connect(self.runMatrixVersion)
        # connect the menus 
        self.actionLinLin.triggered.connect(self.setActionLinLin)
        self.actionLinLog.triggered.connect(self.setActionLinLog)
        self.actionLogLin.triggered.connect(self.setActionLogLin)
        self.actionLogLog.triggered.connect(self.setActionLogLog)
        self.actionKratky.triggered.connect(self.setActionKratky)
        self.colorMenu = self.menuEdit.addMenu("Color")
        self.colorActionGroup = QtGui.QActionGroup(self)
        self.colorActionGroup.setExclusive(True)
        self.defaultColorAction = self.colorMenu.addAction(
            "Current color list (Default)"
        )
        self.originColorAction = self.colorMenu.addAction("Origin Color4Line")
        for action in (self.defaultColorAction, self.originColorAction):
            action.setCheckable(True)
            self.colorActionGroup.addAction(action)
        self.defaultColorAction.setChecked(True)
        self.defaultColorAction.triggered.connect(
            lambda _checked=False: self.setColorPalette('default')
        )
        self.originColorAction.triggered.connect(
            lambda _checked=False: self.setColorPalette('color4line')
        )
        
        self.statusbar.showMessage(self.appImportPath)
        # table handling
        self.tableModel = TableModelContent()
        self.tableView.setModel(self.tableModel)
        self.tableModel.rowsReordered.connect(self._table_rows_reordered)
        self.tableView.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.tableView.setDragEnabled(True)
        self.tableView.setAcceptDrops(True)
        self.tableView.setDropIndicatorShown(True)
        self.tableView.setDragDropOverwriteMode(False)
        self.tableView.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )
        self.tableView.setDefaultDropAction(Qt.DropAction.MoveAction)
        header = self.tableView.horizontalHeader()
        header.setMinimumSectionSize(25)
        header.setStretchLastSection(False)
        for i in range(0,self.tableModel.columnCount(0)):
            # Interactive keeps every column manually resizable by dragging its
            # header boundary after applying the screenshot-based defaults.
            header.setSectionResizeMode(i,QtWidgets.QHeaderView.ResizeMode.Interactive)
        default_column_widths = (
            25, 42, 350, 72, 94, 96, 102,
            38, 58, 34, 47, 47, 52, 38,
        )
        for column, width in enumerate(default_column_widths):
            self.tableView.setColumnWidth(column, width)
        # Color selection is now centralized under Plot -> Color.
        self.tableView.setColumnHidden(8, True)
        self.tableView.clicked.connect(self.onTableViewClick)
        self.tableView.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tableView.customContextMenuRequested.connect(self.openContextMenu)
    
    def initPlot(self):
        ''' initialize the three plots - later I can add the parms here to modify the look '''
        # Use the normal light plotting palette to match the host application.
        pg.setConfigOptions(background='w', foreground='k')
        self.plotWidget = list()
        self.plotView = list()
        self.plotLayout = list()
        # graphics view 1
        view = self.graphicsView_1
        view.setBackground('w')
        l = pg.GraphicsLayout()
        view.setCentralItem(l)
        view.show()
        self.plotWidget.append(l.addPlot(0,0))
        self.plotWidget[-1].addLegend(offset=(10, 10))
        l.layout.setSpacing(20)
        l.setContentsMargins(20,20,20,20)
        self.plotLayout.append(l)
        self.plotView.append(view)
        self.plotWidget[-1].clear()
        self.plotWidget[-1].showGrid(x=True,y=True,alpha=0.3)
        if param['plotstyle'] == 'loglog':
            self.plotWidget[-1].setLabel('bottom','q (nm<sup>-1</sup>)',  **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLabel('left','I(q) (cm<sup>-1</sup>)', **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLogMode(x=True,y=True)
        elif param['plotstyle'] == 'kratky':
            self.plotWidget[-1].setLabel('bottom','q (nm<sup>-1</sup>)',  **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLabel('left','I(q)q<sup>2</sup> (cm<sup>-1</sup>nm<sup>-2</sup>)', **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLogMode(x=True,y=False)
        elif param['plotstyle'] == 'linlog':
            self.plotWidget[-1].setLabel('bottom','q (nm<sup>-1</sup>)',  **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLabel('left','I(q) (cm<sup>-1</sup>)', **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLogMode(x=False,y=True)
        elif param['plotstyle'] == 'loglin':
            self.plotWidget[-1].setLabel('bottom','q (nm<sup>-1</sup>)',  **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLabel('left','I(q) (cm<sup>-1</sup>)', **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLogMode(x=True,y=False)
        else:
            self.plotWidget[-1].setLabel('bottom','q (nm<sup>-1</sup>)',  **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLabel('left','I(q) (cm<sup>-1</sup>)', **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLogMode(x=False,y=False)
        

        # graphics view 2
        view = self.graphicsView_2
        view.setBackground('w')
        l = pg.GraphicsLayout()
        view.setCentralItem(l)
        view.show()
        self.plotWidget.append(l.addPlot(0,0))
        self.plotWidget[-1].addLegend(offset=(10, 10))
        l.layout.setSpacing(20)
        l.setContentsMargins(20,20,20,20)
        self.plotLayout.append(l)
        self.plotView.append(view)
        self.plotWidget[-1].clear()
        self.plotWidget[-1].showGrid(x=True,y=True,alpha=0.3)
        if param['plotstyle'] == 'loglog':
            self.plotWidget[-1].setLabel('bottom','q (nm<sup>-1</sup>)',  **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLabel('left','I(q) (cm<sup>-1</sup>)', **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLogMode(x=True,y=True)
        elif param['plotstyle'] == 'kratky':
            self.plotWidget[-1].setLabel('bottom','q (nm<sup>-1</sup>)',  **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLabel('left','I(q)q<sup>2</sup> (cm<sup>-1</sup>nm<sup>-2</sup>)', **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLogMode(x=True,y=False)
        elif param['plotstyle'] == 'linlog':
            self.plotWidget[-1].setLabel('bottom','q (nm<sup>-1</sup>)',  **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLabel('left','I(q) (cm<sup>-1</sup>)', **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLogMode(x=False,y=True)
        elif param['plotstyle'] == 'loglin':
            self.plotWidget[-1].setLabel('bottom','q (nm<sup>-1</sup>)',  **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLabel('left','I(q) (cm<sup>-1</sup>)', **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLogMode(x=True,y=False)
        else:
            self.plotWidget[-1].setLabel('bottom','q (nm<sup>-1</sup>)',  **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLabel('left','I(q) (cm<sup>-1</sup>)', **{'font-size':'12pt','color':'#202020'})
            self.plotWidget[-1].setLogMode(x=False,y=False)

        # graphics view 3
        view = self.graphicsView_3
        view.setBackground('w')
        l = pg.GraphicsLayout()
        view.setCentralItem(l)
        view.show()
        self.plotWidget.append(l.addPlot(0,0))
        l.layout.setSpacing(20)
        l.setContentsMargins(20,20,20,20)
        self.plotLayout.append(l)
        self.plotView.append(view)
        self.plotWidget[-1].clear()
        self.plotWidget[-1].showGrid(x=True,y=True,alpha=0.3)
        #self.plotWidget[-1].setLabels(bottom='energy (eV)', left='f<sub>eff</sub><sup>2</sup>(E)')
        self.plotWidget[-1].setLabel('bottom','energy (eV)', **{'font-size':'12pt','color':'#202020'})
        self.plotWidget[-1].setLabel('left','f<sub>eff</sub>(E)', **{'font-size':'12pt','color':'#202020'})
        self.plotWidget[-1].setLogMode(x=False,y=False)

    def setActionLinLin(self):
        param['plotstyle'] = 'linlin'        
        self.initPlot()
        self.updatePlotWidget1()

    def setActionLinLog(self):
        param['plotstyle'] = 'linlog'        
        self.initPlot()
        self.updatePlotWidget1()
    
    def setActionLogLin(self):
        param['plotstyle'] = 'loglin'        
        self.initPlot()
        self.updatePlotWidget1()

    def setActionLogLog(self):
        param['plotstyle'] = 'loglog'        
        self.initPlot()
        self.updatePlotWidget1()
    
    def setActionKratky(self):
        param['plotstyle'] = 'kratky'
        self.initPlot()
        self.updatePlotWidget1()
    
    def openExportDialog(self):
        # Use a dialog instance so New Folder works reliably and the selected
        # directory/filename can be handled independently on Windows.
        start_directory = self.appImportPath
        if not os.path.isdir(start_directory):
            start_directory = os.path.dirname(start_directory)
        if not start_directory or not os.path.isdir(start_directory):
            start_directory = os.getcwd()

        dialog = QFileDialog(self, 'Export ASCII curves!', start_directory)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter('Dat Files (*.dat)')
        dialog.setDefaultSuffix('dat')
        first_filename = None
        result = getattr(self, 'result', None)
        if isinstance(result, dict) and result.get('files'):
            first_filename = result['files'][0]
        elif self.tableModel.rowCount() > 0:
            first_filename = self.tableModel.get(0, 2)
        dialog.selectFile(first_filename or 'ASAXS_curves.dat')
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        selected = dialog.selectedFiles()
        if not selected:
            return

        # The chosen name is a base name: three related curve files are saved.
        selected_path = os.path.abspath(os.path.normpath(selected[0]))
        fname, _ = os.path.splitext(selected_path)
        fname_I0 = f'{fname}_I0.dat'
        fname_I0R = f'{fname}_I0R.dat'
        fname_IR = f'{fname}_IR.dat'

        try:
            required = ('files', 'nrj', 'f1', 'f2', 'feff', 'q', 'I0', 'I0R', 'IR')
            missing = [key for key in required if key not in self.result]
            if missing:
                raise ValueError(
                    'Calculate the ASAXS curves before exporting. Missing: '
                    + ', '.join(missing)
                )

            header = 'File 1: %s (energy=%.2f eV, f0=%d, f1=%.4f, f2=%.4f, feff=%.4f)'%(self.result['files'][0],self.result['nrj'][0], self.Z, self.result['f1'][0],self.result['f2'][0],self.result['feff'][0])
            header = '%s\nFile 2: %s (energy=%.2f eV, f0=%d, f1=%.4f, f2=%.4f, feff=%.4f)'%(header,self.result['files'][1], self.result['nrj'][1], self.Z, self.result['f1'][1],self.result['f2'][1],self.result['feff'][1])
            header = '%s\nFile 3: %s (energy=%.2f eV, f0=%d, f1=%.4f, f2=%.4f, feff=%.4f)'%(header,self.result['files'][2], self.result['nrj'][2], self.Z, self.result['f1'][2],self.result['f2'][2],self.result['feff'][2])
            source_header = self.result.get('source_header', 'q_nm^-1\tIntensity')
            header = f'{header}\n{source_header}\nq,I(q)'
            dataI0 = np.transpose(np.vstack(([self.result['q']],[self.result['I0']])))
            dataI0R = np.transpose(np.vstack(([self.result['q']],[self.result['I0R']])))
            dataIR = np.transpose(np.vstack(([self.result['q']],[self.result['IR']])))
            np.savetxt(fname_I0, dataI0, delimiter=',',header=header)
            np.savetxt(fname_I0R, dataI0R, delimiter=',',header=header)
            np.savetxt(fname_IR, dataIR, delimiter=',',header=header)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, 'Export Curve Failed',
                f'Unable to save ASAXS curves:\n{exc}'
            )
            return

        self.appImportPath = os.path.dirname(selected_path)
        QtWidgets.QMessageBox.information(
            self, 'Export Curve Complete',
            'Saved three curve files:\n'
            f'{os.path.basename(fname_I0)}\n'
            f'{os.path.basename(fname_I0R)}\n'
            f'{os.path.basename(fname_IR)}'
        )
            

    def openFileDialog(self, type=1):
        title = 'Import ASCII curves!'
        options = QFileDialog.Option.DontUseNativeDialog
        options |= QFileDialog.Option.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, title, self.appImportPath, "All Files (*);;Dat Files (*.dat)","Dat Files (*.dat)",options=options)
        if files:
            # New rows inherit the shift settings already in use. This keeps a
            # later import consistent with the current measurement series.
            if self.tableModel.rowCount() > 0:
                mono_shift = float(self.tableModel.get(0, 4))
                chem_shift = float(self.tableModel.get(0, 5))
            else:
                mono_shift = float(param['energy_offset_momo'])
                chem_shift = 0.0
            for item in files:
                if self.tableModel.rowCount() == 0:
                    idx = 0
                    self.tableView.setEnabled(True)
                else:
                    idx = self.tableModel.rowCount()
                idx +=1
                # Split on the final path separator so the Control Panel shows
                # the directory and the .dat filename in separate columns.
                normalized_item = os.path.normpath(item)
                file_path, fname = os.path.split(normalized_item)
                # try to figure out if energy is part of the sample name
                name = fname.rsplit('.',1)[0]
                try:
                    result = scanf('%s_E%f',name)
                    if len(result)==2:
                        result = float(result[-1]) 
                    else:
                        result = 0.0
                except:
                    result = 0.0    # set default energy to zero to make the user aware of  the missing energy information
                if result == 0.0:
                    try:
                        with open(item, 'r', encoding='utf-8-sig', errors='replace') as stream:
                            header = ''.join(stream.readline() for _ in range(30))
                        match = re.search(
                            r'Energy\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*eV',
                            header,
                            flags=re.IGNORECASE,
                        )
                        if match:
                            result = float(match.group(1))
                    except OSError:
                        pass
                
                try:
                    source_header, q_unit, has_error = self._read_curve_header(item)
                    data = np.loadtxt(item,delimiter=param['import_ascii_delimiter'],comments=param['import_ascii_comments'])
                    self.curves.append(np.array(data))
                    self.curve_headers.append(source_header)
                    self.curve_units.append(q_unit)
                    self.curve_has_error.append(has_error and data.shape[1] >= 3)
                except:
                    continue
                
                # get color
                cindex = self.lastcolorindex + 1
                if cindex >= len(param['color_plots']): cindex = -1
                self.lastcolorindex = cindex
                r2 = result + mono_shift + chem_shift
                palette = param[
                    'color4line' if param['active_color_palette'] == 'color4line'
                    else 'color_plots'
                ]
                color = palette[cindex % len(palette)]
                values = [idx,file_path,fname,result,mono_shift,chem_shift,r2, True,color,False,0.0,0.0,0.0,'Del']
                self.tableModel.add(values)
            self.updatePlotWidget1()
            # activate some buttons
            self.pushButton_4.setEnabled(True)
            self.pushButton_8.setEnabled(True)
            self.pushButton_9.setEnabled(True)
            self.pushButton_3.setEnabled(True)

    @staticmethod
    def _read_curve_header(filename):
        """Read the comment immediately above the first numeric data row."""
        previous_line = ''
        header_line = ''
        with open(filename, 'r', encoding='utf-8-sig', errors='replace') as stream:
            for line in stream:
                stripped = line.strip()
                if not stripped:
                    previous_line = ''
                    continue
                if stripped.startswith(param['import_ascii_comments']):
                    previous_line = stripped
                    continue
                try:
                    fields = re.split(r'[\s,]+', stripped)
                    float(fields[0])
                    float(fields[1])
                except (ValueError, IndexError):
                    previous_line = stripped
                    continue
                if previous_line.startswith(param['import_ascii_comments']):
                    header_line = previous_line
                break

        tokens = re.split(
            r'[\s,]+',
            header_line.lstrip(param['import_ascii_comments']).strip(),
        )
        tokens = [token for token in tokens if token]
        source_header = '\t'.join(tokens[:2]) if len(tokens) >= 2 else 'q_nm^-1\tIntensity'
        first_column = tokens[0].lower() if tokens else 'q_nm^-1'
        q_unit = 'A' if first_column.startswith('q_a^-1') else 'nm'
        third_column = tokens[2].casefold() if len(tokens) >= 3 else ''
        error_names = {
            'sigma', 'error', 'errors', 'uncertainty', 'uncertainties',
            'std', 'stdev', 'stddev', 'standard_deviation',
        }
        has_error = third_column in error_names
        return source_header, q_unit, has_error

    def updatePlotWidget1(self):
        ''' this will update the plot widget of the input curves'''
        rows = self.tableModel.rowCount()
        p = self.plotWidget[0]
        p.clear()
        for i in range(0,rows):
            legend = f"{self.tableModel.get(i, 2)} ({self.tableModel.get(i, 6):g} eV)"
            if self.tableModel.get(i,7)==True and self.tableModel.get(i,9) == False:
                c = self.curves[i]
                if param['plotstyle'] == 'kratky':
                    p.plot(c[:,0],c[:,1]*c[:,0]**2,pen=pg.mkPen(self.tableModel.get(i,8),width=param['linewidth_show'],style=param['linestyle_show']), name=legend)
                else:
                    p.plot(c[:,0],c[:,1],pen=pg.mkPen(self.tableModel.get(i,8),width=param['linewidth_show'],style=param['linestyle_show']), name=legend)
            elif self.tableModel.get(i,7)==True and self.tableModel.get(i,9)==True:
                c = self.curves[i]
                if param['plotstyle'] == 'kratky':
                    p.plot(c[:,0],c[:,1]*c[:,0]**2,pen=pg.mkPen(self.tableModel.get(i,8),width=param['linewidth_use'],style=param['linestyle_use']), name=legend)
                else:
                    p.plot(c[:,0],c[:,1],pen=pg.mkPen(self.tableModel.get(i,8),width=param['linewidth_use'],style=param['linestyle_use']), name=legend)
        self._auto_scale_plot(p)
        self._update_q_axis_labels()

    def _update_q_axis_labels(self, curve_index=0):
        """Use the imported q header for the input and result X axes."""
        unit = (
            self.curve_units[curve_index]
            if 0 <= curve_index < len(self.curve_units)
            else 'nm'
        )
        label = 'q (A<sup>-1</sup>)' if unit == 'A' else 'q (nm<sup>-1</sup>)'
        self.plotWidget[0].setLabel(
            'bottom', label, **{'font-size': '12pt', 'color': '#202020'}
        )
        self.plotWidget[1].setLabel(
            'bottom', label, **{'font-size': '12pt', 'color': '#202020'}
        )

    @staticmethod
    def _auto_scale_plot(plot):
        """Enable and immediately apply automatic X/Y range scaling."""
        plot.autoRange()
        plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

    def setColorPalette(self, palette_name):
        """Apply a Plot-menu color list to existing and future curves."""
        param['active_color_palette'] = palette_name
        palette = param['color4line' if palette_name == 'color4line' else 'color_plots']
        for row in range(self.tableModel.rowCount()):
            self.tableModel.setAt(row, 8, palette[row % len(palette)])
        self.lastcolorindex = self.tableModel.rowCount() - 1
        self.updatePlotWidget1()

    def onTableViewClick(self,evt):
        r = evt.row()
        c = evt.column()
        # show/use columns toggle their boolean values.
        if c == 7 or c == 9:
            current = self.tableModel.get(r,c)
            if current == True: 
                current = False
            else:
                current = True
            self.tableModel.setAt(r,c,current)
            self.updatePlotWidget1()
            if c == 9:
                if not 'nrj_all' in self.factors:
                    self.getAnomalousFactors()
                else:
                    self.updatePlotWidget3()
                self.enableCalculation()
        elif c == 8:
            # color choose
            color = QtWidgets.QColorDialog.getColor()
            if color.isValid:
                self.tableModel.setAt(r,c,color.name().upper())
                self.updatePlotWidget1()
        elif c == 13:
            self.removeDataset(evt)
        elif c == 0:
            # open remove context menu
            pass

        self.tableView.clearSelection()

    def openContextMenu(self,point):
        index = self.tableView.indexAt(point)
        if index.isValid():
            menu = QMenu()
            remove = menu.addAction('remove dataset')
            remove.triggered.connect(lambda x: self.removeDataset(index))
            menu.exec(self.tableView.viewport().mapToGlobal(point))
            menu.close()
    
    def removeDataset(self,index):
        self.curves.pop(index.row())
        self.curve_headers.pop(index.row())
        self.curve_units.pop(index.row())
        self.curve_has_error.pop(index.row())
        self.tableModel.removeDataset(index)
        self.updatePlotWidget1()
        if self.tableModel.rowCount() == 0:
            self.factors = dict()
            self.plotWidget[1].clear()
            self.plotWidget[2].clear()
            self.pushButton_5.setEnabled(False)
            self.pushButton_6.setEnabled(False)
        elif 'nrj_all' in self.factors:
            self.getAnomalousFactors()
        self.enableCalculation()

    @QtCore.pyqtSlot(int, int)
    def _table_rows_reordered(self, source_row, target_row):
        """Keep curve arrays aligned with rows moved by table drag-and-drop."""
        curve = self.curves.pop(source_row)
        self.curves.insert(target_row, curve)
        header = self.curve_headers.pop(source_row)
        self.curve_headers.insert(target_row, header)
        unit = self.curve_units.pop(source_row)
        self.curve_units.insert(target_row, unit)
        has_error = self.curve_has_error.pop(source_row)
        self.curve_has_error.insert(target_row, has_error)
        self.updatePlotWidget1()
        if 'nrj_all' in self.factors:
            self.getAnomalousFactors()

    def setChemicalShift(self):
        ''' upate all chemical shifts'''
        val, okPressed = QtWidgets.QInputDialog.getDouble(self, "Get value", "<b>Get chemical energy shift in eV:</b><br><i>energy<sub>sample</sub> = energy + energy<sub>shift</sub></i> ",0.0,-50.0, 50.0,3)
        if okPressed:
            for i in range(0,self.tableModel.rowCount()):
                self.tableModel.setAt(i,5,val)
                # recalculate energy final
                nrj = self.tableModel.get(i,3) +self.tableModel.get(i,4)+self.tableModel.get(i,5)
                self.tableModel.setAt(i,6,nrj)
            self.getAnomalousFactors()
    
    def setMonoShift(self):
        ''' upate all monochromator energy shifts'''
        val, okPressed = QtWidgets.QInputDialog.getDouble(self, "Get value", "<b>Get monochromator energy shift in eV:</b><br><i>energy<sub>real</sub> = energy + energy<sub>shift</sub></i> ",0.0,-50.0, 50.0,3)
        if okPressed:
            for i in range(0,self.tableModel.rowCount()):
                self.tableModel.setAt(i,4,val)
                # recalculate energy final
                nrj = self.tableModel.get(i,3) +self.tableModel.get(i,4)+self.tableModel.get(i,5)
                self.tableModel.setAt(i,6,nrj)
            self.getAnomalousFactors()
    
    def setElement(self):
        """Select the resonant element from a periodic-table dialog."""
        dialog = PeriodicTableDialog(self.Z, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.Z = dialog.selected_z
            self._update_element_button()

    def _update_element_button(self):
        """Show both the selected symbol and atomic number on the main button."""
        symbol = xraydb.atomic_symbol(self.Z)
        self.pushButton_3.setText(f"Element: {symbol} (Z={self.Z})")
        self.pushButton_3.setToolTip(
            "Click to select the resonant element from the periodic table"
        )
                 
    def getAnomalousFactors(self):
        self.statusbar.showMessage('get anomalous scattering factors')
        self.factors = dict()
        if self.tableModel.rowCount() < 1 or self.tableModel.get(0, 0) < 0:
            QtWidgets.QMessageBox.warning(
                self, 'Energy data required',
                'Import at least one ASCII curve before calculating anomalous factors.'
            )
            return
        # get energy list 
        nrj_real = list()
        nrj_chem = list()
        for i in range(0,self.tableModel.rowCount()):
            nrj_real.append(self.tableModel.get(i,3)+self.tableModel.get(i,4))
            nrj_chem.append(self.tableModel.get(i,6))
        #_, idx = np.unique(np.array(nrj_real),True)
        nrj_real = np.array(nrj_real)[:]  # idx
        nrj_chem = np.array(nrj_chem)[:]  # idx
        
        index_use = list()
        for i in range(0,self.tableModel.rowCount()):
            if self.tableModel.get(i,9):
                index_use.append(i)

        nrj_min = min(nrj_real)
        nrj_max = max(nrj_real)
        if not np.all(np.isfinite(nrj_real)) or nrj_min <= 0:
            QtWidgets.QMessageBox.warning(
                self, 'Invalid energy',
                'All imported curves require a positive energy in eV. '
                'Use filenames containing _E<energy> or pyFAI files with an '
                '"Energy: <value> eV" header.'
            )
            return
             
        
        # convolution stuff
        c_nrj_start = nrj_min - param['feff_plot_offset'] - 50
        c_nrj_end = nrj_max + param['feff_plot_offset'] + 50
        chantler_energy = np.asarray(xraydb.chantler_energies(self.Z), dtype=float)
        c_nrj_start = max(c_nrj_start, float(np.min(chantler_energy)))
        c_nrj_end = min(c_nrj_end, float(np.max(chantler_energy)))
        if c_nrj_start >= c_nrj_end:
            QtWidgets.QMessageBox.warning(
                self, 'Energy outside Chantler range',
                f'The selected energies are outside the xraydb Chantler range '
                f'for Z={self.Z}.'
            )
            return
        c_nrj_steps = int((c_nrj_end-c_nrj_start)/(param['feff_plot_stepsize']))
        c_X = np.linspace(c_nrj_start,c_nrj_end,c_nrj_steps+1)      
        c_f0 = xraydb.f0(self.Z,0.0)
        c_f1raw = xraydb.f1_chantler(self.Z,c_X)
        c_f2raw = xraydb.f2_chantler(self.Z,c_X)
        c_feffraw = np.sqrt(c_f0**2 + 2.0*c_f0*c_f1raw + c_f1raw**2 + c_f2raw**2)

        deltaX = np.mean(np.diff(c_X))
        
        # gaussion kernal
        s = nrj_max * param['energy resolution']
        gx = np.linspace(-50,50,int(100/deltaX)+1)
        gy = np.exp(-(gx)**2/(2*s**2))/(s*np.sqrt(2*np.pi))
        conv_feff = np.convolve(c_feffraw,gy,'same')/np.sum(gy)
        conv_f1 = np.convolve(c_f1raw,gy,'same')/np.sum(gy)
        conv_f2 = np.convolve(c_f2raw,gy,'same')/np.sum(gy)
        
        feff_raw = c_feffraw[50:-50]
        feff = conv_feff[50:-50]
        X = c_X[50:-50]
        
        self.factors['nrj_all']=X
        self.factors['f0_all'] = c_f0
        self.factors['f1_all'] = conv_f1[50:-50]
        self.factors['f2_all'] = conv_f2[50:-50]
        self.factors['feff_all']=feff
        self.factors['feff_raw']=feff_raw

        self.factors['nrj_real'] = nrj_real
        self.factors['nrj_chem'] = nrj_chem

        try:
            feff_use = np.interp(nrj_chem, X,feff)
        
            f1_use = np.interp(nrj_chem,X,conv_f1[50:-50])
            f2_use = np.interp(nrj_chem,X,conv_f2[50:-50])

            self.factors['feff_use'] = feff_use[:]
            self.factors['f1_use'] = f1_use[:]
            self.factors['f2_use'] = f2_use[:]
            self.factors['index_use']=index_use
        except:
            pass

        # update table model
        for i in range(0,self.tableModel.rowCount()):
            self.tableModel.setAt(i,10,self.factors['f1_use'][i])
            self.tableModel.setAt(i,11,self.factors['f2_use'][i])
            self.tableModel.setAt(i,12,self.factors['feff_use'][i])
        self.updatePlotWidget3()
        #activate the calculation buttons
        self.enableCalculation()
    
    def enableCalculation(self):
        ''' enable the calculation only if the use number is 3 and more'''
        nr = 0
        for i in range(0,self.tableModel.rowCount()):
            if self.tableModel.get(i,9):
                nr +=1
        if nr >= 3:
            self.pushButton_5.setEnabled(True)
        else:
            self.pushButton_5.setEnabled(False)
    
    def updatePlotWidget3(self):
        # check which columns are selected
        index_use = list()
        for i in range(0,self.tableModel.rowCount()):
            if self.tableModel.get(i,9):
                index_use.append(i)
        
        p = self.plotWidget[2]

        p.clear()
        cc = self.tableModel.get(0,5)
        p.plot(self.factors['nrj_all'],self.factors['feff_raw'],pen='g')
        p.plot(self.factors['nrj_all'],self.factors['feff_all'],pen='r')
        #print(self.factors['nrj_real'])
        #print(self.factors['feff_use'])
        #print(index_use)
        try:
            p.plot(self.factors['nrj_real'][index_use]+cc,self.factors['feff_use'][index_use],pen=None, symbolBrush = param['feff_plot_color'],symbolPen = 'w', symbol = 'o', symbolSize=14)
            #for i in range(0,self.tableModel.rowCount()):
                #self.tableModel.get(i,7)
                #p.plot(self.factors['nrj_real'][index_use],self.factors['feff_use'][index_use],pen=None, symbolBrush = param['feff_plot_color'],symbolPen = 'w', symbol = 'o', symbolSize=14)
                #for i in index_use:
                #   print('hier')
                #    print(self.factors['nrj_real'][i])
                #    print(self.factors['feff_use'][i])
                #    p.plot(self.factors['nrj_real'][i],self.factors['feff_use'][i],pen=None, symbolBrush = param['feff_plot_color'], symbolPen = 'w' , symbol = 'o', symbolSize=14)
        except:
            pass
        self._auto_scale_plot(p)


    def runMatrixVersion(self):
        index_use = list()
        f0 = list()
        for i in range(0,self.tableModel.rowCount()):
            if self.tableModel.get(i,9):
                index_use.append(i)
                f0.append(self.factors['f0_all'][0])
        
        #index_use = np.flip(np.array(index_use))
        f0 = np.array(f0)
        f1 = self.factors['f1_use'][index_use]
        f2 = self.factors['f2_use'][index_use] 

        curves = list()
        for item in index_use:
            curves.append(self.curves[item])
        # Keep curves as a list: two-column and Sigma-bearing three-column
        # files may be selected together and can have different shapes.
        use_errors = all(self.curve_has_error[item] for item in index_use)
        q1 = curves[0][:,0]
        q2 = curves[1][:,0]
        q3 = curves[2][:,0]
        i1 = curves[0][:,1]
        i2 = curves[1][:,1]
        i3 = curves[2][:,1]
        #print(f0)
        #print(f1)
        #print(f2)
        
        p = self.plotWidget[1]
        p.clear()
        self._update_q_axis_labels(index_use[0] if index_use else 0)
        #p.plot(q1,i1)
        #p.plot(q2,i2)
        #p.plot(q3,i3)

        # now apply the equations for the resonant curve
        i12 = i1-i2
        i13 = i1-i3

        f1_12 = f1[0]-f1[1]
        f1_13 = f1[0]-f1[2]
        K = f1[1]-f1[2] + (f2[0]**2-f2[1]**2)/f1_12 - (f2[0]**2-f2[2]**2)/f1_13
        t1 = (i12)/f1_12
        t2 = (i13)/f1_13
        #v2 = 1.0/K * ((i12)/f1_12 - (i13)/f1_13)
        #if K > 0:
        #    v2 = 1.0/K * (t2-t1)    
        #else:
        v2 = 1.0/K * (t1-t2)
        #print(K)
        #print(t1-t2)
                
        #p.plot(q1,np.abs(v2),pen='r')

        # try with the matrix version
        # Edit's my Martin Fisk's group below here
        matA = np.zeros((len(f1),3))
        I_E = np.zeros((len(q1),len(f1)))
        for i in range(len(f1)):
            I_E[:,i] = curves[i][:,1]
            matA[i,0] = 1.0
            matA[i,1] = 2.0*f1[i]
            matA[i,2] = f1[i]**2 + f2[i]**2

        #matA[0,0] = 1.0
        #matA[0,1] = 2.0*f1[0]
        #matA[0,2] = f1[0]**2 + f2[0]**2
        #matA[1,0] = 1.0
        #matA[1,1] = 2.0*f1[1]
        #matA[1,2] = f1[1]**2 + f2[1]**2
        #matA[2,0] = 1.0
        #matA[2,1] = 2.0*f1[2]
        #matA[2,2] = f1[2]**2 + f2[2]**2

        I0 = np.copy(q1)
        I0R = np.copy(q1)
        IR = np.copy(q1)


        for i, q in enumerate(q1):
            #y = [i1[i],i2[i],i3[i]]
            y= I_E[i,:]

            #x = np.linalg.solve(matA,y)
            if use_errors:
                sigma = np.asarray([curve[i, 2] for curve in curves])
                valid_sigma = np.all(np.isfinite(sigma)) and np.all(sigma > 0)
            else:
                valid_sigma = False
            if valid_sigma:
                weights = 1.0 / sigma
                res = np.linalg.lstsq(
                    matA * weights[:, np.newaxis], y * weights, rcond=None
                )
            else:
                res = np.linalg.lstsq(matA, y, rcond=None)
            x = res[0]
            #print(x)
            I0[i] = x[0]
            I0R[i] = x[1]
            IR[i] = x[2]
        
        # Three distinct result colors, each one width step thicker than the
        # previous default-width curves for clearer comparison.
        p.plot(q1,np.abs(I0),pen=pg.mkPen('#2ca02c', width=2),name='I0')
        p.plot(q1,np.abs(I0R),pen=pg.mkPen('#1f77b4', width=2),name='I0R')
        p.plot(
            q1, np.abs(IR),
            pen=pg.mkPen('#d95f02', width=2, style=Qt.PenStyle.DotLine),
            name='IR',
        )
        self._auto_scale_plot(p)

        #Icheck = np.sqrt(np.abs(I0))*np.sqrt(np.abs(IR))
        #p.plot(q1,Icheck,pen=pg.mkPen('#FF44F4', style=QtCore.Qt.DotLine))
        #p.plot(q1,np.abs(I0R-Icheck),pen=pg.mkPen('#0344F4', style=QtCore.Qt.DotLine))
        self.result = dict()
        self.result['q'] = q1[:]
        self.result['I0'] = np.abs(I0[:])
        self.result['I0R'] = np.abs(I0R[:])
        self.result['IR'] = np.abs(IR[:])
        self.result['index_use']=index_use
        fname = list()
        nrj = list()
        f1_res = list()
        f2_res = list()
        feff_res = list()
        for i, item in enumerate(index_use):
            fname.append(self.tableModel.get(i,2))
            nrj.append(self.tableModel.get(i,6))
            f1_res.append(self.tableModel.get(i,10))
            f2_res.append(self.tableModel.get(i,11))
            feff_res.append(self.tableModel.get(i,12))

        self.result['files'] = fname
        self.result['nrj'] = nrj
        self.result['f1'] = f1_res
        self.result['f2'] = f2_res
        self.result['feff'] = feff_res
        first_source_index = index_use[0] if index_use else 0
        self.result['source_header'] = (
            self.curve_headers[first_source_index]
            if first_source_index < len(self.curve_headers)
            else 'q_nm^-1\tIntensity'
        )
        
        self.pushButton_6.setEnabled(True)
        


class TableModelContent(QtCore.QAbstractTableModel):
    ''' class to handle the table look and the data '''
    rowsReordered = QtCore.pyqtSignal(int, int)
    MIME_TYPE = 'application/x-pyanomscat-table-row'

    def __init__(self, *args, **kwargs):
        super(TableModelContent,self).__init__(*args,**kwargs)
        self._data = []
        self._header = ['id', 'path', 'file name','energy (eV)','mono shift (eV)','chem. shift (eV)','energy final (eV)','show','color','use','f1(E)','f2(E)','feff(E)','Del']

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.ItemIsDropEnabled
        flags = (
            Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled
        )
        if index.column() == 4 or index.column() == 5:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def supportedDropActions(self):
        return Qt.DropAction.MoveAction

    def mimeTypes(self):
        return [self.MIME_TYPE]

    def mimeData(self, indexes):
        mime = QtCore.QMimeData()
        rows = sorted({index.row() for index in indexes if index.isValid()})
        if rows:
            mime.setData(self.MIME_TYPE, str(rows[0]).encode('ascii'))
        return mime

    def dropMimeData(self, data, action, row, column, parent):
        if action != Qt.DropAction.MoveAction or not data.hasFormat(self.MIME_TYPE):
            return False
        source_row = int(bytes(data.data(self.MIME_TYPE)).decode('ascii'))
        destination = row
        if destination < 0:
            destination = parent.row() if parent.isValid() else self.rowCount()
        destination = max(0, min(destination, self.rowCount()))
        target_row = destination - 1 if destination > source_row else destination
        if target_row == source_row:
            return False
        self.beginResetModel()
        record = self._data.pop(source_row)
        self._data.insert(target_row, record)
        for index, item in enumerate(self._data, start=1):
            item[0] = index
        self.endResetModel()
        self.rowsReordered.emit(source_row, target_row)
        return True

    def setData(self, index, value, role):
        if role == QtCore.Qt.EditRole:
            if index.column() == 4 or index.column()==5:
                try:
                    self._data[index.row()][index.column()] = float(value)
                except:
                    return False
            elif index.column() == 8:
                try:
                    self._data[index.row()][index.column()] = value
                except:
                    return False
            self.layoutChanged.emit()
            return True
        return False       

    def removeDataset(self, index):
        r = index.row()
        self.beginRemoveRows(QtCore.QModelIndex(), r, r)
        self._data.pop(r)
        self.endRemoveRows()

        nr = self.rowCount()
        for i in range(0,nr):
            self._data[i][0] = i+1

        self.layoutChanged.emit()

    def add(self,values):
        row = len(self._data)
        self.beginInsertRows(QtCore.QModelIndex(), row, row)
        self._data.append(values)
        self.endInsertRows()

    def setAt(self,row,col,value):
        self._data[row][col] = value
        self.layoutChanged.emit()

    def get(self, row, col):
        return self._data[row][col]

    def getColumnValues(self, row):
        col = range(0,self.columnCount())
        cell = []
        cell = [self._data[row][c] for c in col]
        return cell

    def set(self, data):
        self._data = data
        self.layoutChanged.emit()
    
    def rowCount(self,index=None):
        try:
            return len(self._data)
        except:
            return 0
    
    def columnCount(self, index=None):
        return len(self._header)

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._header[section])  

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QtCore.QVariant()

        elif role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 13:
                return 'Del'
            value = self._data[index.row()][index.column()]
            if isinstance(value, float):
                return "%.4f" % value
            if isinstance(value, str):
                return '%s' % value
            if isinstance(value, bool):
                if value == True:
                    return 'x'
                else:
                    return '-'
            return value
        
        elif role == Qt.ItemDataRole.ForegroundRole:
            if index.column() == 13:
                return QtGui.QColor(190, 30, 30)
            value = self._data[index.row()][index.column()]
            if isinstance(value,bool):
                if value:
                    return QtGui.QColor(0,255,0)
                else:
                    return QtGui.QColor(255,0,0)

        elif role == Qt.ItemDataRole.FontRole:
            if index.column() == 13:
                font = QtGui.QFont()
                font.setBold(True)
                return font
            value = self._data[index.row()][index.column()]
            if isinstance(value,bool):
                font = QtGui.QFont()
                font.setBold(True)
                return font
        
        elif role == Qt.ItemDataRole.BackgroundRole:
            value = self._data[index.row()][index.column()]
            if isinstance(value,str):
                if value.startswith('#'):
                    return QtGui.QBrush(QtGui.QColor(value))


        #elif role == QtCore.Qt.DecorationRole:
        #    value = self._data[index.row()][index.column()]
        #    if isinstance(value,bool):
        #        if value:
        #            return QtGui.QIcon('tick.png')
        #        else:
        #            return QtGui.QIcon('cross.png')
        
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if index.column() > 0:
                return Qt.AlignmentFlag.AlignCenter
            else:
                return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        else:
            return QtCore.QVariant()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
