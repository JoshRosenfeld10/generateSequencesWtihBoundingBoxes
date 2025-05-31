import logging
import os
from typing import Annotated, Optional


import slicer
import scipy
import cv2
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
import numpy

from slicer import vtkMRMLScalarVolumeNode
try:
    import pandas
except ModuleNotFoundError:
    slicer.util.pip_install('pandas')
try:
    import cv2
except ModuleNotFoundError:
    slicer.util.pip_install('opencv-python')
import vtk

#
# generateSequencesWithBoundingBoxes
#


class generateSequencesWithBoundingBoxes(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Generate Sequences with Bounding Boxes"
        self.parent.categories = ["Sequence Utilities"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Rebecca Hisey (Queen's University)", "Josh Rosenfeld (Queen's University)"]
        self.parent.helpText = """
This module converts a collection of images and segmentations into a synchronized sequence and incorporates bounding box annotations.
"""

#
# generateSequencesParameterNode
#


@parameterNodeWrapper
class generateSequencesWithBoundingBoxesParameterNode:
    pass


#
# generateSequencesWithBoundingBoxesWidget
#

class generateSequencesWithBoundingBoxesWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        # needed for parameter node observation
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath(
            'UI/generateSequencesWithBoundingBoxes.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = generateSequencesWithBoundingBoxesLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene,
                         slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.generateSequenceButton.connect(
            'clicked(bool)', self.onGenerateSequenceButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[generateSequencesWithBoundingBoxesParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

    def onGenerateSequenceButton(self) -> None:
        self.logic.generateSequence(self.ui.imageDirectoryButton.directory)
#
# generateSequencesWithBoundingBoxesLogic
#


class generateSequencesWithBoundingBoxesLogic(ScriptedLoadableModuleLogic):

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.segLogic = slicer.modules.segmentations.logic()
        self.sequenceObserver = None

    def getParameterNode(self):
        return generateSequencesWithBoundingBoxesParameterNode(super().getParameterNode())

    def createImageNode(self):
        self.imageNode = slicer.vtkMRMLVectorVolumeNode()
        self.imageNode.SetName('RGB_Image')
        slicer.mrmlScene.AddNode(self.imageNode)
        displayNode = slicer.vtkMRMLVectorVolumeDisplayNode()
        slicer.mrmlScene.AddNode(displayNode)
        self.imageNode.SetAndObserveDisplayNodeID(displayNode.GetID())

    def createMarkupsNode(self):
        self.markupsNode = slicer.vtkMRMLMarkupsFiducialNode()
        self.markupsNode.SetName('Markups')
        slicer.mrmlScene.AddNode(self.markupsNode)
        displayNode = slicer.vtkMRMLMarkupsFiducialDisplayNode()
        slicer.mrmlScene.AddNode(displayNode)
        self.markupsNode.SetAndObserveDisplayNodeID(displayNode.GetID())

    def createSequenceBrowser(self):
        seqBrow = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSequenceBrowserNode")
        seqBrow.SetName("SequenceBrowser")
        seqBrow.SetAndObserveMasterSequenceNodeID(self.imageSequence.GetID())
        seqBrow.AddSynchronizedSequenceNode(self.markupsSequence)

    def loadImageVolume(self, img_filepath, timeRecorded):
        imageVol = slicer.util.loadVolume(
            img_filepath, properties={"singleFile": True})
        self.imageNode.CopyContent(imageVol, True)
        self.imageSequence.SetDataNodeAtValue(
            self.imageNode, str(timeRecorded))
        slicer.mrmlScene.RemoveNode(imageVol)

    def clearAllSegments(self):
        self.markupsNode.GetSegmentation().RemoveAllSegments()

    def loadBoundingBoxes(self, boundingBoxes, timeRecorded):
        # Set position of 4 markup points within markupsNode to US bbox coords
        self.markupsNode
        self.segmentationSequence.SetDataNodeAtValue(
            self.markupsNode, str(timeRecorded))

    def getRecordingSlider(self):
        mainWindow = slicer.util.mainWindow()
        sequenceSeekWidget = mainWindow.findChildren(
            "qMRMLSequenceBrowserSeekWidget")[0]
        self.sequenceSlider = sequenceSeekWidget.findChildren("QSlider")[0]
        '''if self.sequenceObserver is None:
            self.sequenceSlider.connect("valueChanged(int)",self.selectFirstSegment)'''

    def selectFirstSegment(self):
        mainWindow = slicer.util.mainWindow()
        segmentEditorWidget = mainWindow.findChildren(
            "qMRMLSegmentEditorWidget")[0]
        currentSegmentIDs = self.markupsNode.GetSegmentation().GetSegmentIDs()
        print(currentSegmentIDs)
        segmentEditorWidget.setCurrentSegmentID(currentSegmentIDs[0])

    def updateSequence(self, caller, eventid):
        self.getRecordingSlider()
        timeLabel = self.sequenceSlider.value
        timeRecorded = self.segmentationSequence.GetNthIndexValue(timeLabel)
        self.segmentationSequence.SetDataNodeAtValue(
            self.markupsNode, timeRecorded)
        # self.saveUpdatedImage()

    def generateSequence(self, image_directory):
        self.image_directory = image_directory
        self.createImageNode()
        self.createMarkupsNode()
        subtype = os.path.basename(image_directory)
        video_ID = os.path.basename(os.path.dirname(image_directory))
        labelFilePath = os.path.join(
            image_directory, "{}_{}_Labels.csv".format(video_ID, subtype))
        self.labelFile = pandas.read_csv(labelFilePath)
        self.imageSequence = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSequenceNode")
        self.imageSequence.SetName("ImageSequence")
        self.markupsSequence = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSequenceNode")
        self.markupsSequence.SetName("MarkupsSequence")
        self.createSequenceBrowser()
        for i in self.labelFile.index:
            img_filename = self.labelFile["FileName"][i]
            timeRecorded = self.labelFile["Time Recorded"][i]
            bboxes = self.labelFile["Tool bounding box"][i]
            self.loadImageVolume(os.path.join(
                image_directory, img_filename), timeRecorded)
            self.loadBoundingBoxes(os.path.join(
                image_directory, bboxes), timeRecorded)

            # TODO: plot fiducials for each corner of ultra sound bounding box
        # self.clearAllSegments()
        seqBrow = slicer.util.getNode("SequenceBrowser")
        seqBrow.AddProxyNode(self.imageNode, self.imageSequence)
        # self.segmentationNode = slicer.util.getFirstNodeByClassByName("vtkMRMLSegmentationNode","SegmentationSequence")
        self.imageNode = slicer.util.getFirstNodeByClassByName(
            "vtkMRMLVectorVolumeNode", "ImageSequence")
        # self.segmentation = self.segmentationNode.GetSegmentation()
        # self.segmentObserver = self.segmentation.AddObserver(self.segmentation.SegmentModified,self.updateSequence)
        # self.segmentationObserver = self.segmentation.AddObserver(self.segmentation.RepresentationModified, self.updateSequence)


#
# generateSequencesWithBoundingBoxesTest
#

class generateSequencesWithBoundingBoxesTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
