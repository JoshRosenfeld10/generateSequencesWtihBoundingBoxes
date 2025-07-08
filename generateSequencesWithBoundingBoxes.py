import logging
import os
from typing import Annotated, Optional

import numpy as np
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
from vtk.util import numpy_support

from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLMarkupsFiducialNode
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
        self.ui.sequenceBrowserNode.setMRMLScene(slicer.mrmlScene)
        self.ui.depthImageSequenceNode.setMRMLScene(slicer.mrmlScene)

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
        if self.ui.checkBox.checked:
            self.logic.generateUsingExistingSequenceBrowser(
                self.ui.RGBImageDirectoryButton.directory,
                self.ui.sequenceBrowserNode.currentNode(),
                self.ui.depthImageSequenceNode.currentNode()
            )
        else:
            self.logic.generateSequence(
                self.ui.RGBImageDirectoryButton.directory,
                self.ui.depthImageDirectoryButton.directory
            )


#
# generateSequencesWithBoundingBoxesLogic
#


class generateSequencesWithBoundingBoxesLogic(ScriptedLoadableModuleLogic):

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.sequenceObserver = None
        self.markupNodes = []  # list of markup nodes (one for each bounding box)
        self.markupSequences = []  # list of markup sequences (one for each set of markups)
        self.imageNodes = []  # list of image nodes (one for RGB, one for depth)
        self.imageSequences = [] # list of image sequences (one for each image node)
        self.sequenceBrowser = None
        self.depthImageSequenceNode = None

    def getParameterNode(self):
        return generateSequencesWithBoundingBoxesParameterNode(super().getParameterNode())

    def addImageNodeWithSequence(self, classname: str):
        # Create image node
        self.imageNodes.append(slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLVectorVolumeNode", f"{classname.upper()} Image"
        ))

        # Create sequence node for image
        self.imageSequences.append(slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSequenceNode", f"{classname.upper()} Image Sequence"
        ))

    def addMarkupNodeWithSequence(self, classname: str) -> None:
        # Create markup node
        self.markupNodes.append(slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", f"{classname.upper()} Markups"
        ))

        # Initialize 4 control points for bounding box
        for _ in range(4):
            self.markupNodes[-1].AddControlPoint(0.0, 0.0, 0.0)

        # Hide markups (will be visible through sequence markups)
        self.markupNodes[-1].GetDisplayNode().SetVisibility(False)

        # Create sequence node for markups
        self.markupSequences.append(slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSequenceNode", f"{classname.upper()} Markups Sequence"
        ))

    def createSequenceBrowser(self, masterImageSequenceIndex: int) -> None:
        # Create sequence browser
        self.sequenceBrowser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", "SequenceBrowser")

        # Set master sequence
        self.sequenceBrowser.SetAndObserveMasterSequenceNodeID(self.imageSequences[masterImageSequenceIndex].GetID())

        # Add synchronized sequences
        for i, imageSequence in enumerate(self.imageSequences):
            if i != masterImageSequenceIndex: self.sequenceBrowser.AddSynchronizedSequenceNode(imageSequence)
        for markupsSequence in self.markupSequences:
            self.sequenceBrowser.AddSynchronizedSequenceNode(markupsSequence)

    def loadImageVolume(self, imageNode, imageSequence, imgFilepath, timeRecorded):
        imageVol = slicer.util.loadVolume(
            imgFilepath, properties={"singleFile": True}
        )
        imageNode.CopyContent(imageVol, True)
        imageSequence.SetDataNodeAtValue(
            imageNode, str(timeRecorded)
        )
        slicer.mrmlScene.RemoveNode(imageVol)

    def binarySearchTimestamp(self, numberOfFrames, targetTimestamp):
        low = 0
        high = numberOfFrames
        bestIndex = -1

        while low <= high:
            mid = (low + high) // 2
            timestamp = float(self.depthImageSequenceNode.GetNthIndexValue(mid))
            if timestamp == targetTimestamp:
                return mid
            elif timestamp < targetTimestamp:
                bestIndex = mid
                low = mid + 1
            else:
                high = mid - 1

        return bestIndex

    def extractROIFromVTK(self, imageData, xmin, xmax, ymin, ymax, z = 0):
        extractor = vtk.vtkExtractVOI()
        extractor.SetInputData(imageData)
        extractor.SetVOI(xmin, xmax - 1, ymin, ymax - 1, z, z)
        extractor.Update()

        roiImageData = extractor.GetOutput()
        return roiImageData

    def extractROIPixelsAtTimestamp(self, targetTimestamp, quad):
        # Find appropriate frame index (<= timestamp)
        numberOfFrames = self.depthImageSequenceNode.GetNumberOfDataNodes()
        matchedFrameIndex = self.binarySearchTimestamp(numberOfFrames, targetTimestamp)
        if matchedFrameIndex < 0:
            raise ValueError(f"No frame found with timestamp <= {targetTimestamp}")

        # Extract image at closest timestamp
        self.sequenceBrowser.SetSelectedItemNumber(matchedFrameIndex)
        frameVolumeNode = self.depthImageSequenceNode.GetDataNodeAtValue(
            self.depthImageSequenceNode.GetNthIndexValue(matchedFrameIndex)
        )
        imageData = frameVolumeNode.GetImageData()
        xcoords, ycoords = quad[:, 0], quad[:, 1]
        xmin, xmax = np.min(xcoords), np.max(xcoords)
        ymin, ymax = np.min(ycoords), np.max(ycoords)
        roiImageData = self.extractROIFromVTK(imageData, xmin, xmax, ymin, ymax)  # get ROI (bounding box) pixels from vtk data

        # Convert to ROI image data to numpy
        dims = roiImageData.GetDimensions()
        scalars = roiImageData.GetPointData().GetScalars()
        npImage = numpy_support.vtk_to_numpy(scalars).reshape(dims[1], dims[0], -1)  # shape: (rows, cols, channels)
        return npImage

    def getAverageRGBPixel(self, pixels):
        return pixels.mean(axis=0).astype(np.uint8)

    def convertRGBToDepth(self, pixel):
        is_disparity = False
        min_depth = 0.16
        max_depth = 300.0
        min_disparity = 1.0 / max_depth
        max_disparity = 1.0 / min_depth
        r_value = float(pixel[0])
        g_value = float(pixel[1])
        b_value = float(pixel[2])
        depthValue = 0
        hue_value = 0
        if (b_value + g_value + r_value) < 255:
            hue_value = 0
        elif (r_value >= g_value and r_value >= b_value):
            if (g_value >= b_value):
                hue_value = g_value - b_value
            else:
                hue_value = (g_value - b_value) + 1529
        elif (g_value >= r_value and g_value >= b_value):
            hue_value = b_value - r_value + 510
        elif (b_value >= g_value and b_value >= r_value):
            hue_value = r_value - g_value + 1020

        if (hue_value > 0):
            if not is_disparity:
                z_value = ((min_depth + (max_depth - min_depth) * hue_value / 1529.0) + 0.5)
                depthValue = z_value
            else:
                disp_value = min_disparity + (max_disparity - min_disparity) * hue_value / 1529.0
                depthValue = ((1.0 / disp_value) / 1000 + 0.5)
        return depthValue

    def computeAverageDepthFromROI(self, npImage, downsampleFactor = 20):
        # TODO: Make downsampling variable based on size of bounding box
        downsampled = npImage[::downsampleFactor, ::downsampleFactor]  # shape: (rows, cols, 3)
        pixels = downsampled.reshape(-1, 3)  # flatten to (N, 3)
        depths = [self.convertRGBToDepth(pixel.tolist()) for pixel in pixels]
        return np.mean(depths)

    def loadBoundingBoxes(self, boundingBoxes, timeRecorded, fromSequence=True):
        boundingBoxes = self.formatBoundingBoxData(boundingBoxes, fromSequence)  # format bounding box data

        # Loop through all markup nodes
        for i, markupNode in enumerate(self.markupNodes):
            classname = markupNode.GetName().split()[0].lower()

            # If annotation exists for a given tool in the given frame, place markups in corresponding position of bounding box
            if classname in boundingBoxes:
                boundingBox = boundingBoxes[classname]

                # Compute average depth within bounding box
                ROIPixels = self.extractROIPixelsAtTimestamp(float(timeRecorded), np.array(boundingBox).astype(np.int32))
                depthValue = self.computeAverageDepthFromROI(ROIPixels)

                # Set markup points
                markupNode.SetNthControlPointPosition(0, boundingBox[0][0], boundingBox[0][1], depthValue)
                markupNode.SetNthControlPointPosition(1, boundingBox[1][0], boundingBox[1][1], depthValue)
                markupNode.SetNthControlPointPosition(2, boundingBox[2][0], boundingBox[2][1], depthValue)
                markupNode.SetNthControlPointPosition(3, boundingBox[3][0], boundingBox[3][1], depthValue)

                # Save position of markups at specific time in corresponding markups sequence
                self.markupSequences[i].SetDataNodeAtValue(
                    markupNode, str(timeRecorded))

            # If annotation does not exist for a given tool in the given frame, place markups at (0.0,0.0,0.0)
            else:
                bottomleft = [0.0,0.0,0.0]
                bottomright = [0.0,0.0,0.0]
                topleft = [0.0,0.0,0.0]
                topright = [0.0,0.0,0.0]

                markupNode.GetNthControlPointPosition(0, bottomleft)  # bottom left
                markupNode.GetNthControlPointPosition(1, bottomright)  # bottom right
                markupNode.GetNthControlPointPosition(2, topleft)  # top left
                markupNode.GetNthControlPointPosition(3, topleft)  # top right

                # Only set position to (0.0, 0.0, 0.0) if markups aren't already there
                if bottomleft != [0.0, 0.0, 0.0] or bottomright != [0.0, 0.0, 0.0] or topleft != [0.0, 0.0, 0.0] or topright != [0.0, 0.0, 0.0]:
                    markupNode.SetNthControlPointPosition(0, 0.0, 0.0, 0.0)  # bottom left
                    markupNode.SetNthControlPointPosition(1, 0.0, 0.0, 0.0)  # bottom right
                    markupNode.SetNthControlPointPosition(2, 0.0, 0.0, 0.0)  # top left
                    markupNode.SetNthControlPointPosition(3, 0.0, 0.0, 0.0)  # top right

                    # Save position of markups at specific time in corresponding markups sequence
                    self.markupSequences[i].SetDataNodeAtValue(
                        markupNode, str(timeRecorded))


    def formatBoundingBoxData(self, rawBoundingBoxes, fromSequence: bool):
        rawBoundingBoxes = eval(rawBoundingBoxes)  # evaluate string to convert to list of dictionaries
        formattedBoundingBoxes = {}

        for label in rawBoundingBoxes:
            formattedBoundingBoxes[label['class']] = [
                [label['xmin'] * -1, label['ymax'] * -1],  # bottom left
                [label['xmax'] * -1, label['ymax'] * -1],  # bottom right
                [label['xmin'] * -1, label['ymin'] * -1],  # top left
                [label['xmax'] * -1, label['ymin'] * -1],  # top right
            ] if fromSequence else [
                [label['xmin'], label['ymax']],  # top left
                [label['xmax'], label['ymax']],  # top right
                [label['xmin'], label['ymin']],  # bottom left
                [label['xmax'], label['ymin']],  # bottom right
            ]

        return formattedBoundingBoxes

    def getLabelFilePathFromDirectory(self, directory):
        subtype = os.path.basename(directory)
        videoId = os.path.basename(os.path.dirname(directory))

        return os.path.join(
            directory, "{}_{}_Labels.csv".format(videoId, subtype)
        )

    def generateSequence(self, RGBImageDirectory, depthImageDirectory):

        # Create image and markup nodes
        self.addImageNodeWithSequence("RGB")
        self.addImageNodeWithSequence("DEPTH")
        self.addMarkupNodeWithSequence("ULTRASOUND")
        self.addMarkupNodeWithSequence("PHANTOM")
        self.addMarkupNodeWithSequence("SYRINGE")

        # Create sequences node
        self.createSequenceBrowser(0)

        # Read label file from csv
        RGBLabelFile = pandas.read_csv(self.getLabelFilePathFromDirectory(RGBImageDirectory))
        depthLabelFile = pandas.read_csv(self.getLabelFilePathFromDirectory(depthImageDirectory))

        # Loop through each frame of RGB images
        for i in RGBLabelFile.index:

            # Retrieve data from each frame
            imgFilename = RGBLabelFile["FileName"][i]
            timeRecorded = RGBLabelFile["Time Recorded"][i]
            bboxes = RGBLabelFile["Tool bounding box"][i]

            # Load image
            self.loadImageVolume(
                self.imageNodes[0],
                self.imageSequences[0],
                os.path.join(RGBImageDirectory, imgFilename),
                timeRecorded
            )

            # Load bounding box data
            self.loadBoundingBoxes(bboxes, timeRecorded, True)

        # Loop through each frame of depth images
        for i in depthLabelFile.index:

            # Retrieve data from each frame
            imgFilename = depthLabelFile["FileName"][i]
            timeRecorded = depthLabelFile["Time Recorded"][i]

            # Load image
            self.loadImageVolume(
                self.imageNodes[1],
                self.imageSequences[1],
                os.path.join(depthImageDirectory, imgFilename),
                timeRecorded
            )

    def generateUsingExistingSequenceBrowser(self, RGBImageDirectory, sequenceBrowser, depthImageSequenceNode):
        # Set sequence browser
        self.sequenceBrowser = sequenceBrowser
        self.depthImageSequenceNode = depthImageSequenceNode

        # Create markup nodes with sequences
        self.addMarkupNodeWithSequence("ULTRASOUND")
        # self.addMarkupNodeWithSequence("PHANTOM")

        # Add markup sequences to selected sequence browser
        for markupsSequence in self.markupSequences:
            self.sequenceBrowser.AddSynchronizedSequenceNode(markupsSequence)

        # Read label file from csv
        RGBLabelFile = pandas.read_csv(self.getLabelFilePathFromDirectory(RGBImageDirectory))

        # Loop through each frame of RGB images
        for i in RGBLabelFile.index:

            # Retrieve data from each frame
            timeRecorded = RGBLabelFile["Time Recorded"][i]
            bboxes = RGBLabelFile["Tool bounding box"][i]

            # Load bounding box data
            self.loadBoundingBoxes(bboxes, timeRecorded, False)


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
