###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# In addition, as a special exception, the copyright holders of
# ilastik give you permission to combine ilastik with applets,
# workflows and plugins which are not covered under the GNU
# General Public License.
#
# See the LICENSE file for details. License information is also available
# on the ilastik web site at:
#		   http://ilastik.org/license.html
###############################################################################

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Oct 23, 2014 16:26:43 EDT$"



import os

import numpy

import vigra

import PyQt4
from PyQt4 import uic, QtCore
from PyQt4.QtGui import QColor
from PyQt4.QtCore import Qt

from ilastik.applets.layerViewer.layerViewerGui import LayerViewerGui
from ilastik.utility.gui import threadRouted

from lazyflow.request import Request

from volumina.pixelpipeline.datasources import ConstantSource, ArraySource
from volumina.layer import ColortableLayer
from volumina.colortables import jet


class NanshePostprocessingGui(LayerViewerGui):
    """
    Simple example of an applet tha
    """

    ###########################################
    ### AppletGuiInterface Concrete Methods ###
    ###########################################

    def appletDrawer(self):
        return self._drawer

    # (Other methods already provided by our base class)

    ###########################################
    ###########################################

    def __init__(self, parentApplet, topLevelOperatorView):
        """
        """
        self.topLevelOperatorView = topLevelOperatorView
        self.ndim = 0
        self.result = None

        super(NanshePostprocessingGui, self).__init__(parentApplet, self.topLevelOperatorView)

        self._register_notify_dirty()

    def initAppletDrawerUi(self):
        # Load the ui file (find it in our own directory)
        localDir = os.path.split(__file__)[0]
        self._drawer = uic.loadUi(localDir+"/drawer.ui")

        # Initialize the gui with the operator's current values
        self.apply_operator_settings_to_gui()

        self.applyAcceptedRegionShapeConstraints_MajorAxisLength_MinEnabled(self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min_Enabled.value)
        self.applyAcceptedRegionShapeConstraints_MajorAxisLength_MaxEnabled(self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max_Enabled.value)
        self.applyAcceptedNeuronShapeConstraints_Area_MinEnabled(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min_Enabled.value)
        self.applyAcceptedNeuronShapeConstraints_Area_MaxEnabled(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max_Enabled.value)
        self.applyAcceptedNeuronShapeConstraints_Eccentricity_MinEnabled(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min_Enabled.value)
        self.applyAcceptedNeuronShapeConstraints_Eccentricity_MaxEnabled(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max_Enabled.value)

        # Add handlers for different selection events

        self._drawer.Apply.clicked.connect(self.apply_gui_settings_to_operator)

        self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MinEnabled.clicked.connect(
            self.applyAcceptedRegionShapeConstraints_MajorAxisLength_MinEnabled
        )

        self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MaxEnabled.clicked.connect(
            self.applyAcceptedRegionShapeConstraints_MajorAxisLength_MaxEnabled
        )

        self._drawer.AcceptedNeuronShapeConstraints_Area_MinEnabled.clicked.connect(
            self.applyAcceptedNeuronShapeConstraints_Area_MinEnabled
        )

        self._drawer.AcceptedNeuronShapeConstraints_Area_MaxEnabled.clicked.connect(
            self.applyAcceptedNeuronShapeConstraints_Area_MaxEnabled
        )

        self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MinEnabled.clicked.connect(
            self.applyAcceptedNeuronShapeConstraints_Eccentricity_MinEnabled
        )

        self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MaxEnabled.clicked.connect(
            self.applyAcceptedNeuronShapeConstraints_Eccentricity_MaxEnabled
        )

    def _register_notify_dirty(self):
        self.topLevelOperatorView.SignificanceThreshold.notifyDirty(self.apply_dirty_operator_settings_to_gui)
        self.topLevelOperatorView.WaveletTransformScale.notifyDirty(self.apply_dirty_operator_settings_to_gui)
        self.topLevelOperatorView.NoiseThreshold.notifyDirty(self.apply_dirty_operator_settings_to_gui)
        self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min.notifyDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min_Enabled.notifyDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max.notifyDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max_Enabled.notifyDirty(
            self.apply_dirty_operator_settings_to_gui
        )

        self.topLevelOperatorView.PercentagePixelsBelowMax.notifyDirty(self.apply_dirty_operator_settings_to_gui)
        self.topLevelOperatorView.MinLocalMaxDistance.notifyDirty(self.apply_dirty_operator_settings_to_gui)

        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min.notifyDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min_Enabled.notifyDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max.notifyDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max_Enabled.notifyDirty(
            self.apply_dirty_operator_settings_to_gui
        )

        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min.notifyDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min_Enabled.notifyDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max.notifyDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max_Enabled.notifyDirty(
            self.apply_dirty_operator_settings_to_gui
        )

        self.topLevelOperatorView.AlignmentMinThreshold.notifyDirty(self.apply_dirty_operator_settings_to_gui)
        self.topLevelOperatorView.OverlapMinThreshold.notifyDirty(self.apply_dirty_operator_settings_to_gui)

        self.topLevelOperatorView.Fuse_FractionMeanNeuronMaxThreshold.notifyDirty(self.apply_dirty_operator_settings_to_gui)

    def _unregister_notify_dirty(self):
        self.topLevelOperatorView.SignificanceThreshold.unregisterDirty(self.apply_dirty_operator_settings_to_gui)
        self.topLevelOperatorView.WaveletTransformScale.unregisterDirty(self.apply_dirty_operator_settings_to_gui)
        self.topLevelOperatorView.NoiseThreshold.unregisterDirty(self.apply_dirty_operator_settings_to_gui)
        self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min.unregisterDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min_Enabled.unregisterDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max.unregisterDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max_Enabled.unregisterDirty(
            self.apply_dirty_operator_settings_to_gui
        )

        self.topLevelOperatorView.PercentagePixelsBelowMax.unregisterDirty(self.apply_dirty_operator_settings_to_gui)
        self.topLevelOperatorView.MinLocalMaxDistance.unregisterDirty(self.apply_dirty_operator_settings_to_gui)

        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min.unregisterDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min_Enabled.unregisterDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max.unregisterDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max_Enabled.unregisterDirty(
            self.apply_dirty_operator_settings_to_gui
        )

        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min.unregisterDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min_Enabled.unregisterDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max.unregisterDirty(
            self.apply_dirty_operator_settings_to_gui
        )
        self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max_Enabled.unregisterDirty(
            self.apply_dirty_operator_settings_to_gui
        )

        self.topLevelOperatorView.AlignmentMinThreshold.unregisterDirty(self.apply_dirty_operator_settings_to_gui)
        self.topLevelOperatorView.OverlapMinThreshold.unregisterDirty(self.apply_dirty_operator_settings_to_gui)

        self.topLevelOperatorView.Fuse_FractionMeanNeuronMaxThreshold.unregisterDirty(self.apply_dirty_operator_settings_to_gui)

    def applyAcceptedRegionShapeConstraints_MajorAxisLength_MinEnabled(self, checked):
        self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MinValue.setEnabled(checked)

    def applyAcceptedRegionShapeConstraints_MajorAxisLength_MaxEnabled(self, checked):
        self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MaxValue.setEnabled(checked)

    def applyAcceptedNeuronShapeConstraints_Area_MinEnabled(self, checked):
        self._drawer.AcceptedNeuronShapeConstraints_Area_MinValue.setEnabled(checked)

    def applyAcceptedNeuronShapeConstraints_Area_MaxEnabled(self, checked):
        self._drawer.AcceptedNeuronShapeConstraints_Area_MaxValue.setEnabled(checked)

    def applyAcceptedNeuronShapeConstraints_Eccentricity_MinEnabled(self, checked):
        self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MinValue.setEnabled(checked)

    def applyAcceptedNeuronShapeConstraints_Eccentricity_MaxEnabled(self, checked):
        self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MaxValue.setEnabled(checked)

    def apply_operator_settings_to_gui(self):
        self.ndim = len(self.topLevelOperatorView.Input.meta.shape) - 1

        self._unregister_notify_dirty()

        # Convert a single value or singleton list into a list of values equal to the number of dimensions
        if not isinstance(self.topLevelOperatorView.WaveletTransformScale.value, (list, tuple)):
            self.topLevelOperatorView.WaveletTransformScale.setValue(self.ndim*[self.topLevelOperatorView.WaveletTransformScale.value])
        elif len(self.topLevelOperatorView.WaveletTransformScale.value) == 1:
            self.topLevelOperatorView.WaveletTransformScale.setValue(self.ndim*[self.topLevelOperatorView.WaveletTransformScale.value[0]])

        self._register_notify_dirty()

        assert(2 <= self.ndim <= 3)

        if self.ndim == 2:
            self._drawer.ScaleValue_Z.hide()

            self._drawer.ScaleLabel.setText(self._drawer.ScaleLabel.text().replace("Z, ", ""))


            self._drawer.SignificanceThresholdValue.setValue(self.topLevelOperatorView.SignificanceThreshold.value)

            self._drawer.ScaleValue_Y.setValue(self.topLevelOperatorView.WaveletTransformScale.value[0])
            self._drawer.ScaleValue_X.setValue(self.topLevelOperatorView.WaveletTransformScale.value[1])

            self._drawer.NoiseThresholdValue.setValue(self.topLevelOperatorView.NoiseThreshold.value)


            self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MinValue.setValue(self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min.value)
            self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MinEnabled.setChecked(self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min_Enabled.value)

            self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MaxValue.setValue(self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max.value)
            self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MaxEnabled.setChecked(self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max_Enabled.value)


            self._drawer.PercentagePixelsBelowMaxValue.setValue(self.topLevelOperatorView.PercentagePixelsBelowMax.value)
            self._drawer.MinLocalMaxDistanceValue.setValue(self.topLevelOperatorView.MinLocalMaxDistance.value)


            self._drawer.AcceptedNeuronShapeConstraints_Area_MinValue.setValue(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min.value)
            self._drawer.AcceptedNeuronShapeConstraints_Area_MinEnabled.setChecked(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min_Enabled.value)

            self._drawer.AcceptedNeuronShapeConstraints_Area_MaxValue.setValue(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max.value)
            self._drawer.AcceptedNeuronShapeConstraints_Area_MaxEnabled.setChecked(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max_Enabled.value)

            self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MinValue.setValue(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min.value)
            self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MinEnabled.setChecked(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min_Enabled.value)

            self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MaxValue.setValue(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max.value)
            self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MaxEnabled.setChecked(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max_Enabled.value)


            self._drawer.AlignmentMinThresholdValue.setValue(self.topLevelOperatorView.AlignmentMinThreshold.value)
            self._drawer.OverlapMinThresholdValue.setValue(self.topLevelOperatorView.OverlapMinThreshold.value)
            self._drawer.FuseFractionMeanNeuronMaxThresholdValue.setValue(self.topLevelOperatorView.Fuse_FractionMeanNeuronMaxThreshold.value)
        elif self.ndim == 3:
            self._drawer.ScaleValue_Z.show()

            if "Z, " not in self._drawer.ScaleLabel.text():
                self._drawer.ScaleLabel.setText(self._drawer.ScaleLabel.text().replace("Y, ", "Z, Y, "))


            self._drawer.SignificanceThresholdValue.setValue(self.topLevelOperatorView.SignificanceThreshold.value)

            self._drawer.ScaleValue_Z.setValue(self.topLevelOperatorView.WaveletTransformScale.value[0])
            self._drawer.ScaleValue_Y.setValue(self.topLevelOperatorView.WaveletTransformScale.value[1])
            self._drawer.ScaleValue_X.setValue(self.topLevelOperatorView.WaveletTransformScale.value[2])

            self._drawer.NoiseThresholdValue.setValue(self.topLevelOperatorView.NoiseThreshold.value)


            self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MinValue.setValue(self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min.value)
            self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MinEnabled.setChecked(self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min_Enabled.value)

            self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MaxValue.setValue(self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max.value)
            self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MaxEnabled.setChecked(self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max_Enabled.value)


            self._drawer.PercentagePixelsBelowMaxValue.setValue(self.topLevelOperatorView.PercentagePixelsBelowMax.value)
            self._drawer.MinLocalMaxDistanceValue.setValue(self.topLevelOperatorView.MinLocalMaxDistance.value)


            self._drawer.AcceptedNeuronShapeConstraints_Area_MinValue.setValue(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min.value)
            self._drawer.AcceptedNeuronShapeConstraints_Area_MinEnabled.setChecked(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min_Enabled.value)

            self._drawer.AcceptedNeuronShapeConstraints_Area_MaxValue.setValue(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max.value)
            self._drawer.AcceptedNeuronShapeConstraints_Area_MaxEnabled.setChecked(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max_Enabled.value)

            self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MinValue.setValue(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min.value)
            self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MinEnabled.setChecked(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min_Enabled.value)

            self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MaxValue.setValue(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max.value)
            self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MaxEnabled.setChecked(self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max_Enabled.value)


            self._drawer.AlignmentMinThresholdValue.setValue(self.topLevelOperatorView.AlignmentMinThreshold.value)
            self._drawer.OverlapMinThresholdValue.setValue(self.topLevelOperatorView.OverlapMinThreshold.value)
            self._drawer.FuseFractionMeanNeuronMaxThresholdValue.setValue(self.topLevelOperatorView.Fuse_FractionMeanNeuronMaxThreshold.value)

    def apply_dirty_operator_settings_to_gui(self, slot, roi, **kwargs):
        self.apply_operator_settings_to_gui()

    def apply_gui_settings_to_operator(self):
        self._unregister_notify_dirty()

        if self.ndim == 2:
            self._drawer.ScaleValue_Z.hide()

            self._drawer.ScaleLabel.setText(self._drawer.ScaleLabel.text().replace("Z, ", ""))


            self.topLevelOperatorView.SignificanceThreshold.setValue(self._drawer.SignificanceThresholdValue.value())

            self.topLevelOperatorView.WaveletTransformScale.setValue([self._drawer.ScaleValue_Y.value(), self._drawer.ScaleValue_X.value()])

            self.topLevelOperatorView.NoiseThreshold.setValue(self._drawer.NoiseThresholdValue.value())


            self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min.setValue(self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MinValue.value())
            self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min_Enabled.setValue(self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MinEnabled.isChecked())

            self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max.setValue(self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MaxValue.value())
            self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max_Enabled.setValue(self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MaxEnabled.isChecked())


            self.topLevelOperatorView.PercentagePixelsBelowMax.setValue(self._drawer.PercentagePixelsBelowMaxValue.value())
            self.topLevelOperatorView.MinLocalMaxDistance.setValue(self._drawer.MinLocalMaxDistanceValue.value())


            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min.setValue(self._drawer.AcceptedNeuronShapeConstraints_Area_MinValue.value())
            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min_Enabled.setValue(self._drawer.AcceptedNeuronShapeConstraints_Area_MinEnabled.isChecked())

            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max.setValue(self._drawer.AcceptedNeuronShapeConstraints_Area_MaxValue.value())
            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max_Enabled.setValue(self._drawer.AcceptedNeuronShapeConstraints_Area_MaxEnabled.isChecked())

            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min.setValue(self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MinValue.value())
            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min_Enabled.setValue(self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MinEnabled.isChecked())

            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max.setValue(self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MaxValue.value())
            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max_Enabled.setValue(self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MaxEnabled.isChecked())


            self.topLevelOperatorView.AlignmentMinThreshold.setValue(self._drawer.AlignmentMinThresholdValue.value())
            self.topLevelOperatorView.OverlapMinThreshold.setValue(self._drawer.OverlapMinThresholdValue.value())
            self.topLevelOperatorView.Fuse_FractionMeanNeuronMaxThreshold.setValue(self._drawer.FuseFractionMeanNeuronMaxThresholdValue.value())

        elif self.ndim == 3:
            self._drawer.ScaleValue_Z.show()

            if "Z, " not in self._drawer.ScaleLabel.text():
                self._drawer.ScaleLabel.setText(self._drawer.ScaleLabel.text().replace("Y, ", "Z, Y, "))


            self.topLevelOperatorView.SignificanceThreshold.setValue(self._drawer.SignificanceThresholdValue.value())

            self.topLevelOperatorView.WaveletTransformScale.setValue([self._drawer.ScaleValue_Z.value(), self._drawer.ScaleValue_Y.value(), self._drawer.ScaleValue_X.value()])

            self.topLevelOperatorView.NoiseThreshold.setValue(self._drawer.NoiseThresholdValue.value())


            self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min.setValue(self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MinValue.value())
            self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Min_Enabled.setValue(self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MinEnabled.isChecked())

            self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max.setValue(self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MaxValue.value())
            self.topLevelOperatorView.AcceptedRegionShapeConstraints_MajorAxisLength_Max_Enabled.setValue(self._drawer.AcceptedRegionShapeConstraints_MajorAxisLength_MaxEnabled.isChecked())


            self.topLevelOperatorView.PercentagePixelsBelowMax.setValue(self._drawer.PercentagePixelsBelowMaxValue.value())
            self.topLevelOperatorView.MinLocalMaxDistance.setValue(self._drawer.MinLocalMaxDistanceValue.value())


            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min.setValue(self._drawer.AcceptedNeuronShapeConstraints_Area_MinValue.value())
            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Min_Enabled.setValue(self._drawer.AcceptedNeuronShapeConstraints_Area_MinEnabled.isChecked())

            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max.setValue(self._drawer.AcceptedNeuronShapeConstraints_Area_MaxValue.value())
            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Area_Max_Enabled.setValue(self._drawer.AcceptedNeuronShapeConstraints_Area_MaxEnabled.isChecked())

            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min.setValue(self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MinValue.value())
            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Min_Enabled.setValue(self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MinEnabled.isChecked())

            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max.setValue(self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MaxValue.value())
            self.topLevelOperatorView.AcceptedNeuronShapeConstraints_Eccentricity_Max_Enabled.setValue(self._drawer.AcceptedNeuronShapeConstraints_Eccentricity_MaxEnabled.isChecked())


            self.topLevelOperatorView.AlignmentMinThreshold.setValue(self._drawer.AlignmentMinThresholdValue.value())
            self.topLevelOperatorView.OverlapMinThreshold.setValue(self._drawer.OverlapMinThresholdValue.value())
            self.topLevelOperatorView.Fuse_FractionMeanNeuronMaxThreshold.setValue(self._drawer.FuseFractionMeanNeuronMaxThresholdValue.value())

        self._register_notify_dirty()


        req = Request(lambda : NanshePostprocessingGui.compute_neurons(self.topLevelOperatorView.Output))
        req.notify_finished(self.handle_the_result)
        req.submit()

        self.updateAllLayers()

        for i in xrange(len(self.layerstack)):
            if self.layerstack[i].name == "Output":
                self.layerstack[i].visible = True

    @staticmethod
    def compute_neurons(outputImageSlot):
        neurons = None

        if outputImageSlot.ready():
            neurons = outputImageSlot[slice(None)].wait()
            neurons = neurons[0]

            if not len(neurons):
                neurons = vigra.taggedView(numpy.zeros((1,) + neurons.shape[1:], dtype=neurons.dtype), neurons.axistags)

            neurons = neurons.withAxes(*[_ for _ in "txyzc"])

        return(neurons)

    @threadRouted
    def handle_the_result(self, result):
        self.result = result
        self.updateAllLayers()

    def setupLayers(self):
        """
        Overridden from LayerViewerGui.
        Create a list of all layer objects that should be displayed.
        """
        layers = []

        if self.result is not None:
            outputSource = ArraySource(self.result)

            neuron_colors = [QColor(0, 0, 0, 0).rgba(),
                     QColor(0, 0, 255).rgba()]#,
                     # QColor(255, 255, 0).rgba(),
                     # QColor(255, 0, 0).rgba(),
                     # QColor(0, 255, 0).rgba(),
                     # QColor(0, 255, 255).rgba(),
                     # QColor(255, 0, 255).rgba(),
                     # QColor(255, 105, 180).rgba(), #hot pink
                     # QColor(102, 205, 170).rgba(), #dark aquamarine
                     # QColor(165,  42,  42).rgba(), #brown
                     # QColor(0, 0, 128).rgba(),     #navy
                     # QColor(255, 165, 0).rgba(),   #orange
                     # QColor(173, 255,  47).rgba(), #green-yellow
                     # QColor(128,0, 128).rgba(),    #purple
                     # QColor(192, 192, 192).rgba(), #silver
                     # QColor(240, 230, 140).rgba(), #khaki
                     # QColor(69, 69, 69).rgba()]    # dark grey

            outputLayer = ColortableLayer(outputSource, neuron_colors)
            outputLayer.name = "Output"
            outputLayer.visible = False
            outputLayer.opacity = 1.0
            layers.append(outputLayer)

        # # Show the resulting label image
        # outputImageSlot = self.topLevelOperatorView.ColorizedOutput
        # if outputImageSlot.ready():
        #     outputLayer = self.createStandardLayerFromSlot( outputImageSlot, lastChannelIsAlpha=True )
        #     outputLayer.name = "OutputLabelImage"
        #     outputLayer.visible = False
        #     outputLayer.opacity = 1.0
        #     layers.append(outputLayer)

        # Show the input data
        inputSlot = self.topLevelOperatorView.Input
        if inputSlot.ready():
            inputLayer = self.createStandardLayerFromSlot( inputSlot )
            inputLayer.name = "Input"
            inputLayer.visible = True
            inputLayer.opacity = 1.0
            layers.append(inputLayer)

        return layers
