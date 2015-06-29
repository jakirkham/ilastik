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
__date__ = "$Oct 14, 2014 16:37:05 EDT$"



from lazyflow.graph import Operator, InputSlot, OutputSlot

from ilastik.applets.nanshe.preprocessing.opNanshePreprocessData import OpNanshePreprocessDataCached


class OpNanshePreprocessing(Operator):
    """
    Given an input image and max/min bounds,
    masks out (i.e. sets to zero) all pixels that fall outside the bounds.
    """
    name = "OpNanshePreprocessing"
    category = "Pointwise"


    Input = InputSlot(allow_mask=True)
    CacheInput = InputSlot(optional=True)


    ToRemoveZeroedLines = InputSlot(value=True)
    ErosionShape = InputSlot(value=[21, 1])
    DilationShape = InputSlot(value=[1, 3])

    ToExtractF0 = InputSlot(value=True)
    HalfWindowSize = InputSlot(value=400, stype='int')
    WhichQuantile = InputSlot(value=0.15, stype='float')
    TemporalSmoothingGaussianFilterStdev = InputSlot(value=5.0, stype='float')
    SpatialSmoothingGaussianFilterStdev = InputSlot(value=5.0, stype='float')
    TemporalSmoothingGaussianFilterWindowSize = InputSlot(value=5.0, stype='float')
    SpatialSmoothingGaussianFilterWindowSize = InputSlot(value=5.0, stype='float')
    BiasEnabled = InputSlot(value=False, stype='bool')
    Bias = InputSlot(value=0.0, stype='float')

    ToWaveletTransform = InputSlot(value=True)
    Scale = InputSlot(value=4)


    OpNansheRemoveZeroedLinesOutput = OutputSlot(allow_mask=True)
    OpNansheExtractF0_dF_F_Output = OutputSlot(allow_mask=True)
    OpNansheExtractF0_F0_Output = OutputSlot(allow_mask=True)
    OpNansheWaveletTransformOutput = OutputSlot(allow_mask=True)

    CleanBlocks = OutputSlot()
    CacheOutput = OutputSlot()
    Output = OutputSlot(allow_mask=True)

    def __init__(self, *args, **kwargs):
        super( OpNanshePreprocessing, self ).__init__( *args, **kwargs )

        self.opPreprocessData = OpNanshePreprocessDataCached(parent=self)


        self.opPreprocessData.ToRemoveZeroedLines.connect(self.ToRemoveZeroedLines)
        self.opPreprocessData.ErosionShape.connect(self.ErosionShape)
        self.opPreprocessData.DilationShape.connect(self.DilationShape)

        self.opPreprocessData.ToExtractF0.connect(self.ToExtractF0)
        self.opPreprocessData.HalfWindowSize.connect(self.HalfWindowSize)
        self.opPreprocessData.WhichQuantile.connect(self.WhichQuantile)
        self.opPreprocessData.TemporalSmoothingGaussianFilterStdev.connect(self.TemporalSmoothingGaussianFilterStdev)
        self.opPreprocessData.SpatialSmoothingGaussianFilterStdev.connect(self.SpatialSmoothingGaussianFilterStdev)
        self.opPreprocessData.TemporalSmoothingGaussianFilterWindowSize.connect(self.TemporalSmoothingGaussianFilterWindowSize)
        self.opPreprocessData.SpatialSmoothingGaussianFilterWindowSize.connect(self.SpatialSmoothingGaussianFilterWindowSize)
        self.opPreprocessData.BiasEnabled.connect(self.BiasEnabled)
        self.opPreprocessData.Bias.connect(self.Bias)

        self.opPreprocessData.ToWaveletTransform.connect(self.ToWaveletTransform)
        self.opPreprocessData.Scale.connect(self.Scale)


        self.opPreprocessData.Input.connect( self.Input )
        self.opPreprocessData.CacheInput.connect( self.CacheInput )

        self.OpNansheRemoveZeroedLinesOutput.connect(self.opPreprocessData.OpNansheRemoveZeroedLinesOutput)
        self.OpNansheExtractF0_dF_F_Output.connect(self.opPreprocessData.OpNansheExtractF0_dF_F_Output)
        self.OpNansheExtractF0_F0_Output.connect(self.opPreprocessData.OpNansheExtractF0_F0_Output)
        self.OpNansheWaveletTransformOutput.connect(self.opPreprocessData.OpNansheWaveletTransformOutput)

        self.CleanBlocks.connect( self.opPreprocessData.CleanBlocks )
        self.CacheOutput.connect( self.opPreprocessData.CacheOutput )
        self.Output.connect( self.opPreprocessData.Output )

    def setInSlot(self, slot, subindex, key, value):
        pass

    def propagateDirty(self, slot, subindex, roi):
        pass
