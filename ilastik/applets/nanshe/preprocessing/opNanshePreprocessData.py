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



import numpy

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.operators import OpArrayCache

from ilastik.applets.nanshe.opConvertType import OpConvertType, OpConvertTypeCached
from ilastik.applets.nanshe.preprocessing.opNansheRegisterMeanOffsets import OpNansheRegisterMeanOffsets, OpNansheRegisterMeanOffsetsCached
from ilastik.applets.nanshe.preprocessing.opNansheRemoveZeroedLines import OpNansheRemoveZeroedLines, OpNansheRemoveZeroedLinesCached
from ilastik.applets.nanshe.preprocessing.opNansheExtractF0 import OpNansheExtractF0, OpNansheExtractF0Cached
from ilastik.applets.nanshe.preprocessing.opNansheWaveletTransform import OpNansheWaveletTransform, OpNansheWaveletTransformCached


class OpNanshePreprocessData(Operator):
    """
    Given an input image and max/min bounds,
    masks out (i.e. sets to zero) all pixels that fall outside the bounds.
    """
    name = "OpNanshePreprocessData"
    category = "Pointwise"

    
    Input = InputSlot(allow_mask=True)

    ToRegisterMeanOffsets = InputSlot(value=False)

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


    OpNansheRegisterMeanOffsetsOutput = OutputSlot(allow_mask=True)
    OpNansheRemoveZeroedLinesOutput = OutputSlot(allow_mask=True)
    OpNansheExtractF0_dF_F_Output = OutputSlot(allow_mask=True)
    OpNansheExtractF0_F0_Output = OutputSlot(allow_mask=True)
    OpNansheWaveletTransformOutput = OutputSlot(allow_mask=True)

    Output = OutputSlot(allow_mask=True)

    def __init__(self, *args, **kwargs):
        super( OpNanshePreprocessData, self ).__init__( *args, **kwargs )

        self.opConvertType = OpConvertType(parent=self)
        self.opConvertType.Dtype.setValue(numpy.float32)

        self.opNansheRegisterMeanOffsets = OpNansheRegisterMeanOffsets(parent=self)

        self.opNansheRemoveZeroedLines = OpNansheRemoveZeroedLines(parent=self)
        self.opNansheRemoveZeroedLines.ErosionShape.connect(self.ErosionShape)
        self.opNansheRemoveZeroedLines.DilationShape.connect(self.DilationShape)

        self.opNansheExtractF0 = OpNansheExtractF0(parent=self)
        self.opNansheExtractF0.HalfWindowSize.connect(self.HalfWindowSize)
        self.opNansheExtractF0.WhichQuantile.connect(self.WhichQuantile)
        self.opNansheExtractF0.TemporalSmoothingGaussianFilterStdev.connect(self.TemporalSmoothingGaussianFilterStdev)
        self.opNansheExtractF0.SpatialSmoothingGaussianFilterStdev.connect(self.SpatialSmoothingGaussianFilterStdev)
        self.opNansheExtractF0.TemporalSmoothingGaussianFilterWindowSize.connect(self.TemporalSmoothingGaussianFilterWindowSize)
        self.opNansheExtractF0.SpatialSmoothingGaussianFilterWindowSize.connect(self.SpatialSmoothingGaussianFilterWindowSize)
        self.opNansheExtractF0.BiasEnabled.connect(self.BiasEnabled)
        self.opNansheExtractF0.Bias.connect(self.Bias)

        self.opNansheWaveletTransform = OpNansheWaveletTransform(parent=self)
        self.opNansheWaveletTransform.Scale.connect(self.Scale)


        self.OpNansheRegisterMeanOffsetsOutput.connect(self.opNansheRegisterMeanOffsets.Output)
        self.OpNansheRemoveZeroedLinesOutput.connect(self.opNansheRemoveZeroedLines.Output)
        self.OpNansheExtractF0_dF_F_Output.connect(self.opNansheExtractF0.dF_F)
        self.OpNansheExtractF0_F0_Output.connect(self.opNansheExtractF0.F0)
        self.OpNansheWaveletTransformOutput.connect(self.opNansheWaveletTransform.Output)
    
    def setupOutputs(self):
        self.opNansheRemoveZeroedLines.Input.disconnect()
        self.opNansheExtractF0.Input.disconnect()
        self.opNansheWaveletTransform.Input.disconnect()

        next_output = self.Input

        self.opConvertType.Input.connect(next_output)
        next_output = self.opConvertType.Output

        if self.ToRegisterMeanOffsets.value:
            self.opNansheRegisterMeanOffsets.Input.connect(next_output)
            next_output = self.opNansheRegisterMeanOffsets.Output

        if self.ToRemoveZeroedLines.value:
            self.opNansheRemoveZeroedLines.Input.connect(next_output)
            next_output = self.opNansheRemoveZeroedLines.Output

        if self.ToExtractF0.value:
            self.opNansheExtractF0.Input.connect(next_output)
            next_output = self.opNansheExtractF0.dF_F

        if self.ToWaveletTransform.value:
            self.opNansheWaveletTransform.Input.connect(next_output)
            next_output = self.opNansheWaveletTransform.Output

        self.Output.connect(next_output)

    # Don't need execute as the output will be drawn through the Output slot.

    def setInSlot(self, slot, subindex, key, value):
        pass

    def propagateDirty(self, slot, subindex, roi):
        if slot.name == "ToRemoveZeroedLines":
            if slot.value:
                self.opNansheRemoveZeroedLines.Output.setDirty( slice(None) )
            else:
                if self.ToExtractF0.value:
                    self.opNansheExtractF0.Input.setDirty( slice(None) )
                elif self.ToWaveletTransform.value:
                    self.opNansheWaveletTransform.Input.setDirty( slice(None) )
                else:
                    self.Output.setDirty( slice(None) )
        elif slot.name == "ToExtractF0":
            if slot.value:
                self.opNansheExtractF0.Output.setDirty( slice(None) )
            else:
                if self.ToWaveletTransform.value:
                    self.opNansheWaveletTransform.Input.setDirty( slice(None) )
                else:
                    self.Output.setDirty( slice(None) )
        elif slot.name == "ToWaveletTransform":
            if slot.value:
                self.opNansheWaveletTransform.Output.setDirty( slice(None) )
            else:
                self.Output.setDirty( slice(None) )


class OpNanshePreprocessDataCached(Operator):
    """
    Given an input image and max/min bounds,
    masks out (i.e. sets to zero) all pixels that fall outside the bounds.
    """
    name = "OpNanshePreprocessDataCached"
    category = "Pointwise"


    Input = InputSlot()
    CacheInput = InputSlot()


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


    OpNansheRegisterMeanOffsetsOutput = OutputSlot(allow_mask=True)
    OpNansheRemoveZeroedLinesOutput = OutputSlot(allow_mask=True)
    OpNansheExtractF0_dF_F_Output = OutputSlot(allow_mask=True)
    OpNansheExtractF0_F0_Output = OutputSlot(allow_mask=True)
    OpNansheWaveletTransformOutput = OutputSlot(allow_mask=True)

    CleanBlocks = OutputSlot()
    CacheOutput = OutputSlot(allow_mask=True)
    Output = OutputSlot(allow_mask=True)

    def __init__(self, *args, **kwargs):
        super( OpNanshePreprocessDataCached, self ).__init__( *args, **kwargs )

        self.opConvertType = OpConvertTypeCached(parent=self)
        self.opConvertType.Dtype.setValue(numpy.float32)

        self.opNansheRegisterMeanOffsets = OpNansheRegisterMeanOffsets(parent=self)

        self.opNansheRemoveZeroedLines = OpNansheRemoveZeroedLinesCached(parent=self)
        self.opNansheRemoveZeroedLines.ErosionShape.connect(self.ErosionShape)
        self.opNansheRemoveZeroedLines.DilationShape.connect(self.DilationShape)

        self.opNansheExtractF0 = OpNansheExtractF0Cached(parent=self)
        self.opNansheExtractF0.HalfWindowSize.connect(self.HalfWindowSize)
        self.opNansheExtractF0.WhichQuantile.connect(self.WhichQuantile)
        self.opNansheExtractF0.TemporalSmoothingGaussianFilterStdev.connect(self.TemporalSmoothingGaussianFilterStdev)
        self.opNansheExtractF0.SpatialSmoothingGaussianFilterStdev.connect(self.SpatialSmoothingGaussianFilterStdev)
        self.opNansheExtractF0.TemporalSmoothingGaussianFilterWindowSize.connect(self.TemporalSmoothingGaussianFilterWindowSize)
        self.opNansheExtractF0.SpatialSmoothingGaussianFilterWindowSize.connect(self.SpatialSmoothingGaussianFilterWindowSize)
        self.opNansheExtractF0.BiasEnabled.connect(self.BiasEnabled)
        self.opNansheExtractF0.Bias.connect(self.Bias)

        self.opNansheWaveletTransform = OpNansheWaveletTransformCached(parent=self)
        self.opNansheWaveletTransform.Scale.connect(self.Scale)


        self.OpNansheRemoveZeroedLinesOutput.connect(self.opNansheRemoveZeroedLines.Output)
        self.OpNansheExtractF0_dF_F_Output.connect(self.opNansheExtractF0.dF_F)
        self.OpNansheExtractF0_F0_Output.connect(self.opNansheExtractF0.F0)
        self.OpNansheWaveletTransformOutput.connect(self.opNansheWaveletTransform.Output)

        self.opCache = OpArrayCache(parent=self)
        self.opCache.fixAtCurrent.setValue(False)
        self.CleanBlocks.connect(self.opCache.CleanBlocks)

        self.CacheOutput.connect(self.opCache.Output)

    def setupOutputs(self):
        self.opNansheRemoveZeroedLines.Input.disconnect()
        self.opNansheExtractF0.Input.disconnect()
        self.opNansheWaveletTransform.Input.disconnect()
        self.opCache.Input.disconnect()

        next_output = self.Input

        self.opConvertType.Input.connect(next_output)
        next_output = self.opConvertType.Output

        self.opNansheRegisterMeanOffsets.Input.connect(next_output)
        next_output = self.opNansheRegisterMeanOffsets.Output

        if self.ToRemoveZeroedLines.value:
            self.opNansheRemoveZeroedLines.Input.connect(next_output)
            next_output = self.opNansheRemoveZeroedLines.Output

        if self.ToExtractF0.value:
            self.opNansheExtractF0.Input.connect(next_output)
            next_output = self.opNansheExtractF0.dF_F

        if self.ToWaveletTransform.value:
            self.opNansheWaveletTransform.Input.connect(next_output)
            next_output = self.opNansheWaveletTransform.Output

        self.Output.connect(next_output)
        self.opCache.Input.connect(next_output)

        self.opCache.blockShape.setValue( self.opCache.Output.meta.shape )

    # Don't need execute as the output will be drawn through the Output slot.

    def setInSlot(self, slot, subindex, key, value):
        assert slot == self.CacheInput

        self.opCache.setInSlot(self.opCache.Input, subindex, key, value)

    def propagateDirty(self, slot, subindex, roi):
        if slot.name == "ToRemoveZeroedLines":
            if slot.value:
                self.opNansheRemoveZeroedLines.Output.setDirty( slice(None) )
            else:
                if self.ToExtractF0.value:
                    self.opNansheExtractF0.Input.setDirty( slice(None) )
                elif self.ToWaveletTransform.value:
                    self.opNansheWaveletTransform.Input.setDirty( slice(None) )
                else:
                    self.opCache.Input.setDirty( slice(None) )
        elif slot.name == "ToExtractF0":
            if slot.value:
                self.opNansheExtractF0.dF_F.setDirty( slice(None) )
                self.opNansheExtractF0.F0.setDirty( slice(None) )
            else:
                if self.ToWaveletTransform.value:
                    self.opNansheWaveletTransform.Input.setDirty( slice(None) )
                else:
                    self.opCache.Input.setDirty( slice(None) )
        elif slot.name == "ToWaveletTransform":
            if slot.value:
                self.opNansheWaveletTransform.Output.setDirty( slice(None) )
            else:
                self.opCache.Input.setDirty( slice(None) )
